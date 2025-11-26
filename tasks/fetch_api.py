# tasks/fetch_api.py
import requests
from datetime import datetime, timedelta
import pandas as pd
from pandas.errors import EmptyDataError
import os
from zoneinfo import ZoneInfo
from filelock import FileLock, Timeout
import logging
import asyncio
import time
from pathlib import Path
import signal

# ------------------- SHUTDOWN EVENT -------------------
shutdown_event = asyncio.Event()

def signal_handler(sig, frame):
    logger.info("ได้รับสัญญาณหยุด → กำลังปิดอย่างสงบ...")
    shutdown_event.set()

def setup_signal_handlers():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            logger.info("Event loop กำลังรัน → ข้ามการตั้ง signal handlers (uvicorn mode)")
            return
    except Exception:
        pass

    try:
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
        logger.info("ตั้งค่า signal handler สำเร็จ")
    except Exception as e:
        logger.exception(f"setup_signal_handlers skipped: {e}")

# ------------------- CONFIG -------------------
STAGING_DIR = Path("data/staging")
STAGING_DIR.mkdir(parents=True, exist_ok=True)

MASTER_FILE = STAGING_DIR / "raw_airbkk.csv"      # raw จาก API
CLEAN_FILE  = STAGING_DIR / "clean_airbkk.csv"    
LOCK_FILE   = STAGING_DIR / ".airbkk_lock"
LOG_FILE    = STAGING_DIR / "capture.log"

URL = "https://official.airbkk.com/airbkk/Report/getData"
TZ = "Asia/Bangkok"

FETCH_RETRY = int(os.getenv("FETCH_RETRY", "1"))
RETRY_SLEEP_SECONDS = int(os.getenv("RETRY_SLEEP_SECONDS", "120"))

# ------------------- LOGGING -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("airbkk_fetcher")

# ------------------- HELPERS -------------------
def _parse_date_time_be_aware(series: pd.Series) -> pd.Series:
    """
    รับ Series ของ Date_Time ที่เป็น string เช่น '14/11/2568 04:00' หรือ '14/11/2025 04:00'
    - พยายาม parse เป็น datetime
    - ถ้าปีเป็น พ.ศ. (เช่น 2568) → แปลงเป็น ค.ศ. (2025)
    - ใส่ timezone Asia/Bangkok ให้เสมอ

    คืนค่าเป็น Series ของ datetime64[ns, Asia/Bangkok] (บางตัวอาจเป็น NaT ถ้า parse ไม่ได้)
    """
    s = series.astype(str).str.strip()

    # ลองตามฟอร์แมต dd/mm/yyyy HH:MM ก่อน
    dt = pd.to_datetime(
        s,
        format="%d/%m/%Y %H:%M",
        errors="coerce",
        dayfirst=True,
    )

    
    if dt.isna().all():
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)

    if dt.isna().all():
        
        return dt

    now_year = datetime.now(ZoneInfo(TZ)).year
    # ปีที่เกินเยอะ ๆ เช่น 2568 → ถือว่าเป็น พ.ศ. แล้วแปลงเป็น ค.ศ.
    be_mask = dt.dt.year > now_year + 1

    if be_mask.any():
        dt.loc[be_mask] = dt.loc[be_mask].map(
            lambda x: x.replace(year=x.year - 543) if pd.notna(x) else x
        )

    # ใส่ timezone ให้ตรง
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(ZoneInfo(TZ))
    else:
        dt = dt.dt.tz_convert(ZoneInfo(TZ))

    return dt

# ------------------- FUNCTIONS -------------------
def fetch_api_data(start=None, end=None):
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://official.airbkk.com/airbkk/th/report",
        "X-Requested-With": "XMLHttpRequest",
    })
    try:
        session.get("https://official.airbkk.com/airbkk/th/report", timeout=15)
    except Exception as e:
        logger.debug(f"Initial GET failed (non-fatal): {e}")

    now = datetime.now(ZoneInfo(TZ))
    if end is None:
        end = now.replace(minute=0, second=0, microsecond=0)
    if start is None:
        start = end - timedelta(hours=1)

    EXACT_ORDER = ["PM10", "WS", "WD", "Temp", "RH", "BP", "PM2.5"]
    payload = [
        ("groupid", "6"),
        ("MeasIndex[]", "91"),
        ("data_type", "hourly"),
        ("display_type", "table"),
        ("date_s", start.strftime("%d/%m/%Y %H:%M")),
        ("date_e", end.strftime("%d/%m/%Y %H:%M")),
    ]
    for p in EXACT_ORDER:
        payload.append(("parameterTags[]", p))

    try:
        r = session.post(URL, data=payload, timeout=30)
        r.raise_for_status()
        j = r.json()
        return j.get("arrData", []), start, end
    except Exception as e:
        logger.exception("fetch_api_data failed")
        raise


def arr_to_df_raw(arr):
    cols = ["Date_Time", "P1_91", "P2_91", "P3_91", "P4_91", "P5_91", "P6_91", "P7_91"]
    if not arr:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(arr)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    return df.astype(str)


def append_to_master_always_overwrite(df_new, master_file=MASTER_FILE, verbose=True):
    """
    อัปเดตไฟล์ raw เสมอ ไม่สนว่ามี Date_Time ซ้ำหรือไม่
    สำคัญมาก: ทำให้ค่าจริงจากกรมทับค่า None/ค่าเก่าได้ทันที

    verbose:
        True  → log ระดับ INFO (ใช้เวลา fetch ปกติ)
        False → log ระดับ DEBUG (ใช้เวลา fill ย้อนหลัง เพื่อลด spam)
    """
    os.makedirs(os.path.dirname(master_file), exist_ok=True)
    lock = FileLock(LOCK_FILE)

    try:
        lock.acquire(timeout=10)

        if master_file.exists():
            try:
                df_master = pd.read_csv(master_file, dtype=str, encoding="utf-8-sig")
            except EmptyDataError:
                df_master = pd.DataFrame()
        else:
            df_master = pd.DataFrame()

        if df_master.empty or "Date_Time" not in df_master.columns:
            df_new.to_csv(master_file, index=False, encoding="utf-8-sig")
            msg = f"สร้างไฟล์ใหม่: {master_file} ({len(df_new)} แถว)"
            if verbose:
                logger.info(msg)
            else:
                logger.debug(msg)
            return len(df_new)

        df_master["Date_Time"] = df_master["Date_Time"].astype(str)
        df_new["Date_Time"] = df_new["Date_Time"].astype(str)

        # รวมข้อมูลเก่า + ใหม่ → ลบซ้ำโดยเก็บแถวล่าสุด
        df_combined = pd.concat([df_master, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["Date_Time"], keep="last")
        df_combined = df_combined.sort_values("Date_Time", ascending=False)

        df_combined.to_csv(master_file, index=False, encoding="utf-8-sig")
        msg = f"อัปเดต raw file สำเร็จ → อัปเดต/เพิ่ม {len(df_new)} แถว (ทับค่าจริงทันที)"
        if verbose:
            logger.info(msg)
        else:
            logger.debug(msg)

        return len(df_new)

    except Timeout:
        logger.error("ล็อกไฟล์ล้มเหลว - มี process อื่นรันอยู่")
        return 0
    except Exception as e:
        logger.exception(f"append_to_master_always_overwrite ล้มเหลว: {e}")
        return 0
    finally:
        try:
            lock.release()
        except:
            pass


async def fill_missing_hours(lookback_hours=24):
    if shutdown_event.is_set():
        return 0

    # ------------------- เลือกไฟล์อ้างอิงสำหรับเช็คชั่วโมงที่ขาด -------------------
    if CLEAN_FILE.exists():
        source_file = CLEAN_FILE
        logger.info(f"ใช้ไฟล์ clean ในการเช็คชั่วโมงที่ขาด: {CLEAN_FILE}")
    elif MASTER_FILE.exists():
        source_file = MASTER_FILE
        logger.info(f"ไม่พบไฟล์ clean → ใช้ master แทน: {MASTER_FILE}")
    else:
        logger.info("ไม่พบไฟล์สำหรับเช็คชั่วโมงที่ขาด → ข้าม")
        return 0

    try:
        df_ref = pd.read_csv(source_file, dtype=str, encoding="utf-8-sig")
    except EmptyDataError:
        df_ref = pd.DataFrame()

    # ------------------- เตรียมช่วงเวลา (window) ที่จะเช็ค -------------------
    now = datetime.now(ZoneInfo(TZ)).replace(minute=0, second=0, microsecond=0)
    start_fill = now - timedelta(hours=lookback_hours)
    end_fill = now

    if df_ref.empty or "Date_Time" not in df_ref.columns:
        # ไม่มีข้อมูล → ไม่ต้อง filter อะไร ใช้ df_ref ว่าง ๆ ไปเลย
        df_ref = pd.DataFrame(columns=["dt"])
    else:
        # แปลง Date_Time เป็น datetime แล้วจัดการ timezone ให้ตรงกัน
        dt = _parse_date_time_be_aware(df_ref["Date_Time"])

        # กันเคสที่ _parse_date_time_be_aware คืนค่าแบบไม่มี tz
        try:
            tzinfo = dt.dt.tz
        except AttributeError:
            tzinfo = None

        if tzinfo is None:
            # ถ้าไม่มี timezone ให้ใส่ Asia/Bangkok ให้เลย
            dt = dt.dt.tz_localize(ZoneInfo(TZ))
        else:
            # ถ้ามี timezone อยู่แล้วก็แปลงให้เป็น Asia/Bangkok
            dt = dt.dt.tz_convert(ZoneInfo(TZ))

        df_ref["dt"] = dt
        df_ref = df_ref.dropna(subset=["dt"])

        # ตัดข้อมูลให้เหลือเฉพาะในช่วง 48 ชม.ล่าสุด
        if not df_ref.empty:
            df_ref = df_ref[(df_ref["dt"] >= start_fill) & (df_ref["dt"] <= end_fill)]

    # ------------------- สร้าง list ชั่วโมงที่ "ควรมี" -------------------
    expected = pd.date_range(start_fill, end_fill, freq="h", tz=ZoneInfo(TZ))

    existing = set(df_ref["dt"].dt.floor("h")) if not df_ref.empty else set()
    missing_hours = [dt for dt in expected if dt.floor("h") not in existing]

    if not missing_hours:
        logger.info("ไม่มีชั่วโมงที่ขาด (ในช่วง lookback)")
        return 0

    logger.info(f"พบ {len(missing_hours)} ชั่วโมงที่ขาด → ดึงใหม่")

    # ------------------- ดึงย้อนหลังแล้วเขียนลง MASTER_FILE -------------------
    total_added = 0
    for dt in missing_hours:
        if shutdown_event.is_set():
            break

        try:
            arr, _, _ = fetch_api_data(start=dt, end=dt + timedelta(hours=1))
            if arr:
                df_new = arr_to_df_raw(arr)
                added = append_to_master_always_overwrite(df_new, verbose=False)
                total_added += added
                logger.debug(f"เติม {dt.strftime('%Y-%m-%d %H:%M')} → +{added}")
        except Exception as e:
            logger.error(f"ดึง {dt} ล้มเหลว: {e}")

        if not shutdown_event.is_set():
            await asyncio.sleep(1)

    logger.info(f"เติมข้อมูลย้อนหลังเสร็จ: +{total_added} แถว จาก {len(missing_hours)} ชั่วโมง")
    return total_added



def main():
    try:
        setup_signal_handlers()
    except Exception:
        logger.debug("setup_signal_handlers skipped")

    try:
        arr, start, end = fetch_api_data()
        logger.info(f"ดึง {start} - {end} → {len(arr)} แถว")

        if len(arr) == 0 and FETCH_RETRY:
            logger.info(f"ไม่มีข้อมูล → รอ {RETRY_SLEEP_SECONDS} วินาที")
            waited = 0
            while waited < RETRY_SLEEP_SECONDS:
                if shutdown_event.is_set():
                    logger.info("ถูกสั่งหยุดขณะรอ retry")
                    break
                time.sleep(1)
                waited += 1
            else:
                arr, start, end = fetch_api_data()

        if len(arr) == 0:
            logger.info("ยังไม่มีข้อมูลใหม่")
            return

        df_new = arr_to_df_raw(arr)
        added = append_to_master_always_overwrite(df_new)  # verbose=True (ค่าเริ่มต้น)
        logger.info(f"สำเร็จ: อัปเดต/เพิ่ม {added} แถว")

    except Exception:
        logger.exception("ดึงข้อมูลล้มเหลว")
        raise


if __name__ == "__main__":
    main()
