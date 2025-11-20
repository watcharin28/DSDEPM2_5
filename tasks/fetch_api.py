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

MASTER_FILE = STAGING_DIR / "raw_airbkk.csv"
LOCK_FILE = STAGING_DIR / ".airbkk_lock"
LOG_FILE = STAGING_DIR / "capture.log"

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


def append_to_master_always_overwrite(df_new, master_file=MASTER_FILE):
    """
    อัปเดตไฟล์ raw เสมอ ไม่สนว่ามี Date_Time ซ้ำหรือไม่
    สำคัญมาก: ทำให้ค่าจริงจากกรมทับค่า None/ค่าเก่าได้ทันที
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
            logger.info(f"สร้างไฟล์ใหม่: {master_file} ({len(df_new)} แถว)")
            return len(df_new)

        df_master["Date_Time"] = df_master["Date_Time"].astype(str)
        df_new["Date_Time"] = df_new["Date_Time"].astype(str)

        # รวมข้อมูลเก่า + ใหม่ → ลบซ้ำโดยเก็บแถวล่าสุด
        df_combined = pd.concat([df_master, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["Date_Time"], keep="last")
        df_combined = df_combined.sort_values("Date_Time", ascending=False)

        df_combined.to_csv(master_file, index=False, encoding="utf-8-sig")
        logger.info(f"อัปเดต raw file สำเร็จ → อัปเดต/เพิ่ม {len(df_new)} แถว (ทับค่าจริงทันที)")

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


async def fill_missing_hours(lookback_hours=48):
    if shutdown_event.is_set():
        return 0

    if not MASTER_FILE.exists():
        logger.info("ไม่พบไฟล์ master → ข้าม")
        return 0

    try:
        df_master = pd.read_csv(MASTER_FILE, dtype=str, encoding="utf-8-sig")
    except EmptyDataError:
        df_master = pd.DataFrame()

    if df_master.empty or "Date_Time" not in df_master.columns:
        start_fill = datetime.now(ZoneInfo(TZ)) - timedelta(hours=lookback_hours)
        start_fill = start_fill.replace(minute=0, second=0, microsecond=0)
    else:
        df_master["dt"] = pd.to_datetime(df_master["Date_Time"], format="%d/%m/%Y %H:%M", errors="coerce")
        df_master = df_master.dropna(subset=["dt"])
        if df_master.empty:
            start_fill = datetime.now(ZoneInfo(TZ)) - timedelta(hours=lookback_hours)
            start_fill = start_fill.replace(minute=0, second=0, microsecond=0)
        else:
            latest = df_master["dt"].max()
            start_fill = latest.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    end_fill = datetime.now(ZoneInfo(TZ)).replace(minute=0, second=0, microsecond=0)
    if start_fill >= end_fill:
        logger.info("ข้อมูลครบแล้ว")
        return 0

    expected = pd.date_range(start_fill, end_fill, freq='h', tz=ZoneInfo(TZ))
    existing = set(df_master["dt"].dt.floor('h')) if "dt" in df_master.columns and not df_master.empty else set()
    missing_hours = [dt for dt in expected if dt.floor('h') not in existing]

    if not missing_hours:
        logger.info("ไม่มีชั่วโมงที่ขาด")
        return 0

    logger.info(f"พบ {len(missing_hours)} ชั่วโมงที่ขาด → ดึงใหม่")

    total_added = 0
    for dt in missing_hours:
        if shutdown_event.is_set():
            break

        try:
            arr, _, _ = fetch_api_data(start=dt, end=dt + timedelta(hours=1))
            if arr:
                df_new = arr_to_df_raw(arr)
                added = append_to_master_always_overwrite(df_new)
                total_added += added
                logger.info(f"เติม {dt.strftime('%Y-%m-%d %H:%M')} → +{added}")
        except Exception as e:
            logger.error(f"ดึง {dt} ล้มเหลว: {e}")

        if not shutdown_event.is_set():
            await asyncio.sleep(1)

    logger.info(f"เติมข้อมูลย้อนหลังเสร็จ: +{total_added} แถว")
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
        added = append_to_master_always_overwrite(df_new)  # แก้ตรงนี้!
        logger.info(f"สำเร็จ: อัปเดต/เพิ่ม {added} แถว")

    except Exception as e:
        logger.exception("ดึงข้อมูลล้มเหลว")
        raise


if __name__ == "__main__":
    main()