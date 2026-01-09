# tasks/load_to_db.py
import pandas as pd
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import AsyncSessionLocal
from app.crud import upsert_air_reading
from app.schemas import AirReading
from datetime import datetime
import shutil
import logging

# ------------------- CONFIG -------------------
BACKUP_DIR = Path("data/backups")
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
LATEST_BACKUP = BACKUP_DIR / "latest.csv"

# ------------------- LOGGING -------------------
STAGING_DIR = Path("data/staging")
LOG_FILE = STAGING_DIR / "load.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("load_to_db")

# ------------------- MAIN -------------------


async def load_and_backup(df: pd.DataFrame, force_resync: bool = False) -> int:
    """
    รับ df → แปลงเวลา → ใส่ DB → backup → คืนจำนวนแถว
    """
    if df.empty:
        logger.info("ไม่มีข้อมูลให้ load")
        return 0

    # --- 1. แปลงเวลา ---
    try:
        df['Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce')
        df = df.dropna(subset=['Date_Time'])
        df['Date_Time'] = df['Date_Time'].dt.tz_localize(
            "Asia/Bangkok", ambiguous='NaT', nonexistent='shift_forward'
        )
        df = df.dropna(subset=['Date_Time'])
    except Exception as e:
        logger.error(f"แปลงเวลา ล้มเหลว: {e}")
        return 0

    if df.empty:
        logger.info("ไม่มีข้อมูลที่ถูกต้องให้โหลด")
        return 0

    logger.info(f"เตรียมโหลด {len(df)} แถว")

    # --- 2. Load to DB ---
    try:
        async with AsyncSessionLocal() as db:
            if force_resync:
                # ลบข้อมูลเก่าใน DB (ถ้าต้องการ resync)
                from sqlalchemy import delete
                await db.execute(delete(AirReading))
                await db.commit()
                logger.info("ล้าง DB เก่า (resync)")

            for _, row in df.iterrows():
                obj = AirReading(
                    Date_Time=row['Date_Time'],
                    PM10=row['PM10 (µg/m³)'],
                    PM25=row['PM2.5 (µg/m³)'],
                    WS=row['WS (m/s)'],
                    WD=row['WD'],
                    Temp=row['Temp (°C)'],
                    RH=row['RH (%)'],
                    BP=row['BP (mBar)'],
                )
                await upsert_air_reading(db, obj)
        logger.info("โหลด DB สำเร็จ")
    except Exception as e:
        logger.error(f"โหลด DB ล้มเหลว: {e}")
        raise

    # --- 3. Backup ---
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        backup_file = BACKUP_DIR / f"{today}.csv"
        df.to_csv(backup_file, index=False, encoding="utf-8-sig")
        logger.info(f"Backup สำเร็จ: {backup_file}")

        # อัพเดท latest.csv อย่างปลอดภัย
        temp_file = BACKUP_DIR / "latest_temp.csv"
        df.to_csv(temp_file, index=False, encoding="utf-8-sig")

        if LATEST_BACKUP.exists():
            LATEST_BACKUP.unlink()  # ลบเก่า
        temp_file.replace(LATEST_BACKUP)  # ย้าย (atomic)
        logger.info("อัพเดท latest.csv สำเร็จ")

    except Exception as e:
        logger.error(f"Backup ล้มเหลว: {e}")
        raise

    return len(df)
