import asyncio
import pandas as pd
from pathlib import Path
import logging
import subprocess
import sys
import os

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ตั้งค่าโฟลเดอร์
STAGING_DIR = Path("data/staging")
BACKUP_DIR = Path("data/backups")
STAGING_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# เปลี่ยนจาก relative import → absolute import (สำคัญมาก!)
from tasks.format_data import format_raw
from tasks.clean_data import clean_and_save
from tasks.load_to_db import load_and_backup


async def run_pipeline():
    logger.info("เริ่ม Pipeline...")

    # 1. รัน fetch_api.py เพื่อดึงข้อมูล (ใช้ Python จาก venv ปัจจุบัน)
    try:
        VENV_PYTHON = sys.executable  # เช่น C:\Pm2_5\venv\Scripts\python.exe
        fetch_script = Path(__file__).parent / "fetch_api.py"

        logger.info("กำลังดึงข้อมูลจาก API...")
        result = subprocess.run(
            [VENV_PYTHON, str(fetch_script)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent  # รันจาก root ของโปรเจกต์
        )

        if result.returncode != 0:
            logger.error(f"Fetch ล้มเหลว!\n{result.stderr}")
            return
        else:
            logger.info("ดึงข้อมูลจาก API สำเร็จ")

    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการรัน fetch_api.py: {e}")
        return

    # 2. อ่านไฟล์ที่ format_raw สร้างไว้
    formatted_file = STAGING_DIR / "formatted_air.csv"
    if not formatted_file.exists():
        logger.warning("ไม่พบไฟล์ data/staging/formatted_air.csv")
        logger.info("ลองรัน tasks/format_data.py ก่อน หรือตรวจสอบ fetch_api.py")
        return

    try:
        df = pd.read_csv(formatted_file, encoding="utf-8-sig")
        logger.info(f"โหลด formatted_air.csv สำเร็จ → {len(df):,} แถว")
    except Exception as e:
        logger.error(f"อ่าน formatted_air.csv ไม่ได้: {e}")
        return

    if df.empty:
        logger.info("ไฟล์ formatted_air.csv ว่างเปล่า")
        return

    # 3. Clean ข้อมูล
    logger.info("กำลังทำความสะอาดข้อมูล...")
    df_cleaned = clean_and_save(df)
    if df_cleaned is None or df_cleaned.empty:
        logger.error("Clean ข้อมูลล้มเหลว")
        return

    # 4. Load เข้า Database + Backup
    logger.info("กำลังโหลดข้อมูลเข้า Database และ Backup...")
    await load_and_backup(df_cleaned)

    logger.info("Pipeline ทำงานสำเร็จทั้งหมด!")


if __name__ == "__main__":
    # รัน pipeline ทั้งหมด
    asyncio.run(run_pipeline())