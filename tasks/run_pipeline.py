# tasks/run_pipeline.py
import asyncio
import pandas as pd
from pathlib import Path
import logging
import subprocess
import sys  # เพิ่มนี้

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STAGING_DIR = Path("data/staging")
BACKUP_DIR = Path("data/backups")
STAGING_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

from .format_data import format_raw
from .clean_data import clean_and_save
from .load_to_db import load_and_backup

async def run_pipeline():
    logger.info("เริ่ม Pipeline...")

    # 1. ดึงข้อมูลด้วย venv python
    try:
        VENV_PYTHON = sys.executable  # ใช้ python จาก venv
        result = subprocess.run(
            [VENV_PYTHON, "tasks/fetch_api.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode != 0:
            logger.error(f"Fetch ล้มเหลว: {result.stderr}")
            return
        logger.info("Fetch สำเร็จ")
    except Exception as e:
        logger.error(f"Fetch ล้มเหลว: {e}")
        return

    # 2. อ่าน formatted_air.csv (format_raw() จัดการเอง)
    formatted_file = STAGING_DIR / "formatted_air.csv"
    if not formatted_file.exists():
        logger.warning("ไม่พบ formatted_air.csv")
        return

    df = pd.read_csv(formatted_file, encoding="utf-8-sig")
    if df.empty:
        logger.info("ไม่มีข้อมูลใน formatted")
        return

    # 3. Clean + Load
    df = clean_and_save(df)
    await load_and_backup(df)
    
    logger.info("Pipeline สำเร็จทั้งหมด!")

if __name__ == "__main__":
    asyncio.run(run_pipeline())