# tasks/clean_data.py
import pandas as pd
import re
from pathlib import Path
import logging

# ------------------- CONFIG -------------------
STAGING_DIR = Path("data/staging")
STAGING_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = STAGING_DIR / "cleaned_air.csv"

# ------------------- LOGGING -------------------
LOG_FILE = STAGING_DIR / "clean.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("clean_data")

# ------------------- FUNCTIONS -------------------
def convert_be_series(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    s2 = s.str.replace(
        r'(\b\d{1,2}/\d{1,2}/)(\d{4})',
        lambda m: m.group(1) + str(int(m.group(2)) - 543),
        regex=True
    )
    return s2

def clean_and_save(df: pd.DataFrame) -> pd.DataFrame:
    """
    รับ df → clean → บันทึก → คืน df ที่ clean แล้ว
    """
    if df.empty:
        logger.info("ไม่มีข้อมูลให้ clean")
        return df

    logger.info(f"เริ่ม clean ข้อมูล: {len(df)} แถว")

    # --- 1. แทนค่า missing ---
    df.replace(['n/a', 'None', '', 'NA', '-'], pd.NA, inplace=True)

    # --- 2. แปลง BE → CE + parse Date_Time ---
    df['Date_Time'] = convert_be_series(df['Date_Time'])
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%d/%m/%Y %H:%M', errors='coerce')

    invalid_dates = df['Date_Time'].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"วันที่ไม่ถูกต้อง: {invalid_dates} แถว")
    df = df.dropna(subset=['Date_Time'])

    # --- 3. แปลงตัวเลข ---
    numeric_cols = [c for c in df.columns if c != 'Date_Time']
    for c in numeric_cols:
        df[c] = df[c].astype(str).str.extract(r'([-+]?\d*\.?\d+)')[0]
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # --- 4. Sort + Interpolate ---
    df_sorted = df.sort_values('Date_Time').set_index('Date_Time')
    df_sorted[numeric_cols] = df_sorted[numeric_cols].interpolate(method='time', limit_direction='both')
    df_sorted[numeric_cols] = df_sorted[numeric_cols].fillna(df_sorted[numeric_cols].mean())
    df_cleaned = df_sorted.reset_index()

    # --- 5. บันทึก ---
    df_cleaned.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    logger.info(f"บันทึกสำเร็จ: {OUTPUT_FILE} ({len(df_cleaned)} แถว)")

    # แสดง 10 ล่าสุด
    latest10 = df_cleaned.sort_values('Date_Time', ascending=False).head(10)
    logger.info("\nล่าสุด 10 แถว:\n" + latest10.to_string(index=False))

    return df_cleaned  # คืน df ที่ clean แล้ว