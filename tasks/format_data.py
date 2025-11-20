# tasks/format_data.py
"""
Formatter for raw AirBKK data (ปรับจาก format_tungkru.py)
อ่านจาก data/staging/raw_airbkk.csv
เขียนไป data/staging/formatted_air.csv
ไม่ซ้ำด้วย Date_Time
"""

import pandas as pd
from pathlib import Path
import logging
import sys

# ------------------- CONFIG -------------------
STAGING_DIR = Path("data/staging")
STAGING_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = STAGING_DIR / "raw_airbkk.csv"
TARGET_FILE = STAGING_DIR / "formatted_air.csv"

# ------------------- LOGGING -------------------
LOG_FILE = STAGING_DIR / "format.log"
STAGING_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ],
)
logger = logging.getLogger("format_tungkru")

# ------------------- FUNCTIONS -------------------
def _read_input(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8-sig", dtype=str)
    logger.info(f"อ่านข้อมูลดิบ: {input_path} → {len(df)} แถว")
    return df

def _map_and_select(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "P1_91": "PM10 (µg/m³)",
        "P2_91": "WS (m/s)",
        "P3_91": "WD",
        "P4_91": "Temp (°C)",
        "P5_91": "RH (%)",
        "P6_91": "BP (mBar)",
        "P7_91": "PM2.5 (µg/m³)",
    }
    df_renamed = df.rename(columns=rename_map, errors="ignore")

    desired_cols = [
        "Date_Time",
        "PM10 (µg/m³)",
        "PM2.5 (µg/m³)",
        "WS (m/s)",
        "WD",
        "Temp (°C)",
        "RH (%)",
        "BP (mBar)"
    ]

    existing_cols = [c for c in desired_cols if c in df_renamed.columns]
    df_new = df_renamed[existing_cols].copy()
    logger.info(f"เลือกคอลัมน์: {len(existing_cols)} คอลัมน์")
    return df_new

def format_raw() -> int:
    """
    อ่าน raw → format → append → return จำนวนแถวที่เพิ่ม
    """
    try:
        input_path = INPUT_FILE
        target_path = TARGET_FILE

        if not input_path.exists():
            logger.warning(f"ไม่พบไฟล์: {input_path} → ข้าม")
            return 0

        df_raw = _read_input(input_path)
        df_new_formatted = _map_and_select(df_raw)

        # --- Append + Deduplicate ---
        target_dir = target_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        if target_path.exists():
            try:
                df_history = pd.read_csv(target_path, encoding="utf-8-sig", dtype=str)
                df_combined = pd.concat([df_history, df_new_formatted], ignore_index=True)
                if "Date_Time" in df_combined.columns:
                    before = len(df_combined)
                    df_final = df_combined.drop_duplicates(subset=['Date_Time'], keep='last')
                    added = len(df_final) - len(df_history)
                    logger.info(f"รวมเก่า {len(df_history)} + ใหม่ {len(df_new_formatted)} → {len(df_final)} (เพิ่ม {added})")
                else:
                    df_final = df_combined
                    added = len(df_new_formatted)
            except Exception as e:
                logger.warning(f"ไฟล์เก่ามีปัญหา: {e} → สร้างใหม่")
                df_final = df_new_formatted
                added = len(df_new_formatted)
        else:
            logger.info("สร้างไฟล์ formatted ใหม่")
            df_final = df_new_formatted
            added = len(df_new_formatted)

        # Sort by Date_Time
        if "Date_Time" in df_final.columns:
            try:
                df_final = df_final.sort_values(by="Date_Time")
            except Exception:
                logger.debug("ไม่สามารถ sort ได้")

        # Save
        df_final.to_csv(target_path, index=False, encoding="utf-8-sig")
        logger.info(f"บันทึก: {target_path}")
        return added

    except FileNotFoundError as fe:
        logger.error(fe)
        return 0
    except Exception as e:
        logger.exception("format_raw() ล้มเหลว")
        return 0

# ------------------- RUN -------------------
if __name__ == "__main__":
    added = format_raw()
    if added == 0:
        sys.exit(1)