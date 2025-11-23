# app/main.py
from fastapi import FastAPI, Depends, HTTPException
from app.db import AsyncSessionLocal
from app.crud import upsert_air_reading
from tasks.clean_data import clean_and_save
from tasks.load_to_db import load_and_backup
from tasks.format_data import format_raw
from tasks.fetch_api import main as fetch_main, fill_missing_hours, setup_signal_handlers
import pandas as pd
from pathlib import Path
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import logging
import traceback
import joblib
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
import xgboost as xgb

# ========================================
#               CONFIG
# ========================================
MAX_LAG = 120
ROLL_WINDOWS = [3, 6, 12, 24]
DB_FETCH_LIMIT = 400  # จำนวนแถวที่ดึงจาก DB (เพียงพอสำหรับ lag/rolling)
# ค่า fallback ถ้าโมเดลไม่มี metadata ให้ตั้งเป็น 168 (ตามที่โมเดลของคุณเคยคาดไว้)
FALLBACK_TARGET_FEATURE_COUNT = 168
def pm25_to_aqi_th(pm25: float) -> dict:
    """
    แปลงค่า PM2.5 (µg/m³) เป็น AQI + ระดับคุณภาพอากาศ
    ตามเกณฑ์ที่ให้มา:

    ช่วงค่า AQI      ระดับคุณภาพอากาศ                 ช่วงค่า PM2.5 (µg/m³)
    0 - 25           ดีมาก                              0 - 15
    26 - 50          ดี                                 16 - 25
    51 - 100         ปานกลาง                           26 - 37.5
    101 - 200        เริ่มมีผลกระทบต่อสุขภาพ          38 - 90
    >200             มีผลกระทบต่อสุขภาพ               > 90

    ใช้ linear interpolation ภายในแต่ละช่วง AQI
    """
    if pm25 is None:
        return {"aqi": None, "level": None}

    pm25 = max(0.0, float(pm25))

    # (C_low, C_high, I_low, I_high, level)
    ranges = [
        (0.0,   15.0,   0,   25,  "ดีมาก"),
        (15.0,  25.0,  26,   50,  "ดี"),
        (25.0,  37.5,  51,  100,  "ปานกลาง"),
        (37.5,  90.0, 101,  200,  "เริ่มมีผลกระทบต่อสุขภาพ"),
        (90.0, 250.0, 201,  300,  "มีผลกระทบต่อสุขภาพ"),  # กำหนดบนสุดที่ 300
    ]

    for C_low, C_high, I_low, I_high, level in ranges:
        if pm25 <= C_high:
            if C_high == C_low:
                aqi = I_high
            else:
                aqi = (I_high - I_low) / (C_high - C_low) * (pm25 - C_low) + I_low
            return {"aqi": int(round(aqi)), "level": level}

    # ถ้าเกินช่วงบนสุดมาก ๆ
    return {"aqi": 300, "level": "มีผลกระทบต่อสุขภาพ"}

# ========================================
#               FASTAPI APP
# ========================================
app = FastAPI(
    title="AirBKK Real-time & Forecast API",
    description="ดึงข้อมูลฝุ่น PM2.5 ทุกชั่วโมง + พยากรณ์ล่วงหน้า 24 ชม. ด้วย AI",
    version="1.0"
)

# ========================================
#               PATHS & MODEL
# ========================================
STAGING_DIR = Path("data/staging")
STAGING_DIR.mkdir(parents=True, exist_ok=True)

FORMATTED_FILE = STAGING_DIR / "formatted_air.csv"
CLEANED_FILE = STAGING_DIR / "cleaned_air.csv"
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_pm25_best.json"

MODEL = None
MODEL_EXPECTED_N = None

if MODEL_PATH.exists():
    try:
        MODEL = xgb.XGBRegressor()
        MODEL.load_model(str(MODEL_PATH))
        # พยายามอ่านจำนวน feature ที่โมเดลคาดไว้ (metadata)
        try:
            MODEL_EXPECTED_N = int(getattr(MODEL, "n_features_in_", None))
        except Exception:
            MODEL_EXPECTED_N = None

        # หากยังไม่ได้ ให้ลองจาก booster
        try:
            if MODEL_EXPECTED_N is None and hasattr(MODEL, "get_booster"):
                booster = MODEL.get_booster()
                if hasattr(booster, "num_features"):
                    MODEL_EXPECTED_N = int(booster.num_features())
        except Exception:
            pass

        logging.info(f"MODEL loaded. expected features (metadata): {MODEL_EXPECTED_N}")
    except Exception as e:
        logging.exception("โหลดโมเดลล้มเหลว")
        MODEL = None
else:
    logging.warning(f"ไม่พบโมเดลที่: {MODEL_PATH}")

# ========================================
#               LOGGING
# ========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

# ========================================
#               PIPELINE & SCHEDULER
# ========================================
async def run_pipeline():
    logging.info("=== PIPELINE START ===")
    try:
        fetch_main()
        logging.info("Fetch สำเร็จ")

        added = await fill_missing_hours(lookback_hours=48)
        # logging.info(f"เติมข้อมูลย้อนหลัง: +{added} แถว")

        added = format_raw()
        if added:
            logging.info(f"Format สำเร็จ: +{added} แถว")
        else:
            logging.info("ไม่มีข้อมูลใหม่ให้ format")

        if FORMATTED_FILE.exists():
            df = pd.read_csv(FORMATTED_FILE, encoding="utf-8-sig")
            if not df.empty:
                df_cleaned = clean_and_save(df)
                if df_cleaned is not None and not df_cleaned.empty:
                    await load_and_backup(df_cleaned)
                    logging.info("Load to DB + Backup สำเร็จ")
    except Exception as e:
        logging.error(f"Pipeline ล้มเหลว: {type(e).__name__}: {e}")
        logging.error(traceback.format_exc())
    logging.info("=== PIPELINE END ===\n")

scheduler = AsyncIOScheduler(timezone="Asia/Bangkok")
scheduler.add_job(run_pipeline, "cron", minute=5, hour="*", id="airbkk_pipeline", replace_existing=True)

# ========================================
#               DATABASE DEPENDENCY
# ========================================
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# ========================================
#               STARTUP
# ========================================
@app.on_event("startup")
async def startup_event():
    try:
        setup_signal_handlers()
    except Exception:
        logging.debug("setup_signal_handlers failed (maybe already set)")

    scheduler.start()
    logging.info("Scheduler เริ่มแล้ว → ทุกชั่วโมง นาทีที่ 5")
    # รัน pipeline ทันที (ไม่บล็อก)
    asyncio.create_task(run_pipeline())

# ========================================
#               SIMPLE ENDPOINTS
# ========================================
@app.get("/")
async def root():
    return {
        "message": "AirBKK API พร้อมใช้งาน",
        "endpoints": {
            "/latest": "ข้อมูลล่าสุด 10 รายการ",
            "/predict": "พยากรณ์ฝุ่น PM2.5 ล่วงหน้า 24 ชั่วโมง (AI)",
            "/resync": "รีโหลดข้อมูลทั้งหมดเข้า DB"
        },
        "model_loaded": MODEL is not None,
        "model_expected_features": MODEL_EXPECTED_N
    }

@app.get("/latest")
async def latest_readings(db=Depends(get_db)):
    try:
        query = text('SELECT * FROM air_readings ORDER BY "Date_Time" DESC LIMIT 10')
        result = await db.execute(query)
        rows = result.fetchall()

        # แปลงเป็น list of dict ให้สวย
        data = []
        for row in rows:
            # ใช้ _mapping (SQLAlchemy 2.0 style)
            data.append(dict(row._mapping))

        return {"data": data[::-1]}  # เรียงจากเก่า→ใหม่ (หรือไม่กลับก็ได้)
    except Exception as e:
        logging.exception("latest_readings error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/resync")
async def resync_db():
    if not CLEANED_FILE.exists():
        raise HTTPException(status_code=404, detail="ไม่พบ cleaned_air.csv")
    df = pd.read_csv(CLEANED_FILE, encoding="utf-8-sig")
    count = await load_and_backup(df, force_resync=True)
    return {"status": "resync สำเร็จ", "rows_loaded": count}

# ========================================
#           PREDICT API (24 ชม.) with debug
# ========================================
@app.get("/predict")
async def predict_next_24h():
    if MODEL is None:
        raise HTTPException(503, "โมเดลยังไม่พร้อม")

    async with AsyncSessionLocal() as db:
        try:
            query = text('''
                SELECT 
                    "Date_Time",
                    "PM2.5 (µg/m³)" as pm25,
                    "PM10 (µg/m³)" as pm10,
                    "WS (m/s)" as ws,
                    "WD" as wd,
                    "Temp (°C)" as temp,
                    "RH (%)" as rh,
                    "BP (mBar)" as bp
                FROM air_readings 
                ORDER BY "Date_Time" DESC 
                LIMIT 200
            ''')
            result = await db.execute(query)
            rows = result.fetchall()
            if len(rows) < 60:
                raise HTTPException(400, "ข้อมูลไม่พอสำหรับพยากรณ์")

            df = pd.DataFrame(rows, columns=[
                "Date_Time","pm25","pm10","ws","wd","temp","rh","bp"
            ])
            df["Date_Time"] = pd.to_datetime(df["Date_Time"])
            df = df.sort_values("Date_Time").reset_index(drop=True)

            # ===== ค่าปัจจุบันจาก DB (ก่อนทำ lag/dropna) =====
            current_row = df.iloc[-1]
            current_pm25 = float(current_row["pm25"])
            current_time = current_row["Date_Time"]
            current_aqi_info = pm25_to_aqi_th(current_pm25)

            # ===== เก็บช่วงค่าจริงของ PM2.5 สำหรับ inverse scaling =====
            pm25_min = float(df["pm25"].min())
            pm25_max = float(df["pm25"].max())
            if pm25_max == pm25_min:
                pm25_min = 0.0

            # ===== Lag 24 สำหรับ 7 ตัวแปร (ตรงกับตอน train) =====
            features_order = ["pm10", "ws", "wd", "temp", "rh", "bp", "pm25"]
            for lag in range(1, 25):
                for col in features_order:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)

            df = df.dropna().reset_index(drop=True)
            latest = df.iloc[-1]

            expected_features = []
            for col in features_order:
                for lag in range(1, 25):
                    expected_features.append(f"{col}_lag_{lag}")

            if len(expected_features) != 168:
                raise ValueError(f"Expected 168 features แต่ได้ {len(expected_features)}")

            X = np.array(
                [latest.get(col, 0.0) for col in expected_features],
                dtype=np.float32
            ).reshape(1, -1)

            preds = []
            current = X.copy()
            base_time = df["Date_Time"].iloc[-1]

            # เตรียม index ของ pm25 lag เอาไว้ใช้ใน loop
            pm25_indices = [expected_features.index(f"pm25_lag_{i}") for i in range(1, 25)]

            for step in range(24):
                pred_scaled = float(MODEL.predict(current)[0])

                # ===== inverse scale แบบสมจริง (อิงช่วง recent data) =====
                pred_actual = pred_scaled * (pm25_max - pm25_min) + pm25_min
                pred_actual = max(0.0, pred_actual)
                preds.append(round(pred_actual, 1))

                # ===== update lag เฉพาะ PM2.5 =====
                current[0, pm25_indices] = np.roll(current[0, pm25_indices], -1)
                current[0, pm25_indices[0]] = pred_scaled

            base_time = current_time  # อิงเวลาจริงที่ station บันทึกไว้

            hours = [
                        (base_time + timedelta(hours=i+1)).strftime("%H:%M")
                            for i in range(24)
                    ]

            max_val = max(preds)
            max_aqi_info = pm25_to_aqi_th(max_val)

            return {
                # --- ค่าปัจจุบันจากสถานี (real data) ---
                "ค่าฝุ่น pm2.5 ปัจจุบัน": round(current_pm25, 1),
                "ดัชนีคุณภาพอากาศปัจจุบัน": current_aqi_info["aqi"],
                "ระดับคุณภาพอากาศปัจจุบัน": current_aqi_info["level"],
                

                # --- พยากรณ์ 24 ชั่วโมงข้างหน้า (PM2.5 ตามชั่วโมง) ---
                "พยากรณ์ฝุ่น_24ชั่วโมง": dict(zip(hours, preds)),

                # --- สรุป 24 ชั่วโมงข้างหน้า ---
                "PM2.5 สูงสุดที่คาดการณ์": round(max_val, 1),
                "AQI สูงสุดที่คาดการณ์": max_aqi_info["aqi"],
                "ระดับโดยรวมจากค่าสูงสุด": max_aqi_info["level"],
                "สรุปผล": f"24 ชม. ข้างหน้า ฝุ่นPM2.5สูงสุด {max_val:.1f} µg/m³ → {max_aqi_info['level']}",
            }

        except Exception as e:
            logging.exception("Predict error")
            raise HTTPException(500, f"พยากรณ์ล้มเหลว: {str(e)}")

# ========================================
#               HEALTH CHECK
# ========================================
@app.get("/health")
async def health():
    rows = 0
    try:
        rows = len(pd.read_csv(CLEANED_FILE)) if CLEANED_FILE.exists() else 0
    except Exception:
        rows = 0
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": MODEL is not None,
        "model_expected_features": MODEL_EXPECTED_N,
        "data_file_exists": CLEANED_FILE.exists(),
        "rows_in_csv": rows
    }
