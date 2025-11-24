
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
from zoneinfo import ZoneInfo

# ========================================
#               CONFIG
# ========================================
TH_TZ = ZoneInfo("Asia/Bangkok")
BASE_DIR = Path(__file__).resolve().parent.parent

STAGING_DIR = BASE_DIR / "data" / "staging"
STAGING_DIR.mkdir(parents=True, exist_ok=True)
FORMATTED_FILE = STAGING_DIR / "formatted_air.csv"
CLEANED_FILE = STAGING_DIR / "cleaned_air.csv"

MODEL_PATH = BASE_DIR / "models" / "xgboost_pm25_best.json"
SCALER_PATH = BASE_DIR / "models" / "pm25_scaler_7feats.pkl"

# ชื่อคอลัมน์ที่ตรงกับ DB จริง ๆ (ต้องใช้ชื่อย่อ)
FEATURES_ORDER = ["pm10", "ws", "wd", "temp", "rh", "bp", "pm25"]

# ========================================
#               LOAD MODEL + SCALER
# ========================================
MODEL = None
SCALER = None
MODEL_EXPECTED_N = None

if MODEL_PATH.exists():
    try:
        MODEL = xgb.XGBRegressor()
        MODEL.load_model(str(MODEL_PATH))
        MODEL_EXPECTED_N = getattr(MODEL, "n_features_in_", None) or 168
        logging.info(f"[MODEL] โหลดสำเร็จ → คาดหวัง {MODEL_EXPECTED_N} features")
    except Exception as e:
        logging.exception("โหลดโมเดลล้มเหลว!")
        MODEL = None
else:
    logging.warning(f"ไม่พบโมเดลที่: {MODEL_PATH}")

if SCALER_PATH.exists():
    try:
        SCALER = joblib.load(SCALER_PATH)
        logging.info(f"[SCALER] โหลดสำเร็จ → {SCALER.n_features_in_} features (ต้องเป็น 168)")
    except Exception as e:
        logging.error(f"โหลด scaler ล้มเหลว: {e}")
else:
    logging.warning("ไม่พบ pm25_scaler.pkl → พยากรณ์จะแบน!")
    # ===== DEBUG สำหรับเทียบ local vs deploy =====
logging.info(f"[DEBUG] MODEL_PATH = {MODEL_PATH}")
logging.info(f"[DEBUG] SCALER_PATH = {SCALER_PATH}")

if SCALER is not None:
    try:
        logging.info(f"[DEBUG] SCALER class = {SCALER.__class__.__name__}")
        logging.info(f"[DEBUG] SCALER.n_features_in_ = {SCALER.n_features_in_}")
    except Exception as e:
        logging.info(f"[DEBUG] อ่าน n_features_in_ ไม่ได้: {e}")

    # ถ้าเป็น MinMaxScaler
    try:
        if hasattr(SCALER, "feature_range"):
            logging.info(f"[DEBUG] MinMaxScaler.feature_range = {SCALER.feature_range}")
        if hasattr(SCALER, "data_min_"):
            logging.info(f"[DEBUG] MinMaxScaler.data_min_ (last feature) = {SCALER.data_min_[-1]:.6f}")
        if hasattr(SCALER, "data_max_"):
            logging.info(f"[DEBUG] MinMaxScaler.data_max_ (last feature) = {SCALER.data_max_[-1]:.6f}")
    except Exception as e:
        logging.info(f"[DEBUG] อ่านข้อมูล MinMaxScaler ไม่ได้: {e}")
else:
    logging.warning("[DEBUG] SCALER ยังไม่ถูกโหลด")
import hashlib

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

print("MODEL MD5:", md5(MODEL_PATH))
print("SCALER MD5:", md5(SCALER_PATH))


# ========================================
#               AQI CONVERTER
# ========================================
def pm25_to_aqi_th(pm25: float) -> dict:
    if pm25 is None or pm25 < 0:
        return {"aqi": None, "level": None}
    pm25 = float(pm25)
    ranges = [
        (0.0, 15.0, 0, 25, "ดีมาก"),
        (15.0, 25.0, 26, 50, "ดี"),
        (25.0, 37.5, 51, 100, "ปานกลาง"),
        (37.5, 90.0, 101, 200, "เริ่มมีผลกระทบต่อสุขภาพ"),
        (90.0, 999.0, 201, 300, "มีผลกระทบต่อสุขภาพ"),
    ]
    for c_low, c_high, i_low, i_high, level in ranges:
        if pm25 <= c_high:
            aqi = i_low + (i_high - i_low) * (pm25 - c_low) / (c_high - c_low)
            return {"aqi": int(round(aqi)), "level": level}
    return {"aqi": 300, "level": "มีผลกระทบต่อสุขภาพ"}

# ========================================
#               LOGGING
# ========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

# ========================================
#               FASTAPI APP
# ========================================
app = FastAPI(
    title="AirBKK Real-time & Forecast API",
    description="ดึงข้อมูลฝุ่น PM2.5 ทุกชั่วโมง + พยากรณ์ล่วงหน้า 24 ชม. ด้วย AI",
    version="2.0"
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
scheduler.add_job(run_pipeline, "cron", minute=5, hour="*", id="airbkk_pipeline_5", replace_existing=True)
scheduler.add_job(run_pipeline, "cron", minute=15, hour="*", id="airbkk_pipeline_15", replace_existing=True)

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
        pass
    scheduler.start()
    logging.info("Scheduler เริ่มแล้ว → ทุกชั่วโมง นาทีที่ 5 และ 15")
    asyncio.create_task(run_pipeline())

# ========================================
#               ENDPOINTS
# ========================================
@app.get("/")
async def root():
    return {
        "message": "AirBKK API พร้อมใช้งาน",
        "model_loaded": MODEL is not None,
        "scaler_loaded": SCALER is not None,
        "endpoints": {"/latest": "ข้อมูลล่าสุด", "/predict": "พยากรณ์ 24 ชม.", "/resync": "รีโหลด DB"}
    }

@app.get("/latest")
async def latest_readings(db=Depends(get_db)):
    try:
        query = text('SELECT * FROM air_readings ORDER BY "Date_Time" DESC LIMIT 10')
        result = await db.execute(query)
        rows = result.fetchall()
        data = [dict(row._mapping) for row in rows]
        return {"data": data[::-1]}
    except Exception as e:
        logging.exception("latest_readings error")
        raise HTTPException(500, detail=str(e))

@app.post("/resync")
async def resync_db():
    if not CLEANED_FILE.exists():
        raise HTTPException(404, detail="ไม่พบ cleaned_air.csv")
    df = pd.read_csv(CLEANED_FILE, encoding="utf-8-sig")
    count = await load_and_backup(df, force_resync=True)
    return {"status": "resync สำเร็จ", "rows_loaded": count}

# ========================================
#               PREDICT 24 ชม. (เวอร์ชันสมบูรณ์ + log เต็ม)
# ========================================
@app.get("/predict")
async def predict_next_24h():
    if MODEL is None or SCALER is None:
        raise HTTPException(503, "Model/Scaler not loaded")

    async with AsyncSessionLocal() as db:
        # 1. ดึงข้อมูล 48 ชม. ล่าสุด
        query = text('''
            SELECT "Date_Time", 
                   "PM10 (µg/m³)" as pm10, "WS (m/s)" as ws, "WD" as wd, 
                   "Temp (°C)" as temp, "RH (%)" as rh, "BP (mBar)" as bp,
                   "PM2.5 (µg/m³)" as pm25
            FROM air_readings ORDER BY "Date_Time" DESC LIMIT 48
        ''')
        result = await db.execute(query)
        rows = result.fetchall()
        
        if not rows:
             raise HTTPException(503, "No data in DB")

        # เตรียม DataFrame
        df = pd.DataFrame(rows, columns=["Date_Time", "pm10", "ws", "wd", "temp", "rh", "bp", "pm25"])
        df["Date_Time"] = pd.to_datetime(df["Date_Time"])
        df = df.sort_values("Date_Time").reset_index(drop=True)
        
        # เก็บค่าปัจจุบัน
        current_pm25 = float(df["pm25"].iloc[-1])
        current_aqi = pm25_to_aqi_th(current_pm25)

        # 2. [แก้จุดที่ 1] Scale ข้อมูลดิบ (N, 7) ก่อนสร้าง Lag (ให้เหมือนตอน Train เป๊ะๆ)
        # FEATURES_ORDER ต้องตรงกับตอน Train: ["PM10", "WS", "WD", "Temp", "RH", "BP", "PM2.5"]
        data_raw = df[FEATURES_ORDER].values.astype(float)
        
        try:
            data_scaled = SCALER.transform(data_raw)
        except ValueError as e:
            return {"error": f"Scaler mismatch: {e}", "hint": "Check feature count (must be 7)"}

        # 3. สร้าง Input ก้อนแรก (24 ชม. ล่าสุด)
        # Shape: (1, 168) -> Flatten จาก (24, 7)
        # ต้องมั่นใจว่า data_scaled มีอย่างน้อย 24 แถว
        if len(data_scaled) < 24:
             raise HTTPException(503, "Not enough data for lag")
             
        # เอา 24 ชม. ล่าสุดมาเป็น window ตั้งต้น
        current_window = data_scaled[-24:].copy() 

        preds_actual = []
        hours = []
        
        # 4. [แก้จุดที่ 2] ลูปพยากรณ์ที่ถูกต้อง (เลื่อนทั้ง Window)
        for i in range(24):
            # Flatten เพื่อเข้าโมเดล (1, 168)
            X_flat = current_window.flatten().reshape(1, -1)
            
            # พยากรณ์ (ได้ค่า Scaled)
            pred_scaled = float(MODEL.predict(X_flat)[0])
            
            # แปลงค่ากลับเป็นหน่วยจริง
            dummy_row = np.zeros((1, 7))
            dummy_row[0, -1] = pred_scaled # ตำแหน่งสุดท้ายคือ PM2.5
            pred_val = SCALER.inverse_transform(dummy_row)[0, -1]
            
            # เก็บผลลัพธ์ (กันค่าติดลบ)
            final_val = max(0.0, round(float(pred_val), 1))
            preds_actual.append(final_val)
            hours.append((datetime.now(TH_TZ) + timedelta(hours=i+1)).strftime("%H:%M"))
            
            # อัปเดต Window:
            # สร้างแถวใหม่: ใช้ Weather เดิม (Persistence) แต่เปลี่ยน PM2.5 เป็นค่าที่เพิ่งพยากรณ์ได้
            next_row = current_window[-1].copy() 
            next_row[-1] = pred_scaled  
            
            # ตัดแถวบนสุดทิ้ง + ต่อแถวใหม่ด้านล่าง
            current_window = np.vstack([current_window[1:], next_row])

        max_val = max(preds_actual)
        max_aqi = pm25_to_aqi_th(max_val)

        return {
            "ค่าฝุ่น pm2.5 ปัจจุบัน": round(current_pm25, 1),
            "ดัชนีคุณภาพอากาศปัจจุบัน": current_aqi["aqi"],
            "ระดับคุณภาพอากาศปัจจุบัน": current_aqi["level"],
            "พยากรณ์ฝุ่น_24ชั่วโมง": dict(zip(hours, preds_actual)),
            "PM2.5 สูงสุดที่คาดการณ์": max_val,
            "AQI สูงสุดที่คาดการณ์": max_aqi["aqi"],
            "ระดับโดยรวมจากค่าสูงสุด": max_aqi["level"],
            "สรุปผล": f"24 ชม. ข้างหน้า ฝุ่นPM2.5สูงสุด {max_val} µg/m³ → {max_aqi['level']}"
        }
        # except Exception as e:
        #     logging.exception("Predict ล้มเหลวอย่างแรง!")
        #     raise HTTPException(500, f"พยากรณ์ล้มเหลว: {str(e)}")

# ========================================
#               HEALTH CHECK
# ========================================
@app.get("/health")
async def health():
    rows = len(pd.read_csv(CLEANED_FILE)) if CLEANED_FILE.exists() else 0
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "scaler_loaded": SCALER is not None,
        "data_rows": rows,
        "timestamp": datetime.now(TH_TZ).isoformat()
    }
@app.get("/debug-scaler")
async def debug_scaler_info():
    """ดูข้อมูล scaler ที่โหลดไว้"""
    if SCALER is None:
        return {"error": "Scaler not loaded"}
    
    info = {
        "class": SCALER.__class__.__name__,
        "n_features": SCALER.n_features_in_,
        "scaler_md5": md5(SCALER_PATH),
        "model_md5": md5(MODEL_PATH),
    }
    
    if hasattr(SCALER, 'data_min_'):
        info["pm25_data_min"] = float(SCALER.data_min_[6])
        info["pm25_data_max"] = float(SCALER.data_max_[6])
        info["pm25_scale"] = float(SCALER.scale_[6])
    
    # ทดสอบ transform
    test = np.array([[50, 2, 180, 30, 60, 1010, 10.5]])
    scaled = SCALER.transform(test)
    back = SCALER.inverse_transform(scaled)
    
    info["test_transform"] = {
        "input_pm25": 10.5,
        "scaled_pm25": float(scaled[0, 6]),
        "back_pm25": float(back[0, 6]),
        "error": float(abs(back[0, 6] - 10.5))
    }
    
    return info