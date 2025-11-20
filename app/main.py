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
        logging.info(f"เติมข้อมูลย้อนหลัง: +{added} แถว")

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
        readings = await upsert_air_reading(db, limit=10)
        return {"data": [r.__dict__ for r in readings]}
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
            # 1) ดึงข้อมูลย้อนหลัง (เรียงจากเก่า->ใหม่)
            query = text(f'SELECT "Date_Time", "PM2.5 (µg/m³)" as pm25 FROM air_readings ORDER BY "Date_Time" DESC LIMIT {DB_FETCH_LIMIT}')
            result = await db.execute(query)
            rows = result.fetchall()
            if not rows:
                raise HTTPException(400, "ไม่มีข้อมูลในฐานข้อมูล")
            df = pd.DataFrame(rows, columns=["Date_Time", "pm25"])
            df["Date_Time"] = pd.to_datetime(df["Date_Time"])
            df = df.sort_values("Date_Time").reset_index(drop=True)

            # 2) เช็คว่ามีแถวเพียงพอสำหรับ lag
            min_rows = MAX_LAG + 1
            if len(df) < min_rows:
                raise HTTPException(400, f"ข้อมูลไม่พอ: ต้องมีอย่างน้อย {min_rows} แถวเพื่อคำนวณ lag_{MAX_LAG}. ตอนนี้มี {len(df)} แถว")

            # 3) สร้าง features: lag, rolling, time
            for i in range(1, MAX_LAG + 1):
                df[f'lag_{i}'] = df['pm25'].shift(i)

            for w in ROLL_WINDOWS:
                df[f'roll_mean_{w}'] = df['pm25'].rolling(w).mean()
                df[f'roll_std_{w}'] = df['pm25'].rolling(w).std()

            df['hour'] = df['Date_Time'].dt.hour
            df['weekday'] = df['Date_Time'].dt.weekday
            df['month'] = df['Date_Time'].dt.month
            # is_weekend เป็น optional — เพิ่มถ้าโมเดลเทรนด้วย
            df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

            # one-hot
            df = pd.get_dummies(df, columns=['hour', 'weekday', 'month'], prefix=['hour', 'weekday', 'month'], dtype=int)

            # เติม dummy ที่ขาด (รับประกัน hour_0..23, weekday_0..6, month_1..12)
            for h in range(24):
                col = f'hour_{h}'
                if col not in df.columns:
                    df[col] = 0
            for d in range(7):
                col = f'weekday_{d}'
                if col not in df.columns:
                    df[col] = 0
            for m in range(1, 13):
                col = f'month_{m}'
                if col not in df.columns:
                    df[col] = 0

            # 4) dropna แล้วเอา latest
            df_clean = df.dropna().reset_index(drop=True)
            if df_clean.empty:
                raise HTTPException(400, "หลัง dropna แล้วไม่มีแถวเหลือ — ข้อมูลย้อนหลังอาจไม่พอสำหรับ rolling/lag")

            latest = df_clean.iloc[-1]

            # 5) สร้าง expected feature order (ชัดเจน)
            expected = []
            expected += [f'lag_{i}' for i in range(1, MAX_LAG + 1)]
            for w in ROLL_WINDOWS:
                expected.append(f'roll_mean_{w}')
                expected.append(f'roll_std_{w}')
            expected += [f'hour_{h}' for h in range(24)]
            expected += [f'weekday_{d}' for d in range(7)]
            expected += [f'month_{m}' for m in range(1, 13)]
            # ถ้าตอนเทรนมี 'is_weekend' ให้เพิ่ม (uncomment ถ้าจำเป็น)
            # expected.append('is_weekend')

            # 6) เติมคอลัมน์ missing เป็น 0 เพื่อความปลอดภัย และจัด feature_cols
            for col in expected:
                if col not in df_clean.columns:
                    df_clean[col] = 0

            # เก็บคอลัมน์อื่นๆ ที่เหลือ (ถ้ามี) ต่อท้าย (จะไม่ตัดทิ้งทันที)
            remaining = [c for c in df_clean.columns if c not in (["Date_Time", "pm25"] + expected)]
            feature_cols = expected + sorted(remaining)

            # 7) สร้าง X จาก latest ตาม feature_cols
            X = latest[feature_cols].astype(float).values.reshape(1, -1)
            actual_n = X.shape[1]

            # 8) หาจำนวน features ที่โมเดลคาดไว้ (fallback เป็นค่าที่ตั้งไว้)
            model_n = MODEL_EXPECTED_N if MODEL_EXPECTED_N is not None else FALLBACK_TARGET_FEATURE_COUNT
            logging.info(f"feature built: actual_n={actual_n}, model_expected_n={model_n}")

            # 9) ปรับให้เข้ากับโมเดล (เติม 0 หรือ ตัดคอลัมน์ท้าย) — ทำเฉพาะเมื่อ model_n เป็น int
            if model_n is not None:
                if actual_n < model_n:
                    pad_n = model_n - actual_n
                    logging.warning(f"ฟีเจอร์น้อยกว่าโมเดลต้องการ: เติม {pad_n} ค่า 0 ต่อท้าย")
                    X = np.hstack([X, np.zeros((1, pad_n))])
                    feature_cols += [f'pad_{i}' for i in range(pad_n)]
                    actual_n = X.shape[1]
                elif actual_n > model_n:
                    logging.warning(f"ฟีเจอร์มากกว่าโมเดลต้องการ: ตัด {actual_n - model_n} ค่า ท้ายรายการ")
                    X = X[:, :model_n]
                    feature_cols = feature_cols[:model_n]
                    actual_n = X.shape[1]

            # ถ้าไม่มี metadata และเราอยากบังคับ exact count ให้ใช้ FALLBACK_TARGET_FEATURE_COUNT:
            if MODEL_EXPECTED_N is None:
                if actual_n != FALLBACK_TARGET_FEATURE_COUNT:
                    raise HTTPException(500, f"Feature mismatch (no model metadata): ได้ {actual_n}, แต่คาดว่าจะเป็น {FALLBACK_TARGET_FEATURE_COUNT}. ตรวจสอบ expected list หรือโมเดล")

            # 10) เริ่มพยากรณ์ autoregressive 24 ชม. (พร้อม debug)
            preds = []
            raw_preds = []
            inv_log1p_preds = []
            current = X.copy()
            col_index = {c: i for i, c in enumerate(feature_cols)}
            base_dt = df_clean["Date_Time"].iloc[-1]
            recent_vals = df_clean['pm25'].tail(24).values if len(df_clean) >= 24 else df_clean['pm25'].values
            recent_mean = float(np.mean(recent_vals)) if len(recent_vals) > 0 else None

            for h in range(24):
                try:
                    raw_val = float(MODEL.predict(current)[0])
                except Exception as ex:
                    logging.exception("Predict call failed")
                    raise HTTPException(500, f"เรียกโมเดลพยากรณ์ล้มเหลว: {ex}")

                # เก็บไว้ debug
                raw_preds.append(raw_val)

                # --- แปลงกลับจาก scaled → µg/m³ ---
                SCALER_MAX = 346.0
                actual_pred = raw_val * SCALER_MAX
                final_pred = max(0.0, actual_pred)
                preds.append(round(final_pred, 1))

                # --- สำคัญ: อัปเดต lag ถัดไปด้วย scaled value (ไม่ใช่ actual!) ---
                if 'lag_1' in col_index:
                    lag_indices = [col_index.get(f'lag_{i}') for i in range(1, MAX_LAG + 1)]
                    lag_indices = [idx for idx in lag_indices if idx is not None]

                    if len(lag_indices) == MAX_LAG:
                        current[0, lag_indices] = np.roll(current[0, lag_indices], -1)
                        current[0, lag_indices[0]] = raw_val  # ใช้ scaled value!
                    elif len(lag_indices) > 0:
                        current[0, lag_indices] = np.roll(current[0, lag_indices], -1)
                        current[0, lag_indices[0]] = raw_val

                # --- อัปเดต time features (เหมือนเดิม) ---
                future_dt = base_dt + timedelta(hours=h+1)
                new_hour = future_dt.hour
                new_wd = future_dt.weekday()
                new_m = future_dt.month

                for i in range(24):
                    cname = f'hour_{i}'
                    if cname in col_index:
                        current[0, col_index[cname]] = 1 if i == new_hour else 0
                for i in range(7):
                    cname = f'weekday_{i}'
                    if cname in col_index:
                        current[0, col_index[cname]] = 1 if i == new_wd else 0
                for i in range(1, 13):
                    cname = f'month_{i}'
                    if cname in col_index:
                        current[0, col_index[cname]] = 1 if i == new_m else 0
                if 'is_weekend' in col_index:
                    current[0, col_index['is_weekend']] = 1 if new_wd in (5, 6) else 0

                # --- update time dummies (hour, weekday, month) for next step ---
                future_dt = base_dt + timedelta(hours=h+1)
                new_hour = future_dt.hour
                new_wd = future_dt.weekday()
                new_m = future_dt.month

                for i in range(24):
                    cname = f'hour_{i}'
                    if cname in col_index:
                        current[0, col_index[cname]] = 1 if i == new_hour else 0
                for i in range(7):
                    cname = f'weekday_{i}'
                    if cname in col_index:
                        current[0, col_index[cname]] = 1 if i == new_wd else 0
                for i in range(1, 13):
                    cname = f'month_{i}'
                    if cname in col_index:
                        current[0, col_index[cname]] = 1 if i == new_m else 0
                if 'is_weekend' in col_index:
                    current[0, col_index['is_weekend']] = 1 if new_wd in (5, 6) else 0

            # หลัง loop: วิเคราะห์ผล raw vs inv_log1p
            raw_mean = float(np.mean(raw_preds)) if raw_preds else None
            inv_log1p_mean = None
            inv_log1p_valid = False
            if any(v is not None for v in inv_log1p_preds):
                inv_vals = [v for v in inv_log1p_preds if v is not None]
                if inv_vals:
                    inv_log1p_mean = float(np.mean(inv_vals))
                    # heuristic: ถ้า inv_log1p_mean อยู่ในช่วงที่สมเหตุสมผลเมื่อเทียบ recent_mean -> mark as plausible
                    if recent_mean is not None and recent_mean > 0:
                        ratio = inv_log1p_mean / recent_mean
                        if 0.2 <= ratio <= 5:
                            inv_log1p_valid = True

            debug_out = {
                "model_expected_n": MODEL_EXPECTED_N,
                "used_feature_count": actual_n,
                "feature_cols_sample": feature_cols[:10] + (feature_cols[-10:] if len(feature_cols) > 20 else []),
                "recent_mean": recent_mean,
                "raw_preds_sample": [round(float(x), 6) for x in raw_preds],
                "raw_mean": round(raw_mean, 6) if raw_mean is not None else None,
                "inv_log1p_sample": [round(float(x), 6) if x is not None else None for x in inv_log1p_preds],
                "inv_log1p_mean": round(inv_log1p_mean, 6) if inv_log1p_mean is not None else None,
                "inv_log1p_plausible": inv_log1p_valid
            }

            hours = [(datetime.now() + timedelta(hours=i+1)).strftime("%H:%M") for i in range(24)]
            max_val = max(preds) if preds else 0.0
            level = "ดี" if max_val <= 25 else "ปานกลาง" if max_val <= 50 else "ปานกลาง-อันตราย" if max_val <= 90 else "อันตราย"

            return {
                "forecast_24h": dict(zip(hours, preds)),
                "max_pm25_next_24h": round(max_val, 1),
                "alert_level": level,
                "message": f"24 ชม. ข้างหน้า ฝุ่นสูงสุด {max_val:.1f} µg/m³ → {level} ค่ะ"
                
            }

        except HTTPException:
            raise
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
