# app/crud.py  ← เวอร์ชันสุดท้าย ใช้งานจริงได้ทันที
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import AirReading
from app.schemas import AirReading as AirReadingSchema
import logging

async def upsert_air_reading(db: AsyncSession, obj: AirReadingSchema):
    """
    อัปเดตทับเสมอเมื่อมีข้อมูลใหม่มา ไม่สนว่าเคยมีแล้วหรือไม่
    วิธีเดียวที่ทำให้ค่าจริงจากกรมมาทันที ไม่ค้างค่าเก่า
    """
    stmt = insert(AirReading).values(
        Date_Time=obj.Date_Time,
        PM10=obj.PM10,
        PM25=obj.PM25,
        WS=obj.WS,
        WD=obj.WD,
        Temp=obj.Temp,
        RH=obj.RH,
        BP=obj.BP,
    )

    # อัปเดตทับทุกคอลัมน์เสมอ (สำคัญมาก!)
    stmt = stmt.on_conflict_do_update(
        index_elements=['Date_Time'],
        set_={
            "PM10 (µg/m³)": obj.PM10,        
            "PM2.5 (µg/m³)": obj.PM25,       
            "WS (m/s)": obj.WS,
            "WD": obj.WD,
            "Temp (°C)": obj.Temp,
            "RH (%)": obj.RH,
            "BP (mBar)": obj.BP,
        }
    )

    await db.execute(stmt)
    await db.commit()
    # logging.info(f"อัปเดต DB สำเร็จ: {obj.Date_Time} → PM2.5 = {obj.PM25}")