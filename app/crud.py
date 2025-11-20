# app/crud.py
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import AirReading
from app.schemas import AirReading as AirReadingSchema
from typing import Optional

async def upsert_air_reading(db: AsyncSession, obj: AirReadingSchema):
    """
    ใส่หรืออัปเดตข้อมูล โดยใช้ Date_Time เป็น key
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

    # ใช้ on_conflict_do_nothing() สำหรับ PostgreSQL
    stmt = stmt.on_conflict_do_nothing(
        index_elements=['Date_Time']
    )

    await db.execute(stmt)
    await db.commit()