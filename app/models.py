# app/models.py
from sqlalchemy import Column, BigInteger, DateTime, Double
from sqlalchemy.sql import func
from .db import Base

class AirReading(Base):
    __tablename__ = "air_readings"
    __table_args__ = {'schema': 'public'}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    Date_Time = Column(DateTime(timezone=True), unique=True, nullable=False, index=True)

    
    PM10 = Column("PM10 (µg/m³)", Double, nullable=True)
    PM25 = Column("PM2.5 (µg/m³)", Double, nullable=True)
    WS = Column("WS (m/s)", Double, nullable=True)
    WD = Column("WD", Double, nullable=True)
    Temp = Column("Temp (°C)", Double, nullable=True)
    RH = Column("RH (%)", Double, nullable=True)
    BP = Column("BP (mBar)", Double, nullable=True)