# app/schemas.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class AirReading(BaseModel):
    Date_Time: datetime
    PM10: Optional[float] = None
    PM25: Optional[float] = None
    WS: Optional[float] = None
    WD: Optional[float] = None
    Temp: Optional[float] = None
    RH: Optional[float] = None
    BP: Optional[float] = None

    class Config:
        from_attributes = True