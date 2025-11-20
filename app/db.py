# db.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, BigInteger, DateTime, Double
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set")

# แปลงให้เป็น async driver สำหรับ SQLAlchemy
# จาก postgresql:// → postgresql+asyncpg://
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


class AirReading(Base):
    __tablename__ = "air_readings"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    Date_Time = Column(DateTime(timezone=True), nullable=False, unique=True, index=True)
    PM10 = Column(Double, nullable=True)
    PM25 = Column(Double, nullable=True)
    WS = Column(Double, nullable=True)
    Temp = Column(Double, nullable=True)
    RH = Column(Double, nullable=True)
    BP = Column(Double, nullable=True)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
