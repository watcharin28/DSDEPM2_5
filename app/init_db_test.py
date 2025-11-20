
from app.db import init_db, engine
import asyncio

async def main():
    print("1. กำลังเชื่อมต่อไปยัง airdb...")
    async with engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: print("เชื่อมต่อสำเร็จ!"))
    
    print("2. กำลังสร้างตาราง air_readings...")
    await init_db()
    print("สร้างตารางสำเร็จ!")

asyncio.run(main())
