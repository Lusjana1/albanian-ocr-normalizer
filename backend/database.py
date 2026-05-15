import aiosqlite

DB_PATH = "history.db"


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                cleaned_text TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                processing_time_ms REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()


async def save_result(
    filename: str,
    raw_text: str,
    cleaned_text: str,
    normalized_text: str,
    word_count: int,
    processing_time_ms: float,
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO history
               (filename, raw_text, cleaned_text, normalized_text, word_count, processing_time_ms)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (filename, raw_text, cleaned_text, normalized_text, word_count, processing_time_ms),
        )
        await db.commit()
        return cursor.lastrowid


async def get_history(limit: int = 20) -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM history ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def delete_history_item(item_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM history WHERE id = ?", (item_id,))
        await db.commit()
