import sqlite3
import os
from config import SQLITE_DB_PATH


def get_connection():
    """Return a sqlite3 connection using the application config.

    The returned connection uses `sqlite3.Row` for dict-like access to columns.
    """
    # Ensure directory exists
    db_dir = os.path.dirname(SQLITE_DB_PATH)
    os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn