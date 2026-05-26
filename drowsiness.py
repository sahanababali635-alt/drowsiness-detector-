from db_connection import get_connection
from datetime import datetime

def save_drowsiness(driver_id, status):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        INSERT INTO drowsiness_log (driver_id, status, timestamp)
        VALUES (%s, %s, %s)
    """
    values = (driver_id, status, datetime.now())

    cursor.execute(query, values)
    conn.commit()

    print(f"✅ Data saved: Driver {driver_id} is {status}")

    cursor.close()
    conn.close()

# ನಿಮ್ಮ drowsiness detection loop ನಲ್ಲಿ ಉಪಯೋಗಿಸಿ:
# if ear_ratio < threshold:
#     save_drowsiness("D001", "drowsy")