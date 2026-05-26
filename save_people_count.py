def save_people_count(bus_id, entry, exit_count):
    conn = get_connection()
    cursor = conn.cursor()

    total = entry - exit_count  # ಬಸ್ ನಲ್ಲಿ ಇರುವ ಒಟ್ಟು ಜನ

    query = """
        INSERT INTO people_count (bus_id, entry_count, exit_count, total_passengers)
        VALUES (%s, %s, %s, %s)
    """
    values = (bus_id, entry, exit_count, total)

    cursor.execute(query, values)
    conn.commit()

    print(f"🚌 Bus {bus_id}: {total} passengers inside")

    cursor.close()
    conn.close()