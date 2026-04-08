import sqlite3

def create_db():
    conn = sqlite3.connect("company.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        department TEXT,
        salary INTEGER
    )
    """)

    cursor.executemany("""
    INSERT INTO employees (name, department, salary) VALUES (?, ?, ?)
    """, [
        ("Alice", "HR", 50000),
        ("Bob", "IT", 70000),
        ("Charlie", "Finance", 60000)
    ])

    conn.commit()
    conn.close()


def run_query(sql):
    conn = sqlite3.connect("company.db")
    cursor = conn.cursor()

    cursor.execute(sql)
    results = cursor.fetchall()

    conn.close()
    return results
