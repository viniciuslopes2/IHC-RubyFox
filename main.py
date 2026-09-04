import os
import sqlite3

from fastapi import FastAPI, HTTPException

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "estoque.db")

SCHEMA = """
CREATE TABLE estoque (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    produto TEXT NOT NULL,
    departamento TEXT NOT NULL,
    data_fabricacao TEXT,
    data_validade TEXT,
    quantidade INTEGER,
    preco REAL
)
"""

SEED_DATA = [
    ("sabonete", "higiene", "2026-01-10", "2027-01-10", 120, 4.50),
    ("agua", "bebidas", "2026-06-01", "2026-12-01", 300, 2.00),
    ("coca", "bebidas", "2026-05-15", "2026-11-15", 200, 6.00),
    ("shampoo", "higiene", "2026-02-20", "2027-02-20", 80, 15.90),
    ("suco de laranja", "bebidas", "2026-07-01", "2026-09-01", 60, 7.50),
]


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS estoque")
    c.execute(SCHEMA)

    c.executemany(
        "INSERT INTO estoque (produto, departamento, data_fabricacao, data_validade, quantidade, preco) VALUES (?, ?, ?, ?, ?, ?)",
        SEED_DATA,
    )

    conn.commit()
    conn.close()


def run_query(sql_query):
    conn = get_connection()
    try:
        cursor = conn.execute(sql_query)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()


app = FastAPI(title="Estoque API")

create_db()


@app.get("/query")
def query(sql: str):
    try:
        results = run_query(sql)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))
    return {"results": results}
