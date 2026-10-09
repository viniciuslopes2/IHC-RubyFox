import os
import sqlite3

from fastapi import FastAPI, HTTPException

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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


class EstoqueRepository:
    """Cuida de todo o acesso ao SQLite do estoque."""

    def __init__(self, db_filename: str = "estoque.db"):
        self.db_path = os.path.join(BASE_DIR, db_filename)

    def create_and_seed(self) -> None:
        # Recria a tabela do zero a cada execucao
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS estoque")
            conn.execute(SCHEMA)
            conn.executemany(
                "INSERT INTO estoque "
                "(produto, departamento, data_fabricacao, data_validade, quantidade, preco) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                SEED_DATA,
            )

    def run_query(self, sql_query: str) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(sql_query)
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            conn.close()


repository = EstoqueRepository()
repository.create_and_seed()

app = FastAPI(title="Estoque API")


@app.get("/query")
def query(sql: str):
    # A validacao (apenas SELECT) e feita no gerador, do lado do bot
    try:
        return {"results": repository.run_query(sql)}
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
