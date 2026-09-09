import os
import sqlite3
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException


@dataclass(frozen=True)
class EstoqueConfig:
    """Configuracao de acesso ao banco de estoque."""

    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    db_filename: str = "estoque.db"

    @property
    def db_path(self) -> str:
        return os.path.join(self.base_dir, self.db_filename)


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
    """Encapsula todo o acesso ao SQLite do estoque (conexao, schema, queries)."""

    def __init__(self, config: EstoqueConfig):
        self._config = config

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._config.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_and_seed(self) -> None:
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS estoque")
            cursor.execute(SCHEMA)
            cursor.executemany(
                "INSERT INTO estoque "
                "(produto, departamento, data_fabricacao, data_validade, quantidade, preco) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                SEED_DATA,
            )
            conn.commit()
        finally:
            conn.close()

    def run_query(self, sql_query: str) -> list[dict]:
        conn = self._connect()
        try:
            cursor = conn.execute(sql_query)
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            conn.close()


def create_app(config: EstoqueConfig | None = None) -> FastAPI:
    config = config or EstoqueConfig()
    repository = EstoqueRepository(config)
    repository.create_and_seed()

    app = FastAPI(title="Estoque API")

    @app.get("/query")
    def query(sql: str):
        try:
            results = repository.run_query(sql)
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"results": results}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
