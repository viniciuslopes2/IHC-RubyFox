import os
import re
import sqlite3
import tempfile

import dspy
import requests
import telebot
import whisper

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIMIZED_PROGRAM_PATH = os.path.join(BASE_DIR, "optimized_generator.json")
API_URL = "http://127.0.0.1:8000/query"

FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "CREATE", "REPLACE", "TRUNCATE", "ATTACH", "PRAGMA",
)

SCHEMA_DDL = """
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

SCHEMA_DESCRIPTION = """
CREATE TABLE estoque (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  produto VARCHAR(50),
  departamento VARCHAR(50),
  data_fabricacao DATE,
  data_validade DATE,
  quantidade INTEGER,
  preco REAL
);
"""

lm = dspy.LM('openai/gemma-3-4b-instruct', api_base='http://localhost:1234/v1', api_key='not-needed')
dspy.configure(lm=lm)


class TextToSQL(dspy.Signature):
    """Generate SQL from natural language.

        Database schema:
          - estoque: id, produto, departamento, data_fabricacao, data_validade, quantidade, preco
    """
    dbschema = dspy.InputField(desc="Database schema")
    question = dspy.InputField(desc="Natural language question")

    sql_query = dspy.OutputField(desc="Valid SQL query")


class ReliableSQLGenerator(dspy.Module):
    def __init__(self, schema=SCHEMA_DESCRIPTION):
        super().__init__()
        self.schema = schema
        self.generate_sql = dspy.ChainOfThought(TextToSQL)

    def forward(self, question):
        pred = self.generate_sql(dbschema=self.schema, question=question)
        return pred


class SQLValidationError(Exception):
    def __init__(self, sql_query, original_error):
        self.sql_query = sql_query
        self.original_error = original_error
        super().__init__(f"Query invalida: {original_error}")


class SelectOnlyError(Exception):
    def __init__(self, sql_query, keyword):
        self.sql_query = sql_query
        self.keyword = keyword
        super().__init__(f"Apenas SELECT e permitido, encontrado: {keyword}")


CODE_FENCE_RE = re.compile(r"^```(?:sql)?\s*\n?(.*?)\n?```$", re.IGNORECASE | re.DOTALL)


def strip_code_fence(sql_query):
    match = CODE_FENCE_RE.match(sql_query.strip())
    if match:
        return match.group(1).strip()
    return sql_query.strip()


def check_select_only(sql_query):
    statements = [s.strip() for s in sql_query.split(";") if s.strip()]

    if len(statements) != 1:
        raise SelectOnlyError(sql_query, "multiplas instrucoes")

    statement = statements[0]

    if not statement.upper().startswith("SELECT"):
        raise SelectOnlyError(sql_query, statement.split()[0].upper())

    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", statement, re.IGNORECASE):
            raise SelectOnlyError(sql_query, keyword)


def validate_sql(sql_query):
    check_select_only(sql_query)

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(SCHEMA_DDL)
        conn.execute(sql_query)
    except sqlite3.Error as error:
        raise SQLValidationError(sql_query, error)
    finally:
        conn.close()


generator = ReliableSQLGenerator()
if os.path.exists(OPTIMIZED_PROGRAM_PATH):
    generator.load(OPTIMIZED_PROGRAM_PATH)


def generate_sql(question):
    try:
        prediction = generator(question=question)
    except Exception as error:
        raise SQLValidationError(None, error)

    sql_query = strip_code_fence(prediction.sql_query)
    validate_sql(sql_query)

    return sql_query


def run_remote_query(sql_query):
    response = requests.get(API_URL, params={"sql": sql_query}, timeout=10)
    response.raise_for_status()
    return response.json()["results"]


def format_results(results):
    if not results:
        return "Nenhum resultado encontrado."
    return "\n".join(str(row) for row in results)


whisper_model = whisper.load_model("base")

bot = telebot.TeleBot(os.environ["TELEGRAM_BOT_TOKEN"])


def answer_question(message, question):
    try:
        sql_query = generate_sql(question)
    except (SQLValidationError, SelectOnlyError) as error:
        bot.reply_to(message, f"Nao consegui gerar uma consulta valida: {error}")
        return

    try:
        results = run_remote_query(sql_query)
    except requests.RequestException as error:
        bot.reply_to(message, f"Erro ao consultar a API: {error}")
        return

    bot.reply_to(message, format_results(results))


@bot.message_handler(commands=["start"])
def handle_start(message):
    bot.reply_to(message, "Pergunte algo sobre o estoque, por texto ou audio.")


@bot.message_handler(content_types=["text"])
def handle_text(message):
    answer_question(message, message.text)


@bot.message_handler(content_types=["voice"])
def handle_voice(message):
    file_info = bot.get_file(message.voice.file_id)
    audio_bytes = bot.download_file(file_info.file_path)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    try:
        transcription = whisper_model.transcribe(tmp_path, language="portuguese")
        question = transcription["text"].strip()
    finally:
        os.remove(tmp_path)

    answer_question(message, question)


if __name__ == "__main__":
    bot.infinity_polling()
