import os
import re
import sqlite3
import sys
import tempfile
import threading

import dspy
import requests
import telebot
import whisper

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIMIZED_PATH = os.path.join(BASE_DIR, "optimized_generator.json")
EVAL_DB = "estoque_eval.db"

API_URL = "http://127.0.0.1:8000/query"
LM_MODEL = "openai/gemma-3-4b-instruct"
LM_API_BASE = "http://localhost:1234/v1"
LM_API_KEY = "not-needed"
# O "professor" do GEPA pode ser outro modelo, via variavel de ambiente
REFLECTION_LM_MODEL = os.environ.get("GEPA_REFLECTION_LM", LM_MODEL)
REFLECTION_LM_API_BASE = os.environ.get("GEPA_REFLECTION_LM_API_BASE", LM_API_BASE)
REFLECTION_LM_API_KEY = os.environ.get("GEPA_REFLECTION_LM_API_KEY", LM_API_KEY)

MAX_ITERS = 6
MAX_RECURSION_DEPTH = 2

FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "CREATE", "REPLACE", "TRUNCATE", "ATTACH", "PRAGMA",
)

# Mesmo schema do main.py, usado para testar o SQL num banco em memoria
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

CODE_FENCE_RE = re.compile(r"^```(?:sql)?\s*\n?(.*?)\n?```$", re.IGNORECASE | re.DOTALL)


class InvalidSQLError(Exception):
    def __init__(self, sql_query, reason):
        self.sql_query = sql_query
        super().__init__(f"Query invalida: {reason}")


class SQLValidator:
    @staticmethod
    def strip_code_fence(sql_query: str) -> str:
        match = CODE_FENCE_RE.match(sql_query.strip())
        return match.group(1).strip() if match else sql_query.strip()

    def check_select_only(self, sql_query: str) -> None:
        statements = [s.strip() for s in sql_query.split(";") if s.strip()]
        if len(statements) != 1:
            raise InvalidSQLError(sql_query, "multiplas instrucoes")

        statement = statements[0]
        if not statement.upper().startswith("SELECT"):
            raise InvalidSQLError(sql_query, f"apenas SELECT e permitido, encontrado {statement.split()[0].upper()}")

        for keyword in FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{keyword}\b", statement, re.IGNORECASE):
                raise InvalidSQLError(sql_query, f"apenas SELECT e permitido, encontrado {keyword}")

    def validate(self, sql_query: str) -> None:
        self.check_select_only(sql_query)

        # Executa num banco vazio so para pegar erros de sintaxe, tabela ou coluna
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(SCHEMA)
            conn.execute(sql_query)
        except sqlite3.Error as error:
            raise InvalidSQLError(sql_query, error) from error
        finally:
            conn.close()


class TextToSQL(dspy.Signature):
    """Generate a valid SQLite SELECT query from a natural language question,
    given the database schema.

        Database schema:
          - estoque: id, produto, departamento, data_fabricacao, data_validade, quantidade, preco

    Column selection rule: default to `SELECT *` for any question about which
    product(s) match a condition, even when the condition itself filters,
    orders, or ranks by a specific column. Filtering/sorting/ranking BY a
    column (preco, data_validade, quantidade) is NOT the same as asking FOR
    that column's value:
      - "quais produtos sao de bebidas" -> SELECT *
      - "quais produtos custam mais de 5 reais" -> SELECT * (filters by preco,
        does not ask for preco)
      - "qual o produto mais caro" / "mais barato" -> SELECT * (ranks by
        preco, does not ask for preco)
      - "qual produto vence primeiro" / "vencem antes de X" -> SELECT * (ranks
        or filters by data_validade, does not ask for the date)
      - "existe produto X" -> SELECT * WHERE produto = 'X' (exact match with
        `=`, never LIKE, unless the question explicitly asks for similar or
        partial names)
    Only select specific columns when the question explicitly asks to see
    that one attribute's value itself (e.g. "qual o preco de X" -> preco,
    "quantas unidades de X" -> quantidade).

    `previous_sql` and `previous_error` are empty on the first attempt. When they
    are filled, it means a previous attempt failed validation/execution: analyze
    `previous_error` and produce a corrected `sql_query` that fixes the problem,
    instead of repeating the same mistake.
    """
    dbschema = dspy.InputField(desc="Database schema")
    question = dspy.InputField(desc="Natural language question")
    previous_sql = dspy.InputField(desc="Previous failed SQL attempt, empty on the first try")
    previous_error = dspy.InputField(desc="Error raised by the previous attempt, empty on the first try")

    sql_query = dspy.OutputField(desc="Valid SQL query")


class RLMTask(dspy.Signature):
    """Produce a single valid SQLite SELECT query that answers `question`
    against the estoque table described by `dbschema`, using the tools
    instead of writing SQL directly.

    Call draft_sql to get a candidate query, then check_sql to validate it.
    If check_sql reports an error, call draft_sql again passing the same
    sql_query as previous_sql and the error as previous_error, so the next
    candidate fixes it. If `question` is really two or more independent
    questions joined together, call recurse once per independent part
    instead of drafting a single query for the whole thing. Only finish with
    a sql_query that check_sql has approved as valid.
    """
    dbschema = dspy.InputField(desc="Database schema")
    question = dspy.InputField(desc="Natural language question")

    sql_query = dspy.OutputField(desc="Single valid SQL SELECT query, already approved by check_sql")


_recursion_depth = threading.local()


class RLMSQLGenerator(dspy.Module):
    def __init__(self, max_iters: int = MAX_ITERS, max_depth: int = MAX_RECURSION_DEPTH):
        super().__init__()
        self.validator = SQLValidator()
        self.max_depth = max_depth
        self.generate_sql = dspy.Predict(TextToSQL)
        self.agent = dspy.ReAct(RLMTask, tools=[self.draft_sql, self.check_sql, self.recurse], max_iters=max_iters)

    def draft_sql(self, question: str, previous_sql: str = "", previous_error: str = "") -> str:
        """Gera um candidato de SQL para a pergunta. Numa nova tentativa, passe o
        SQL e o erro anteriores em previous_sql/previous_error para corrigi-lo."""
        pred = self.generate_sql(
            dbschema=SCHEMA,
            question=question,
            previous_sql=previous_sql,
            previous_error=previous_error,
        )
        return self.validator.strip_code_fence(pred.sql_query)

    def check_sql(self, sql_query: str) -> str:
        """Valida um SQL contra o schema real. Devolve 'valido' ou o erro encontrado."""
        try:
            self.validator.validate(sql_query)
            return "valido"
        except InvalidSQLError as error:
            return f"invalido: {error}"

    def recurse(self, sub_question: str) -> str:
        """Chama este mesmo agente recursivamente para resolver uma sub-pergunta
        independente, e devolve o SQL que ele produziu."""
        depth = getattr(_recursion_depth, "value", 0)
        if depth >= self.max_depth:
            return "profundidade maxima de recursao atingida, resolva sem recursar mais"
        return self(sub_question, _depth=depth + 1).sql_query

    def forward(self, question, _depth=0):
        _recursion_depth.value = _depth
        pred = self.agent(dbschema=SCHEMA, question=question)

        # Checagem final, mesmo que o agente diga que ja validou
        sql_query = self.validator.strip_code_fence(pred.sql_query)
        self.validator.validate(sql_query)
        return dspy.Prediction(sql_query=sql_query)


def configure_lm() -> None:
    lm = dspy.LM(LM_MODEL, api_base=LM_API_BASE, api_key=LM_API_KEY)
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())


def build_generator() -> RLMSQLGenerator:
    configure_lm()
    generator = RLMSQLGenerator()
    if os.path.exists(OPTIMIZED_PATH):
        try:
            generator.load(OPTIMIZED_PATH)
        except Exception as error:
            print(f"Aviso: nao foi possivel carregar o programa otimizado ({error}).", file=sys.stderr)
    return generator


def generate_sql(generator: RLMSQLGenerator, question: str) -> str:
    try:
        return generator(question=question).sql_query
    except InvalidSQLError:
        raise
    except Exception as error:
        raise InvalidSQLError(None, error) from error


def run_query_api(sql_query: str) -> list[dict]:
    response = requests.get(API_URL, params={"sql": sql_query}, timeout=10)
    response.raise_for_status()
    return response.json()["results"]


def format_value(key: str, value) -> str:
    if key == "preco" and isinstance(value, (int, float)):
        return f"R$ {value:.2f}"
    return str(value)


def format_results(results: list[dict]) -> str:
    if not results:
        return "Nenhum resultado encontrado."

    lines = []
    for row in results:
        fields = {k: v for k, v in row.items() if k != "id"}
        lines.append(", ".join(format_value(k, v) for k, v in fields.items()))
    return "\n".join(lines)


class EstoqueTelegramBot:
    def __init__(self, generator: RLMSQLGenerator):
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not token:
            raise RuntimeError("Defina a variavel de ambiente TELEGRAM_BOT_TOKEN antes de iniciar o bot.")

        self.generator = generator
        self.bot = telebot.TeleBot(token)
        self.whisper_model = whisper.load_model("base")

        self.bot.message_handler(commands=["start"])(self.handle_start)
        self.bot.message_handler(content_types=["text"])(self.handle_text)
        self.bot.message_handler(content_types=["voice"])(self.handle_voice)

    def handle_start(self, message) -> None:
        self.bot.reply_to(message, "Pergunte algo sobre o estoque, por texto ou audio.")

    def handle_text(self, message) -> None:
        self.answer(message, message.text)

    def handle_voice(self, message) -> None:
        file_info = self.bot.get_file(message.voice.file_id)
        audio_bytes = self.bot.download_file(file_info.file_path)

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            question = self.whisper_model.transcribe(tmp_path, language="portuguese")["text"].strip()
        except Exception as error:
            self.bot.reply_to(message, f"Nao consegui transcrever o audio: {error}")
            return
        finally:
            os.remove(tmp_path)

        self.answer(message, question)

    def answer(self, message, question: str) -> None:
        try:
            sql_query = generate_sql(self.generator, question)
        except InvalidSQLError as error:
            self.bot.reply_to(message, f"Nao consegui gerar uma consulta valida: {error}")
            return

        print(f"[SQL] {question!r} -> {sql_query!r}", file=sys.stderr)

        try:
            results = run_query_api(sql_query)
        except requests.RequestException as error:
            self.bot.reply_to(message, f"Erro ao consultar a API: {error}")
            return

        self.bot.reply_to(message, format_results(results))

    def run(self) -> None:
        self.bot.infinity_polling()


# ---------------------------------------------------------------------------
# Avaliacao e otimizacao (GEPA)
# ---------------------------------------------------------------------------

EVALUATION_DATASET = [
    {"question": "Liste todos os produtos do estoque", "expected_sql": "SELECT * FROM estoque"},
    {"question": "Quais produtos sao do departamento bebidas?", "expected_sql": "SELECT * FROM estoque WHERE departamento = 'bebidas'"},
    {"question": "Quais produtos sao do departamento higiene?", "expected_sql": "SELECT * FROM estoque WHERE departamento = 'higiene'"},
    {"question": "Qual o preco do sabonete?", "expected_sql": "SELECT preco FROM estoque WHERE produto = 'sabonete'"},
    {"question": "Quantas unidades de agua tem em estoque?", "expected_sql": "SELECT quantidade FROM estoque WHERE produto = 'agua'"},
    {"question": "Quais produtos custam mais de 5 reais?", "expected_sql": "SELECT * FROM estoque WHERE preco > 5"},
    {"question": "Quais produtos tem menos de 100 unidades em estoque?", "expected_sql": "SELECT * FROM estoque WHERE quantidade < 100"},
    {"question": "Qual e o produto mais caro do estoque?", "expected_sql": "SELECT * FROM estoque ORDER BY preco DESC LIMIT 1"},
    {"question": "Qual e o produto mais barato do estoque?", "expected_sql": "SELECT * FROM estoque ORDER BY preco ASC LIMIT 1"},
    {"question": "Quantos produtos existem no estoque?", "expected_sql": "SELECT COUNT(*) FROM estoque"},
    {"question": "Qual a quantidade total somando todos os produtos em estoque?", "expected_sql": "SELECT SUM(quantidade) FROM estoque"},
    {"question": "Qual o preco medio dos produtos?", "expected_sql": "SELECT AVG(preco) FROM estoque"},
    {"question": "Quantos produtos existem em cada departamento?", "expected_sql": "SELECT departamento, COUNT(*) FROM estoque GROUP BY departamento"},
    {"question": "Liste os produtos ordenados por quantidade em estoque, do maior para o menor", "expected_sql": "SELECT * FROM estoque ORDER BY quantidade DESC"},
    {"question": "Qual produto tem a data de validade mais proxima de vencer?", "expected_sql": "SELECT * FROM estoque ORDER BY data_validade ASC LIMIT 1"},
    {"question": "Quais produtos vencem antes de 01 de outubro de 2026?", "expected_sql": "SELECT * FROM estoque WHERE data_validade < '2026-10-01'"},
    {"question": "Qual o nome e o departamento de todos os produtos?", "expected_sql": "SELECT produto, departamento FROM estoque"},
    {"question": "Existe algum produto chamado coca no estoque?", "expected_sql": "SELECT * FROM estoque WHERE produto = 'coca'"},
]


def normalize_rows(rows: list[dict]) -> list[tuple]:
    # Ordena para comparar resultados independente da ordem das linhas
    return sorted(tuple(row.values()) for row in rows)


def create_eval_repository():
    from main import EstoqueRepository

    repository = EstoqueRepository(EVAL_DB)
    repository.create_and_seed()
    return repository


def evaluate() -> None:
    generator = build_generator()
    repository = create_eval_repository()

    passed = 0
    for item in EVALUATION_DATASET:
        question, expected_sql = item["question"], item["expected_sql"]
        expected_rows = normalize_rows(repository.run_query(expected_sql))

        generated_sql, error, ok = None, None, False
        try:
            generated_sql = generate_sql(generator, question)
            ok = normalize_rows(repository.run_query(generated_sql)) == expected_rows
        except InvalidSQLError as err:
            generated_sql, error = err.sql_query, str(err)

        passed += ok
        print(f"[{'OK' if ok else 'FALHOU'}] {question}")
        print(f"  esperado : {expected_sql}")
        print(f"  gerado   : {generated_sql}")
        if error:
            print(f"  erro     : {error}")
        print()

    total = len(EVALUATION_DATASET)
    print(f"Acuracia de execucao: {passed}/{total} ({passed / total:.1%})")


def make_gepa_metric(repository):
    # Alem da nota, devolve um texto explicando o erro. Esse feedback e o que o
    # GEPA usa para reescrever as instrucoes do agente.
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        expected_rows = normalize_rows(repository.run_query(gold.expected_sql))
        sql_query = getattr(pred, "sql_query", None)

        if not sql_query:
            return dspy.Prediction(score=0.0, feedback="Nenhum SQL foi gerado para a pergunta.")

        try:
            generated_rows = normalize_rows(repository.run_query(sql_query))
        except Exception as error:
            return dspy.Prediction(
                score=0.0,
                feedback=(
                    f"O SQL gerado ('{sql_query}') falhou ao executar: {error}. "
                    f"Um SQL correto seria: '{gold.expected_sql}'."
                ),
            )

        if generated_rows == expected_rows:
            return dspy.Prediction(score=1.0, feedback="SQL correto: mesmo resultado do gabarito.")

        return dspy.Prediction(
            score=0.0,
            feedback=(
                f"O SQL gerado ('{sql_query}') retornou linhas diferentes do esperado. "
                f"Um SQL correto seria: '{gold.expected_sql}'. "
                f"Esperado: {expected_rows}. Obtido: {generated_rows}."
            ),
        )

    return metric


def optimize() -> None:
    configure_lm()
    repository = create_eval_repository()

    dataset = [
        dspy.Example(question=item["question"], expected_sql=item["expected_sql"]).with_inputs("question")
        for item in EVALUATION_DATASET
    ]
    split = max(1, int(len(dataset) * 0.7))
    trainset, valset = dataset[:split], dataset[split:] or dataset[:split]

    reflection_lm = dspy.LM(REFLECTION_LM_MODEL, api_base=REFLECTION_LM_API_BASE, api_key=REFLECTION_LM_API_KEY)
    optimizer = dspy.GEPA(metric=make_gepa_metric(repository), auto="light", reflection_lm=reflection_lm)

    print(f"Rodando GEPA: {len(trainset)} treino / {len(valset)} validacao...")
    optimized = optimizer.compile(RLMSQLGenerator(), trainset=trainset, valset=valset)
    optimized.save(OPTIMIZED_PATH)
    print(f"Programa otimizado salvo em: {OPTIMIZED_PATH}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "bot"
    if command == "eval":
        evaluate()
    elif command == "optimize":
        optimize()
    else:
        EstoqueTelegramBot(build_generator()).run()
