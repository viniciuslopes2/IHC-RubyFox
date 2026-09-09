import os
import re
import sqlite3
import sys
import tempfile
import threading
from dataclasses import dataclass

import dspy
import requests
import telebot
import whisper

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


@dataclass(frozen=True)
class BotConfig:
    """Configuracao do bot: caminhos, endpoint da API e parametros do LM."""

    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    optimized_program_filename: str = "optimized_generator.json"
    eval_db_filename: str = "estoque_eval.db"
    api_url: str = "http://127.0.0.1:8000/query"
    lm_model: str = "openai/gemma-3-4b-instruct"
    lm_api_base: str = "http://localhost:1234/v1"
    lm_api_key: str = "not-needed"
    whisper_model_name: str = "base"
    max_generation_attempts: int = 3

    @property
    def optimized_program_path(self) -> str:
        return os.path.join(self.base_dir, self.optimized_program_filename)

    @property
    def eval_db_path(self) -> str:
        return os.path.join(self.base_dir, self.eval_db_filename)

    @property
    def reflection_lm_model(self) -> str:
        """Modelo usado pelo GEPA para propor melhorias de instrucao.
        Pode ser diferente do modelo de geracao (ex: um modelo maior/melhor
        na propria LM Studio, ou uma API externa), via env vars."""
        return os.environ.get("GEPA_REFLECTION_LM", self.lm_model)

    @property
    def reflection_lm_api_base(self) -> str:
        return os.environ.get("GEPA_REFLECTION_LM_API_BASE", self.lm_api_base)

    @property
    def reflection_lm_api_key(self) -> str:
        return os.environ.get("GEPA_REFLECTION_LM_API_KEY", self.lm_api_key)

    @property
    def telegram_token(self) -> str:
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not token:
            raise RuntimeError(
                "Defina a variavel de ambiente TELEGRAM_BOT_TOKEN antes de iniciar o bot."
            )
        return token


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


class SQLValidator:
    """Encapsula as checagens de seguranca/sanidade de um SQL antes de execucao:
    apenas SELECT e permitido, e o SQL precisa rodar sem erro num sqlite in-memory
    com o schema real (pega erros de sintaxe/coluna antes de ir para a API)."""

    def __init__(self, schema_ddl: str = SCHEMA_DDL, forbidden_keywords=FORBIDDEN_KEYWORDS):
        self._schema_ddl = schema_ddl
        self._forbidden_keywords = forbidden_keywords

    @staticmethod
    def strip_code_fence(sql_query: str) -> str:
        match = CODE_FENCE_RE.match(sql_query.strip())
        if match:
            return match.group(1).strip()
        return sql_query.strip()

    def check_select_only(self, sql_query: str) -> None:
        statements = [s.strip() for s in sql_query.split(";") if s.strip()]

        if len(statements) != 1:
            raise SelectOnlyError(sql_query, "multiplas instrucoes")

        statement = statements[0]

        if not statement.upper().startswith("SELECT"):
            raise SelectOnlyError(sql_query, statement.split()[0].upper())

        for keyword in self._forbidden_keywords:
            if re.search(rf"\b{keyword}\b", statement, re.IGNORECASE):
                raise SelectOnlyError(sql_query, keyword)

    def validate(self, sql_query: str) -> None:
        self.check_select_only(sql_query)

        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(self._schema_ddl)
            conn.execute(sql_query)
        except sqlite3.Error as error:
            raise SQLValidationError(sql_query, error) from error
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


class RecursiveSQLGenerator(dspy.Module):
    """Tarefa 7: substitui o dspy.ChainOfThought por uma estrategia recursiva
    (RLM): gera um SQL, valida (SQLValidator), e se invalido chama a si mesma de
    novo passando o SQL e o erro anteriores para o LM se autocorrigir -- ate
    `max_attempts` tentativas. So propaga a excecao se nem assim conseguir um
    SQL valido."""

    def __init__(
        self,
        validator: SQLValidator,
        schema: str = SCHEMA_DESCRIPTION,
        max_attempts: int = 3,
    ):
        super().__init__()
        self.schema = schema
        self.validator = validator
        self.max_attempts = max_attempts
        self.generate_sql = dspy.Predict(TextToSQL)

    def forward(self, question, _attempt=1, _previous_sql="", _previous_error=""):
        pred = self.generate_sql(
            dbschema=self.schema,
            question=question,
            previous_sql=_previous_sql,
            previous_error=_previous_error,
        )
        sql_query = self.validator.strip_code_fence(pred.sql_query)

        try:
            self.validator.validate(sql_query)
        except (SQLValidationError, SelectOnlyError) as error:
            if _attempt >= self.max_attempts:
                raise
            return self(
                question,
                _attempt=_attempt + 1,
                _previous_sql=sql_query,
                _previous_error=str(error),
            )

        return dspy.Prediction(sql_query=sql_query, attempts=_attempt)


class SQLGeneratorService:
    """Envolve o RecursiveSQLGenerator com tratamento de excecao num unico ponto,
    para que qualquer chamador (bot, script de avaliacao, etc.) receba sempre um
    SQL valido ou uma excecao tipada."""

    def __init__(
        self,
        generator: RecursiveSQLGenerator,
        optimized_program_path: str | None = None,
    ):
        self._generator = generator
        if optimized_program_path and os.path.exists(optimized_program_path):
            try:
                self._generator.load(optimized_program_path)
            except Exception as error:
                print(
                    f"Aviso: nao foi possivel carregar o programa otimizado em "
                    f"'{optimized_program_path}' ({error}). Usando o modulo sem otimizacao.",
                    file=sys.stderr,
                )

    def generate(self, question: str) -> str:
        try:
            prediction = self._generator(question=question)
        except (SQLValidationError, SelectOnlyError):
            raise
        except Exception as error:
            raise SQLValidationError(None, error) from error

        return prediction.sql_query


class EstoqueAPIClient:
    """Cliente HTTP para a API de estoque encapsulada em main.py."""

    def __init__(self, api_url: str):
        self._api_url = api_url

    def run_query(self, sql_query: str) -> list[dict]:
        response = requests.get(self._api_url, params={"sql": sql_query}, timeout=10)
        response.raise_for_status()
        return response.json()["results"]


def _format_value(key: str, value) -> str:
    if key == "preco" and isinstance(value, (int, float)):
        return f"R$ {value:.2f}"
    return str(value)


def format_results(results: list[dict]) -> str:
    if not results:
        return "Nenhum resultado encontrado."

    lines = []
    for row in results:
        fields = {k: v for k, v in row.items() if k != "id"}
        if len(fields) == 1:
            (key, value), = fields.items()
            lines.append(_format_value(key, value))
        else:
            lines.append(", ".join(_format_value(k, v) for k, v in fields.items()))

    return "\n".join(lines)


class EstoqueTelegramBot:
    """Encapsula o bot do Telegram: handlers, transcricao de audio (whisper) e
    orquestracao entre SQLGeneratorService e EstoqueAPIClient."""

    def __init__(
        self,
        config: BotConfig,
        sql_service: SQLGeneratorService,
        api_client: EstoqueAPIClient,
    ):
        self._config = config
        self._sql_service = sql_service
        self._api_client = api_client
        self._bot = telebot.TeleBot(config.telegram_token)
        self._whisper_model = whisper.load_model(config.whisper_model_name)
        self._register_handlers()

    def _register_handlers(self) -> None:
        self._bot.message_handler(commands=["start"])(self._handle_start)
        self._bot.message_handler(content_types=["text"])(self._handle_text)
        self._bot.message_handler(content_types=["voice"])(self._handle_voice)

    def _handle_start(self, message) -> None:
        self._bot.reply_to(message, "Pergunte algo sobre o estoque, por texto ou audio.")

    def _handle_text(self, message) -> None:
        self._answer_question(message, message.text)

    def _handle_voice(self, message) -> None:
        file_info = self._bot.get_file(message.voice.file_id)
        audio_bytes = self._bot.download_file(file_info.file_path)

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            transcription = self._whisper_model.transcribe(tmp_path, language="portuguese")
            question = transcription["text"].strip()
        except Exception as error:
            # Nunca deixa uma falha de transcricao (ex: ffmpeg ausente) derrubar
            # o processo inteiro do bot -- so essa mensagem falha.
            self._bot.reply_to(message, f"Nao consegui transcrever o audio: {error}")
            return
        finally:
            os.remove(tmp_path)

        self._answer_question(message, question)

    def _answer_question(self, message, question: str) -> None:
        try:
            sql_query = self._sql_service.generate(question)
        except (SQLValidationError, SelectOnlyError) as error:
            self._bot.reply_to(message, f"Nao consegui gerar uma consulta valida: {error}")
            return

        print(f"[SQL] {question!r} -> {sql_query!r}", file=sys.stderr)

        try:
            results = self._api_client.run_query(sql_query)
        except requests.RequestException as error:
            self._bot.reply_to(message, f"Erro ao consultar a API: {error}")
            return

        self._bot.reply_to(message, format_results(results))

    def run(self) -> None:
        self._bot.infinity_polling()


def configure_lm(config: BotConfig) -> None:
    lm = dspy.LM(config.lm_model, api_base=config.lm_api_base, api_key=config.lm_api_key)
    # ChatAdapter faz parsing por marcadores de texto, sem depender de
    # response_format/json_schema -- o LM Studio nao aceita o modo JSON que o
    # DSPy tenta usar por padrao para modelos "openai/*".
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())


def build_sql_service(config: BotConfig) -> SQLGeneratorService:
    """Monta so a camada de geracao/validacao de SQL, sem telegram nem whisper.
    Usado pelo bot e tambem pelo script de avaliacao (tarefa 6)."""
    configure_lm(config)
    validator = SQLValidator()
    generator = RecursiveSQLGenerator(validator, max_attempts=config.max_generation_attempts)
    return SQLGeneratorService(generator, config.optimized_program_path)


def build_bot(config: BotConfig | None = None) -> EstoqueTelegramBot:
    config = config or BotConfig()
    sql_service = build_sql_service(config)
    api_client = EstoqueAPIClient(config.api_url)
    return EstoqueTelegramBot(config, sql_service, api_client)


# ---------------------------------------------------------------------------
# Tarefa 6: avaliacao (execution accuracy) do gerador de SQL.
# Roda com: python bot.py eval
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


@dataclass
class EvaluationResult:
    question: str
    expected_sql: str
    generated_sql: str | None
    error: str | None
    passed: bool


def _normalize_rows(rows: list[dict]) -> list[tuple]:
    return sorted(tuple(row.values()) for row in rows)


def evaluate(
    config: BotConfig | None = None,
    dataset: list[dict] = EVALUATION_DATASET,
) -> list[EvaluationResult]:
    """Roda cada pergunta do dataset pelo SQLGeneratorService real (LM + validacao)
    e compara o resultado da execucao do SQL gerado com o do SQL esperado (gabarito),
    ambos executados contra o mesmo banco (execution accuracy)."""
    from main import EstoqueConfig, EstoqueRepository

    config = config or BotConfig()
    sql_service = build_sql_service(config)

    repository = EstoqueRepository(EstoqueConfig(base_dir=config.base_dir, db_filename=config.eval_db_filename))
    repository.create_and_seed()

    results = []
    for item in dataset:
        question, expected_sql = item["question"], item["expected_sql"]
        expected_rows = _normalize_rows(repository.run_query(expected_sql))

        generated_sql = None
        error = None
        passed = False
        try:
            generated_sql = sql_service.generate(question)
            generated_rows = _normalize_rows(repository.run_query(generated_sql))
            passed = generated_rows == expected_rows
        except (SQLValidationError, SelectOnlyError) as err:
            generated_sql = err.sql_query
            error = str(err)

        results.append(
            EvaluationResult(question, expected_sql, generated_sql, error, passed)
        )

    return results


def print_evaluation_report(results: list[EvaluationResult]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)

    for r in results:
        status = "OK" if r.passed else "FALHOU"
        print(f"[{status}] {r.question}")
        print(f"  esperado : {r.expected_sql.strip()}")
        print(f"  gerado   : {r.generated_sql}")
        if r.error:
            print(f"  erro     : {r.error}")
        print()

    accuracy = passed / total if total else 0.0
    print(f"Acuracia de execucao: {passed}/{total} ({accuracy:.1%})")


# ---------------------------------------------------------------------------
# Tarefa 4/7: otimizacao com GEPA do RecursiveSQLGenerator.
# Roda com: python bot.py optimize
# ---------------------------------------------------------------------------

def _build_gepa_devset(dataset: list[dict] = EVALUATION_DATASET) -> list["dspy.Example"]:
    return [
        dspy.Example(question=item["question"], expected_sql=item["expected_sql"]).with_inputs("question")
        for item in dataset
    ]


_gepa_repository_lock = threading.Lock()
_gepa_repository = None


def _get_gepa_repository():
    global _gepa_repository
    if _gepa_repository is None:
        with _gepa_repository_lock:
            if _gepa_repository is None:
                from main import EstoqueConfig, EstoqueRepository

                config = BotConfig()
                repository = EstoqueRepository(
                    EstoqueConfig(base_dir=config.base_dir, db_filename=config.eval_db_filename)
                )
                repository.create_and_seed()
                _gepa_repository = repository
    return _gepa_repository


def _gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None, program_trace=None):
    """Metrica de execution accuracy com feedback textual para o GEPA usar na
    reflexao: roda o SQL esperado e o gerado contra o mesmo banco e compara
    as linhas resultantes."""
    repository = _get_gepa_repository()

    expected_rows = _normalize_rows(repository.run_query(gold.expected_sql))
    sql_query = getattr(pred, "sql_query", None)

    if not sql_query:
        return dspy.Prediction(score=0.0, feedback="Nenhum SQL foi gerado para a pergunta.")

    try:
        generated_rows = _normalize_rows(repository.run_query(sql_query))
    except Exception as error:
        return dspy.Prediction(
            score=0.0,
            feedback=(
                f"O SQL gerado ('{sql_query}') falhou ao executar: {error}. "
                f"Um SQL correto para esta pergunta seria: '{gold.expected_sql}'."
            ),
        )

    if generated_rows == expected_rows:
        return dspy.Prediction(score=1.0, feedback="SQL correto: mesmo resultado do gabarito.")

    return dspy.Prediction(
        score=0.0,
        feedback=(
            f"O SQL gerado ('{sql_query}') executou mas retornou linhas diferentes "
            f"do esperado. Um SQL correto seria: '{gold.expected_sql}'. "
            f"Linhas esperadas: {expected_rows}. Linhas obtidas: {generated_rows}."
        ),
    )


def run_gepa_optimization(config: BotConfig | None = None) -> None:
    config = config or BotConfig()
    configure_lm(config)

    dataset = _build_gepa_devset()
    split = max(1, int(len(dataset) * 0.7))
    trainset, valset = dataset[:split], dataset[split:] or dataset[:split]

    validator = SQLValidator()
    student = RecursiveSQLGenerator(validator, max_attempts=config.max_generation_attempts)

    reflection_lm = dspy.LM(
        config.reflection_lm_model,
        api_base=config.reflection_lm_api_base,
        api_key=config.reflection_lm_api_key,
    )

    optimizer = dspy.GEPA(metric=_gepa_metric, auto="light", reflection_lm=reflection_lm)

    print(f"Rodando GEPA: {len(trainset)} treino / {len(valset)} validacao...")
    optimized = optimizer.compile(student, trainset=trainset, valset=valset)
    optimized.save(config.optimized_program_path)
    print(f"Programa otimizado salvo em: {config.optimized_program_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        print_evaluation_report(evaluate())
    elif len(sys.argv) > 1 and sys.argv[1] == "optimize":
        run_gepa_optimization()
    else:
        build_bot().run()
