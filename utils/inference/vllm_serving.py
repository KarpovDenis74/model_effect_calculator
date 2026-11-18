"""vLLM: сервер инференса, PagedAttention и нагрузка.

``transformers.pipeline`` обслуживает запросы по очереди. vLLM принимает
их по HTTP, складывает в continuous batch и хранит KV-cache страницами:
запрос получает не сплошной кусок «на максимальную длину», а столько
блоков, сколько уже занято. Освободившиеся страницы возвращаются в пул.

Пакет vLLM собран под NVIDIA. ``build_vllm_command`` и ``vllm_request``
повторяют запуск и клиент урока. На машине без CUDA их некому исполнить:
пропускную способность пачки тогда снимает ``benchmark_batch`` на обычном
pipeline, а ``burst_metrics`` переводит время одной волны в ожидание
пользователя и время обслуживания.
"""

from __future__ import annotations

import gc
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import GenerationConfig, pipeline

from utils.inference.kv_cache import synchronize_device
from utils.models.long_context import inference_device

LESSON_VLLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
VLLM_PROMPT = "Кратко объясни, что такое контейнеризация."
VLLM_MAX_NEW_TOKENS = 80
VLLM_WARMUP_TOKENS = 5
VLLM_TOTAL_REQUESTS = 16
VLLM_CONCURRENCIES = (1, 4, 16)
VLLM_PORT = 8000
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_READY_URL = "http://localhost:8000/v1/models"
VLLM_API_KEY = "EMPTY"
VLLM_MAX_MODEL_LEN = 2048
VLLM_GPU_MEMORY_UTILIZATION = 0.85
VLLM_DTYPE = "half"

# Четыре ответа разной длины. Сплошная бронь — 128 токенов на запрос,
# страница PagedAttention — 16. Числа учебные, не замер vLLM.
PAGE_DEMO_LENGTHS = (20, 50, 10, 100)
PAGE_DEMO_RESERVED = 128
PAGE_DEMO_PAGE_SIZE = 16

# Примерный прогон урока на GPU. На другой карте секунды другие.
LESSON_VLLM_SAMPLE = (
    {"concurrency": 1, "time_sec": 58.3, "throughput": 0.27, "avg_user_latency": 30.60, "avg_service": 3.58},
    {"concurrency": 4, "time_sec": 10.8, "throughput": 1.48, "avg_user_latency": 6.30, "avg_service": 2.65},
    {"concurrency": 16, "time_sec": 3.2, "throughput": 5.07, "avg_user_latency": 3.07, "avg_service": 3.07},
)

FLAG_MAX_MODEL_LEN = "--max-model-len"
FLAG_GPU_MEMORY = "--gpu-memory-utilization"
FLAG_TENSOR_PARALLEL = "--tensor-parallel-size"
FLAG_DTYPE = "--dtype"


@dataclass(frozen=True)
class LessonClaim:
    """Утверждение про vLLM или PagedAttention."""

    text: str
    correct: bool
    reason: str


@dataclass(frozen=True)
class FlagCard:
    """Карточка задания: что делает флаг запуска."""

    text: str
    flag: str
    reason: str


@dataclass(frozen=True)
class AdviceChoice:
    """Совет команде поддержки при пике 20–30 одновременных запросов."""

    text: str
    correct: bool
    reason: str


VLLM_CLAIMS: tuple[LessonClaim, ...] = (
    LessonClaim(
        text="vLLM — это просто обёртка над transformers.pipeline, которая добавляет HTTP-эндпоинт.",
        correct=False,
        reason=(
            "У vLLM свой движок. Continuous batching и PagedAttention "
            "не появляются от того, что pipeline обернули в HTTP."
        ),
    ),
    LessonClaim(
        text=(
            "vLLM использует собственный inference engine с continuous batching "
            "и PagedAttention, а не просто вызывает pipeline() внутри сервера."
        ),
        correct=True,
        reason="Память KV-cache и состав батча управляются движком, а не последовательным вызовом pipeline.",
    ),
    LessonClaim(
        text=(
            "KV-cache хранится не одним непрерывным фрагментом памяти, "
            "а небольшими блоками, аналогичными страницам памяти."
        ),
        correct=True,
        reason="Страницы одного запроса могут лежать не подряд. Новая страница выдаётся, когда ответ подрос.",
    ),
    LessonClaim(
        text=(
            "PagedAttention уменьшает потери памяти, потому что освободившиеся "
            "блоки KV-cache можно вернуть в общий пул и использовать для других запросов."
        ),
        correct=True,
        reason="После конца генерации страницы не остаются забронированными до максимальной длины.",
    ),
    LessonClaim(
        text="Снижение фрагментации помогает обслуживать больше одновременных запросов на том же GPU.",
        correct=True,
        reason="Меньше пустот в кеше — больше запросов влезает в тот же объём, батч становится плотнее.",
    ),
    LessonClaim(
        text="PagedAttention увеличивает количество арифметических операций с плавающей точкой на каждый токен.",
        correct=False,
        reason="Меняется адресация кеша, а не формула внимания. Выигрыш — в памяти, не в числе FLOP.",
    ),
    LessonClaim(
        text="PagedAttention гарантированно ускоряет генерацию одиночного запроса в несколько раз.",
        correct=False,
        reason=(
            "Одному запросу не с кем делить батч. Выигрыш появляется, "
            "когда на ту же карту помещается больше одновременных запросов."
        ),
    ),
)

VLLM_FLAG_CARDS: tuple[FlagCard, ...] = (
    FlagCard(
        text="Максимальная суммарная длина промпта и ответа в токенах",
        flag=FLAG_MAX_MODEL_LEN,
        reason="Под эту длину vLLM и собирает пул KV-cache.",
    ),
    FlagCard(
        text="При нехватке памяти его можно уменьшить, например до 2048 токенов",
        flag=FLAG_MAX_MODEL_LEN,
        reason="Короче окно — меньше страниц в пуле. 2048 хватает учебному тесту.",
    ),
    FlagCard(
        text="Какую долю видеопамяти может использовать экземпляр vLLM для весов, активаций и KV-cache",
        flag=FLAG_GPU_MEMORY,
        reason="vLLM при старте резервирует эту долю почти целиком, поэтому вторую копию модели рядом не держат.",
    ),
    FlagCard(
        text="Его можно немного снизить, например до 0.85, чтобы оставить запас видеопамяти",
        flag=FLAG_GPU_MEMORY,
        reason="0.85 оставляет запас до потолка карты. Если памяти не хватает, флаг снижают дальше, например до 0.7.",
    ),
    FlagCard(
        text="На сколько групп тензорного параллелизма разделить модель",
        flag=FLAG_TENSOR_PARALLEL,
        reason="Размер больше единицы режет веса между несколькими GPU.",
    ),
    FlagCard(
        text="Модель не помещается на одну видеокарту, поэтому её веса распределяют между четырьмя GPU",
        flag=FLAG_TENSOR_PARALLEL,
        reason="Четыре карты в этом сценарии — это --tensor-parallel-size 4.",
    ),
    FlagCard(
        text="В каком числовом формате хранить веса и выполнять вычисления модели",
        flag=FLAG_DTYPE,
        reason="Формат задаётся явно, иначе vLLM выберет bfloat16.",
    ),
    FlagCard(
        text=(
            "Для модели с весами BF16 на видеокарте T4 его задают как half, "
            "поскольку T4 не поддерживает bfloat16"
        ),
        flag=FLAG_DTYPE,
        reason="half здесь — float16. На Turing без этого флага сервер не стартует.",
    ),
)

VLLM_ADVICE: tuple[AdviceChoice, ...] = (
    AdviceChoice(
        text=(
            "Перейти на vLLM: при параллельной нагрузке continuous batching "
            "обеспечивает рост throughput, а при использовании пайплайна "
            "запросы накапливаются в очереди."
        ),
        correct=True,
        reason=(
            "Пик 20–30 одновременных запросов — как раз та нагрузка, "
            "на которой пайплайн копит очередь, а vLLM уплотняет батч."
        ),
    ),
    AdviceChoice(
        text="Остаться на пайплайне: он проще, а разница заметна только в синтетическом тесте.",
        correct=False,
        reason="При десятках одновременных запросов очередь видна пользователям, не только в учебном прогоне.",
    ),
    AdviceChoice(
        text="Перейти на vLLM, потому что он ускоряет каждый отдельный запрос в несколько раз.",
        correct=False,
        reason="На одном запросе за раз vLLM и pipeline близки. Выигрыш — в совместной обработке пачки.",
    ),
    AdviceChoice(
        text="Запустить вторую копию модели через transformers на той же видеокарте.",
        correct=False,
        reason="Вторая копия делит ту же память и не объединяет запросы в один батч.",
    ),
)


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} должен быть целым числом не меньше 1")


def pages_for_length(length: int, page_size: int) -> int:
    """Сколько страниц нужно ответу этой длины.

    Последняя страница может быть заполнена не до конца. Пустота внутри
    неё — единственный остаток, который PagedAttention не отдаёт другому
    запросу, пока запрос ещё жив.
    """
    _positive_int("length", length)
    _positive_int("page_size", page_size)
    return (length + page_size - 1) // page_size


def contiguous_reserved_tokens(num_requests: int, reserved_tokens: int) -> int:
    """Сколько токенов кеша бронирует сплошная схема.

    Длина ответа заранее неизвестна, поэтому на каждый запрос резервируют
    один непрерывный кусок под максимум. Короткий ответ этот кусок не отдаёт.
    """
    _positive_int("num_requests", num_requests)
    _positive_int("reserved_tokens", reserved_tokens)
    return num_requests * reserved_tokens


def paged_reserved_tokens(lengths: tuple[int, ...] | list[int], page_size: int) -> int:
    """Сколько токенов кеша занимают те же ответы по страницам."""
    if not lengths:
        raise ValueError("lengths не должен быть пустым")
    return sum(pages_for_length(length, page_size) * page_size for length in lengths)


def reservation_waste(reserved_tokens: int, useful_tokens: int) -> dict[str, int | float]:
    """Пустота в забронированном кеше.

    ``useful_tokens`` — то, что генерация реально записала. Остальное
    зарезервировано и другим запросам недоступно.
    """
    _positive_int("reserved_tokens", reserved_tokens)
    if isinstance(useful_tokens, bool) or not isinstance(useful_tokens, int):
        raise ValueError("useful_tokens должен быть целым числом")
    if useful_tokens < 0 or useful_tokens > reserved_tokens:
        raise ValueError("useful_tokens должен лежать между 0 и reserved_tokens")
    wasted = reserved_tokens - useful_tokens
    return {
        "reserved": reserved_tokens,
        "useful": useful_tokens,
        "wasted": wasted,
        "waste_share": wasted / reserved_tokens,
    }


def build_vllm_command(
    model_name: str = LESSON_VLLM_MODEL,
    max_model_len: int = VLLM_MAX_MODEL_LEN,
    gpu_memory_utilization: float = VLLM_GPU_MEMORY_UTILIZATION,
    dtype: str = VLLM_DTYPE,
    port: int = VLLM_PORT,
    enforce_eager: bool = True,
    enable_prefix_caching: bool = False,
) -> list[str]:
    """Команда ``vllm serve`` из урока.

    ``--enable-prefix-caching`` по умолчанию выключен. В бенчмарке все
    16 запросов одинаковые, и кеш общего префикса дал бы vLLM фору перед
    pipeline, который префикс каждый раз считает заново. Для FAQ-бота
    с длинным system-промптом флаг как раз включают.

    ``--enforce-eager`` отключает компиляцию CUDA-графов: сервер стартует
    быстрее и ест меньше памяти, инференс чуть медленнее.

    Args:
        model_name: Модель, которую поднимет сервер.
        max_model_len: Промпт плюс ответ, в токенах.
        gpu_memory_utilization: Доля VRAM под веса, активации и KV-cache.
        dtype: ``half`` — float16. На T4 bfloat16 нет.
        port: Порт OpenAI-совместимого HTTP.
        enforce_eager: Не компилировать CUDA-графы.
        enable_prefix_caching: Переиспользовать KV-cache общего префикса.

    Returns:
        Аргументы для ``subprocess``, начиная с исполняемого файла ``vllm``.
    """
    _positive_int("max_model_len", max_model_len)
    _positive_int("port", port)
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization должен быть в интервале (0, 1]")
    if not dtype:
        raise ValueError("dtype не должен быть пустым")
    command = [
        "vllm",
        "serve",
        model_name,
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--dtype",
        dtype,
        "--port",
        str(port),
    ]
    if enforce_eager:
        command.append("--enforce-eager")
    if enable_prefix_caching:
        command.append("--enable-prefix-caching")
    return command


def vllm_runtime_available() -> bool:
    """Можно ли на этой машине исполнить ``vllm serve``.

    Нужны и исполняемый файл, и CUDA. Пакет не ставится в этот проект
    автоматически: на Apple Silicon сервер урока не запускается.
    """
    return shutil.which("vllm") is not None and torch.cuda.is_available()


def server_is_ready(url: str = VLLM_READY_URL, timeout: float = 2) -> bool:
    """Ответил ли ``/v1/models`` кодом 200.

    Локальный сервер так сообщает, что веса загружены и можно слать
    запросы. Ошибка сети и любой другой код — ещё не готов.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_until_ready(
    url: str = VLLM_READY_URL,
    timeout: int = 600,
    poll_interval: float = 3,
    process=None,
) -> bool:
    """Ждёт готовности сервера.

    Загрузка весов занимает минуты, поэтому одна проверка не подходит.
    Если передан процесс и он уже завершился, ожидание прекращается сразу:
    дальше сервер сам не поднимется.

    Args:
        url: Адрес ``/v1/models``.
        timeout: Сколько секунд ждать в сумме.
        poll_interval: Пауза между опросами.
        process: ``Popen`` сервера. ``None``, если процесс запущен снаружи.

    Returns:
        ``True``, когда сервер ответил 200.
    """
    _positive_int("timeout", timeout)
    if poll_interval <= 0:
        raise ValueError("poll_interval должен быть больше нуля")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            return False
        if server_is_ready(url):
            return True
        time.sleep(poll_interval)
    return False


def make_vllm_client(base_url: str = VLLM_BASE_URL, api_key: str = VLLM_API_KEY, *, async_client: bool = False):
    """Клиент OpenAI, который смотрит на локальный vLLM.

    Ключ локальный сервер не проверяет, но библиотека требует непустое
    поле. ``async_client=True`` нужен нагрузочному тесту: синхронный
    клиент не отпускает event loop, и concurrency не с чем сравнивать.
    """
    if async_client:
        from openai import AsyncOpenAI

        return AsyncOpenAI(base_url=base_url, api_key=api_key)
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)


async def vllm_request(semaphore, client, prompt: str, model_name: str, max_tokens: int = VLLM_MAX_NEW_TOKENS):
    """Один запрос к vLLM под семафором.

    Время пользователя считается от создания задачи: запрос уже «отправлен»,
    даже если семафор ещё не пустил его на сервер. Время обслуживания —
    только участок внутри семафора, пока HTTP-вызов реально шёл.
    ``temperature=0`` держит жадную генерацию, как ``do_sample=False``
    у pipeline. Иначе разница замеров смешается с режимом декодирования.

    Returns:
        Ожидание пользователя, время обслуживания и текст ответа.
    """
    _positive_int("max_tokens", max_tokens)
    submitted = time.perf_counter()
    async with semaphore:
        start = time.perf_counter()
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        service = time.perf_counter() - start
        user_latency = time.perf_counter() - submitted
        return user_latency, service, response.choices[0].message.content


async def load_test_vllm(
    concurrency: int,
    client,
    prompt: str = VLLM_PROMPT,
    model_name: str = LESSON_VLLM_MODEL,
    total: int = VLLM_TOTAL_REQUESTS,
    max_tokens: int = VLLM_MAX_NEW_TOKENS,
) -> dict[str, float | int]:
    """Пачка из ``total`` запросов, одновременно летит не больше ``concurrency``.

    Все задачи создаются сразу. При ``concurrency=1`` пятнадцать из шестнадцати
    стоят в очереди, и среднее ожидание пользователя гораздо больше времени
    обслуживания. При ``concurrency=total`` очереди нет, и метрики сходятся.
    """
    import asyncio

    _positive_int("concurrency", concurrency)
    _positive_int("total", total)
    semaphore = asyncio.Semaphore(concurrency)
    started = time.perf_counter()
    tasks = [
        vllm_request(semaphore, client, prompt, model_name, max_tokens)
        for _ in range(total)
    ]
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started
    user_latencies = [item[0] for item in results]
    service_times = [item[1] for item in results]
    return {
        "concurrency": concurrency,
        "total": total,
        "time_sec": elapsed,
        "throughput": total / elapsed,
        "avg_user_latency": sum(user_latencies) / len(user_latencies),
        "avg_service": sum(service_times) / len(service_times),
    }


def burst_metrics(total: int, concurrency: int, wave_seconds: float) -> dict[str, float | int]:
    """Ожидание и обслуживание, если пачку подали разом, а сервер идёт волнами.

    Волна — это до ``concurrency`` запросов, которые заканчиваются вместе
    и занимают ``wave_seconds``. Все запросы считаются отправленными в
    нулевой момент, как ``asyncio.gather`` в нагрузочном тесте. Обслуживание
    одного запроса равно длине его волны. Ожидание пользователя — момент,
    когда волна закончилась: у поздних волн в него входит очередь.

    Для равных промптов одна волна — это статический батч того же размера.
    Continuous batching на одинаковой длине ответов к тому же и приходит.
    """
    _positive_int("total", total)
    _positive_int("concurrency", concurrency)
    if wave_seconds <= 0:
        raise ValueError("wave_seconds должен быть больше нуля")
    n_waves = (total + concurrency - 1) // concurrency
    user_latency = sum((index // concurrency + 1) * wave_seconds for index in range(total))
    elapsed = n_waves * wave_seconds
    return {
        "concurrency": concurrency,
        "total": total,
        "time_sec": elapsed,
        "throughput": total / elapsed,
        "avg_user_latency": user_latency / total,
        "avg_service": wave_seconds,
    }


def chat_prompt(tokenizer, prompt: str = VLLM_PROMPT) -> str:
    """Собирает тот же вход, который vLLM получит на ``/v1/chat/completions``."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _generation_config(tokenizer, max_new_tokens: int) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )


def load_text_pipeline(model_name: str):
    """Пайплайн генерации на CUDA, MPS или CPU.

    Урок ставит ``device_map="cuda"`` и ``torch_dtype=float16``. Здесь
    устройство выбирается само. Pad-токен и левый паддинг нужны, чтобы
    пачка одинаковых промптов собралась в один батч.
    """
    device = inference_device()
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    try:
        text_pipe = pipeline("text-generation", model=model_name, dtype=dtype, device=device)
    except TypeError:
        text_pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=dtype,
            device=device,
        )
    tokenizer = text_pipe.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if getattr(text_pipe.model, "generation_config", None) is not None:
        text_pipe.model.generation_config.pad_token_id = tokenizer.pad_token_id
    text_pipe.model.eval()
    return text_pipe


def release_pipeline(text_pipe) -> None:
    """Убирает веса пайплайна из памяти ускорителя.

    Вызывающий код после этого должен отпустить и свою ссылку на пайплайн.
    Иначе объект останется живым, и ``empty_cache`` ничего не освободит.
    """
    model = getattr(text_pipe, "model", None)
    text_pipe.model = None
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def benchmark_pipeline(
    text_pipe,
    chat_text: str,
    total: int = VLLM_TOTAL_REQUESTS,
    max_new_tokens: int = VLLM_MAX_NEW_TOKENS,
) -> dict[str, float | int]:
    """Сколько pipeline тратит на ``total`` запросов строго по одному.

    Это потолок последовательного обслуживания: параллельные клиенты
    всё равно встанут в очередь к одному вызову. Перед каждым вызовом
    и после него устройство синхронизируется, иначе MPS и CUDA запишут
    в таймер постановку ядра, а не саму генерацию.
    """
    _positive_int("total", total)
    _positive_int("max_new_tokens", max_new_tokens)
    config = _generation_config(text_pipe.tokenizer, max_new_tokens)
    device = text_pipe.model.device
    latencies: list[float] = []
    started = time.perf_counter()
    for index in range(total):
        print(f"pipeline: {index + 1}/{total}", flush=True)
        synchronize_device(device)
        tick = time.perf_counter()
        text_pipe(chat_text, generation_config=config)
        synchronize_device(device)
        latencies.append(time.perf_counter() - tick)
    elapsed = time.perf_counter() - started
    return {
        "total": total,
        "time_sec": elapsed,
        "throughput": total / elapsed,
        "avg_service": sum(latencies) / len(latencies),
    }


def benchmark_batch(
    text_pipe,
    chat_text: str,
    batch_size: int,
    max_new_tokens: int = VLLM_MAX_NEW_TOKENS,
) -> float:
    """Секунды одного статического батча из одинаковых промптов.

    Длины совпадают, поэтому паддинга нет. Это не continuous batching
    vLLM: состав батча не меняется по ходу генерации. Для равных ответов
    разница с «посчитать пачку разом» небольшая, и по этому времени видно,
    уплотняется ли карта, когда запросов несколько.
    """
    _positive_int("batch_size", batch_size)
    _positive_int("max_new_tokens", max_new_tokens)
    config = _generation_config(text_pipe.tokenizer, max_new_tokens)
    device = text_pipe.model.device
    texts = [chat_text] * batch_size
    synchronize_device(device)
    started = time.perf_counter()
    text_pipe(texts, generation_config=config, batch_size=batch_size)
    synchronize_device(device)
    return time.perf_counter() - started


def plot_throughput(
    sequential_throughput: float,
    points: list[tuple[int, float]] | tuple[tuple[int, float], ...],
    path: str | Path,
) -> Path:
    """Рисует потолок pipeline и throughput волн разного размера.

    Горизонтальная линия — запросы по одному. Точки — пачка из 16,
    которую сервер забирает волнами размера concurrency.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if sequential_throughput <= 0:
        raise ValueError("sequential_throughput должен быть больше нуля")
    if not points:
        raise ValueError("points не должен быть пустым")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    concurrencies = [item[0] for item in points]
    throughputs = [item[1] for item in points]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(concurrencies, throughputs, marker="o", color="#2e6db4", label="волны батча")
    ax.axhline(
        sequential_throughput,
        color="#c0392b",
        linestyle="--",
        label="pipeline по одному",
    )
    ax.set_xlabel("Одновременных запросов")
    ax.set_ylabel("Запросов в секунду")
    ax.set_title("Throughput при росте пачки")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output
