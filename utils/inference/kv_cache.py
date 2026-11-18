"""KV-cache: ключи и значения уже прочитанных токенов не пересчитываются.

Без кеша каждый новый токен заставляет внимание заново пройти всю
последовательность. С кешем prefill один раз складывает K и V промпта,
а шаг генерации добавляет только один токен. Память за это растёт
линейно с длиной контекста и с размером батча.

Prefix caching идёт дальше: общий префикс разных запросов считается
один раз и переиспользуется. Формулы ниже повторяют разбор урока.
Замер ``benchmark_generation`` — та же схема, что в авторском решении:
``use_cache``, жадная генерация и синхронизация устройства до и после
``generate``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

# Урок замеряет Qwen2.5-3B-Instruct на CUDA. Локальный пример берёт уже
# скачанную Qwen2.5-0.5B-Instruct: отдельной видеокарты NVIDIA здесь нет,
# а 3B для формулы кеша не нужна.
LESSON_KV_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LESSON_PROMPT_SENTENCE = (
    "Расскажи подробно про историю развития языков программирования. "
)
LESSON_PROMPT_REPEATS = 50
LESSON_NEW_TOKENS = 100
LESSON_WARMUP_TOKENS = 5

# Авторский прогон на GPU урока. На другой машине секунды другие,
# длина промпта у токенизатора Qwen2.5 совпадает.
LESSON_REFERENCE_PROMPT_TOKENS = 801
LESSON_REFERENCE_CACHED_SECONDS = 5.20
LESSON_REFERENCE_NOCACHE_SECONDS = 43.74

FLOAT16_BYTES = 2
# В условии задания 1 МБ = 1024 × 1024 байт.
BYTES_PER_MEGABYTE = 1024 * 1024

# Задание 2: модель без GQA, один запрос, контекст 4096, float16.
QUIZ_LAYERS = 32
QUIZ_KV_HEADS = 32
QUIZ_HEAD_DIM = 128
QUIZ_SEQ_LEN = 4096
QUIZ_BATCH_SIZE = 1
QUIZ_CACHE_MEGABYTES = 2048

# Qwen2.5-3B в тексте урока: 16 голов запроса делят 2 группы K/V.
QWEN25_3B_ATTENTION_HEADS = 16
QWEN25_3B_KV_HEADS = 2

# FAQ-бот из разбора: база знаний 2000, вопрос 20, 100 запросов в день.
FAQ_PREFIX_TOKENS = 2000
FAQ_UNIQUE_TOKENS = 20
FAQ_REQUESTS = 100

# Задание 4.
QUIZ_PREFIX_TOKENS = 5000
QUIZ_UNIQUE_TOKENS = 50
QUIZ_REQUESTS = 200
QUIZ_PREFIX_PREFILL_TOKENS = 15_000


@dataclass(frozen=True)
class QuizStatement:
    """Один вариант задания про то, что лежит в KV-cache."""

    text: str
    correct: bool
    reason: str


CACHE_QUIZ: tuple[QuizStatement, ...] = (
    QuizStatement(
        text=(
            "После prefill модель сохраняет K и V для всех 10 000 токенов "
            "промпта. На каждом следующем шаге она добавляет K и V только "
            "для одного нового токена, поэтому память растёт линейно."
        ),
        correct=True,
        reason=(
            "Резкий скачок — это кеш всего промпта. Дальше каждый шаг "
            "дописывает одну пару K и V."
        ),
    ),
    QuizStatement(
        text=(
            "Модель сохраняет только V, а K пересчитывает каждый шаг, "
            "потому что ключи зависят от позиции токена."
        ),
        correct=False,
        reason=(
            "И ключ, и значение токена на своей позиции не меняются, "
            "сколько бы токенов ни сгенерировали после него."
        ),
    ),
    QuizStatement(
        text=(
            "Резкий скачок памяти в начале — это заполнение кеша ключами "
            "и значениями всего промпта на этапе prefill."
        ),
        correct=True,
        reason="Prefill сразу кладёт в кеш K и V всех токенов промпта.",
    ),
    QuizStatement(
        text=(
            "KV-cache экономит вычисления: без него на каждом шаге "
            "пришлось бы заново считать K и V для всей последовательности."
        ),
        correct=True,
        reason="Кеш заменяет повторный полный проход уже прочитанных токенов.",
    ),
    QuizStatement(
        text=(
            "Память растёт линейно, потому что модель постепенно догружает "
            "свои веса в видеопамять по мере генерации."
        ),
        correct=False,
        reason=(
            "Веса загружаются один раз при старте. Линейный рост — это "
            "новые K и V, а не веса."
        ),
    ),
    QuizStatement(
        text=(
            "Кеш можно безопасно очищать после каждого сгенерированного "
            "токена — на скорость это не повлияет."
        ),
        correct=False,
        reason=(
            "Очистка после каждого токена возвращает полный пересчёт "
            "последовательности на каждом шаге."
        ),
    ),
)


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} должен быть целым числом не меньше 1")


def _non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} должен быть целым числом не меньше 0")


def naive_token_work(prompt_tokens: int, new_tokens: int) -> int:
    """Сколько токенов внимание обрабатывает без KV-cache.

    Первый шаг читает промпт и выдаёт первый токен ответа. Каждый
    следующий шаг заново читает всю уже известную последовательность.
    Для ``new_tokens`` шагов это сумма длин
    ``P + (P + 1) + ... + (P + L - 1)``. Сумма квадратична по полной
    длине ``P + L``: урок записывает её как O((P + L)²).

    Args:
        prompt_tokens: Длина промпта P.
        new_tokens: Сколько токенов ответа сгенерировать, L.

    Returns:
        Суммарное число токенов, прошедших через шаги генерации.
    """
    _positive_int("prompt_tokens", prompt_tokens)
    _positive_int("new_tokens", new_tokens)
    return new_tokens * prompt_tokens + new_tokens * (new_tokens - 1) // 2


def cached_token_work(prompt_tokens: int, new_tokens: int) -> int:
    """Сколько токенов внимание обрабатывает с KV-cache.

    Prefill один раз читает промпт. Каждый следующий токен ответа
    добавляет в проход только себя, потому что K и V прошлого уже
    лежат в кеше. Итого ``P + L - 1``: линейно по полной длине.

    Args:
        prompt_tokens: Длина промпта P.
        new_tokens: Сколько токенов ответа сгенерировать, L.

    Returns:
        Суммарное число токенов в prefill и в шагах декодирования.
    """
    _positive_int("prompt_tokens", prompt_tokens)
    _positive_int("new_tokens", new_tokens)
    return prompt_tokens + new_tokens - 1


def naive_attention_cells(prompt_tokens: int, new_tokens: int) -> int:
    """Сколько ячеек внимания считается без кеша.

    На шаге длины s модель строит полную матрицу s × s. Длины шагов —
    P, P + 1, ..., P + L - 1. Меняется на каждом шаге одна новая строка,
    но без кеша матрица собирается заново целиком.
    """
    _positive_int("prompt_tokens", prompt_tokens)
    _positive_int("new_tokens", new_tokens)
    return sum(
        length * length
        for length in range(prompt_tokens, prompt_tokens + new_tokens)
    )


def cached_attention_cells(prompt_tokens: int, new_tokens: int) -> int:
    """Сколько ячеек внимания считается с кешем.

    Prefill по-прежнему строит матрицу P × P: у промпта кеша ещё нет.
    Дальше запрос Q нужен только новому токену, и вместо матрицы
    остаётся одна строка. Её длина — все уже известные позиции,
    включая сам новый токен: P + 1, P + 2, ..., P + L - 1.
    Один токен ответа — это только prefill.
    """
    _positive_int("prompt_tokens", prompt_tokens)
    _positive_int("new_tokens", new_tokens)
    total = prompt_tokens * prompt_tokens
    total += sum(range(prompt_tokens + 1, prompt_tokens + new_tokens))
    return total


def kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = FLOAT16_BYTES,
) -> int:
    """Байты KV-cache.

    Формула урока:
    ``2 × слои × KV-головы × d_head × seq_len × batch × байт на число``.
    Множитель 2 — отдельные тензоры ключей и значений. ``num_kv_heads``
    совпадает с числом голов запроса, если GQA нет. ``seq_len`` растёт
    во время генерации, ``batch_size`` умножает кеш на число запросов
    в пачке: у каждого запроса свой кеш.

    Args:
        num_layers: Число слоёв трансформера.
        num_kv_heads: Число голов, для которых хранятся K и V.
        head_dim: Длина вектора одной головы.
        seq_len: Длина последовательности в токенах.
        batch_size: Сколько запросов держат кеш одновременно.
        dtype_bytes: Байт на одно число. Для float16 это 2.

    Returns:
        Размер кеша в байтах.
    """
    _positive_int("num_layers", num_layers)
    _positive_int("num_kv_heads", num_kv_heads)
    _positive_int("head_dim", head_dim)
    _positive_int("seq_len", seq_len)
    _positive_int("batch_size", batch_size)
    _positive_int("dtype_bytes", dtype_bytes)
    return (
        2
        * num_layers
        * num_kv_heads
        * head_dim
        * seq_len
        * batch_size
        * dtype_bytes
    )


def kv_cache_megabytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = FLOAT16_BYTES,
) -> float:
    """Тот же кеш в МБ. 1 МБ в уроке равен 1024 × 1024 байт.

    Для параметров задания 2 функция возвращает ровно 2048.
    """
    return kv_cache_bytes(
        num_layers,
        num_kv_heads,
        head_dim,
        seq_len,
        batch_size,
        dtype_bytes,
    ) / BYTES_PER_MEGABYTE


def gqa_cache_factor(num_attention_heads: int, num_kv_heads: int) -> int:
    """Во сколько раз GQA уменьшает кеш.

    Несколько голов запроса делят одни и те же K и V, поэтому в формуле
    стоит ``num_kv_heads``, а не полное число голов. У Qwen2.5-3B это
    16 / 2 = 8: кеш в восемь раз меньше, чем был бы без общих групп.
    Число голов запроса должно нацело делиться на число KV-голов.
    """
    _positive_int("num_attention_heads", num_attention_heads)
    _positive_int("num_kv_heads", num_kv_heads)
    if num_attention_heads % num_kv_heads != 0:
        raise ValueError(
            "число голов запроса должно нацело делиться на число KV-голов"
        )
    return num_attention_heads // num_kv_heads


def attention_layout(config: object) -> dict[str, int]:
    """Достаёт из конфига модели числа, от которых зависит размер кеша.

    ``head_dim`` берётся из конфига. Если поля нет, это
    ``hidden_size / num_attention_heads``. Модель без GQA не задаёт
    ``num_key_value_heads``: тогда KV-голов столько же, сколько голов
    запроса.

    Args:
        config: Конфиг ``transformers`` загруженной causal-модели.

    Returns:
        Слои, головы запроса, KV-головы и размер головы.
    """
    query_heads = int(config.num_attention_heads)
    kv_heads = getattr(config, "num_key_value_heads", None)
    if kv_heads is None:
        kv_heads = query_heads
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = int(config.hidden_size) // query_heads
    layout = {
        "num_layers": int(config.num_hidden_layers),
        "num_attention_heads": query_heads,
        "num_kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
    }
    for name, value in layout.items():
        _positive_int(name, value)
    return layout


def prefill_tokens(
    prefix_tokens: int,
    unique_tokens: int,
    num_requests: int,
    *,
    reuse_prefix: bool,
) -> int:
    """Сколько токенов суммарно проходит этап prefill.

    Без prefix caching каждый запрос заново читает общий префикс:
    ``запросы × (префикс + уникальный хвост)``. С кешем префикс
    считается один раз, а на запрос остаётся только хвост:
    ``префикс + запросы × хвост``. Нулевой префикс даёт одно и то же
    число в обоих режимах: общего куска, который можно сохранить, нет.

    Args:
        prefix_tokens: Длина неизменного system-промпта. Может быть 0.
        unique_tokens: Средняя длина уникальной части одного запроса.
        num_requests: Сколько запросов обрабатывается.
        reuse_prefix: Включён ли prefix caching.

    Returns:
        Суммарные токены prefill.
    """
    _non_negative_int("prefix_tokens", prefix_tokens)
    _positive_int("unique_tokens", unique_tokens)
    _positive_int("num_requests", num_requests)
    if reuse_prefix:
        return prefix_tokens + num_requests * unique_tokens
    return num_requests * (prefix_tokens + unique_tokens)


def prefix_savings_factor(
    prefix_tokens: int,
    unique_tokens: int,
    num_requests: int,
) -> float:
    """Во сколько раз prefix caching сокращает токены prefill.

    Это отношение объёма без переиспользования к объёму с ним.
    Единица означает, что экономии нет.
    """
    cached = prefill_tokens(
        prefix_tokens,
        unique_tokens,
        num_requests,
        reuse_prefix=True,
    )
    plain = prefill_tokens(
        prefix_tokens,
        unique_tokens,
        num_requests,
        reuse_prefix=False,
    )
    return plain / cached


def repeat_prompt(
    sentence: str = LESSON_PROMPT_SENTENCE,
    repeats: int = LESSON_PROMPT_REPEATS,
) -> str:
    """Повторяет фразу урока, чтобы префикс стал длинным.

    На коротком тексте разница кеша почти не видна: шаги и так дешёвые.
    """
    _positive_int("repeats", repeats)
    if not sentence:
        raise ValueError("sentence не должна быть пустой")
    return sentence * repeats


def synchronize_device(device: torch.device) -> None:
    """Ждёт, пока устройство закончит уже поставленные вычисления.

    На CUDA и MPS ядра работают асинхронно. Без синхронизации таймер
    снимает постановку задачи в очередь, а не само выполнение. На CPU
    ждать нечего.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_generation(
    model: torch.nn.Module,
    inputs: dict,
    use_cache: bool,
    max_new_tokens: int = LESSON_NEW_TOKENS,
) -> dict[str, float | int | bool]:
    """Измеряет один прогон ``generate`` с кешем или без него.

    ``do_sample=False`` держит жадную траекторию: у Qwen2.5-Instruct
    в generation config включено сэмплирование, и без этого флага два
    режима ушли бы в разные ответы. ``min_new_tokens`` равен
    ``max_new_tokens``, чтобы конец последовательности не оборвал один
    из прогонов раньше и сравнение не смешало скорость с длиной.

    Args:
        model: Causal-модель уже на своём устройстве, в режиме eval.
        inputs: Тензоры токенизатора на том же устройстве.
        use_cache: ``True`` — обычный KV-cache, ``False`` — полный
            пересчёт на каждом шаге.
        max_new_tokens: Сколько новых токенов генерировать.

    Returns:
        Секунды, длина промпта, число новых токенов и флаг кеша.
    """
    _positive_int("max_new_tokens", max_new_tokens)
    device = next(model.parameters()).device
    prompt_tokens = int(inputs["input_ids"].shape[1])
    synchronize_device(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            use_cache=use_cache,
            do_sample=False,
        )
    synchronize_device(device)
    elapsed = time.perf_counter() - started
    return {
        "use_cache": use_cache,
        "seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "new_tokens": int(output.shape[1] - prompt_tokens),
    }


def cache_speedup(nocache_seconds: float, cached_seconds: float) -> float:
    """Во сколько раз прогон без кеша медленнее прогона с кешем.

    Урок делит ``time_nocache / time_cached``. Ноль в знаменателе
    означает, что замер с кешем не снялся.
    """
    if cached_seconds <= 0:
        raise ValueError("время с кешем должно быть больше нуля")
    if nocache_seconds <= 0:
        raise ValueError("время без кеша должно быть больше нуля")
    return nocache_seconds / cached_seconds
