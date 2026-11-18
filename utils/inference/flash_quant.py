"""Flash-attention и квантизация: память GPU и точность весов.

Flash-attention считает то же внимание, но не записывает полную матрицу
seq × seq в медленную HBM. Квантизация хранит веса в INT8 или INT4.
NF4 — сетка из 16 уровней bitsandbytes: гуще около нуля, где лежит
большая часть весов. GPTQ и AWQ подбирают эти уровни по калибровочным
текстам, а не округляют веса «как есть».

На CUDA ``load_model(use_int4=True)`` повторяет авторскую загрузку через
bitsandbytes. Без NVIDIA-ядер локальный прогон округляет Linear-веса
в NF4 и записывает восстановленные float-значения обратно: так видна
ошибка квантизации. Память процесса при этом не сжимается, её оценивает
``weight_storage_report``.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from utils.inference.kv_cache import synchronize_device

LESSON_QUANT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
FLASH_ATTENTION_IMPLEMENTATION = "flash_attention_2"

# Урок берёт круглую оценку «3 миллиарда параметров». Замер на GPU
# (5.75 ГБ) чуть выше: у Qwen2.5-3B параметров немного больше, и
# memory_allocated считает всю загрузку, не только сырые веса.
LESSON_3B_PARAMETERS = 3_000_000_000
LESSON_70B_PARAMETERS = 70_000_000_000
FLOAT16_BITS = 16
INT8_BITS = 8
INT4_BITS = 4
# Округление из текста урока, не результат деления на 1024³.
LESSON_ROUNDED_GB = {16: 5.8, 8: 2.9, 4: 1.4}

LESSON_GPU_FP16_MEMORY_GB = 5.75
LESSON_GPU_INT4_MEMORY_GB = 1.93
LESSON_GPU_FP16_SECONDS = 13.10
LESSON_GPU_INT4_SECONDS = 18.20
LESSON_GPU_FP16_TOKENS = 334
LESSON_GPU_INT4_TOKENS = 356
LESSON_GPU_FP16_TPS = 25.5
LESSON_GPU_INT4_TPS = 19.6

# «Десятки мегабайт» SRAM из урока. 20 МиБ — конкретный стол, с которым
# сравнивается матрица внимания. Это не паспорт конкретной видеокарты.
DEFAULT_SRAM_BYTES = 20 * 1024 * 1024
ATTENTION_DEMO_LENGTHS = (128, 1024, 4096, 8192)
GIB = 1024**3

# 16 уровней NF4 из bitsandbytes / QLoRA, уже нормированы на [-1, 1].
# Нуль входит в сетку, положительных уровней на один больше, чем отрицательных.
NF4_CODEBOOK: tuple[float, ...] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)
NF4_BLOCK_SIZE = 64
NF4_SCALE_GROUP = 256

EXAMPLE_WEIGHT = 0.0231
EXAMPLE_WEIGHT_LOW = -0.4
EXAMPLE_WEIGHT_HIGH = 0.4

COMPARISON_QUESTIONS: tuple[str, ...] = (
    "Какая столица у Франции? Ответь одним словом.",
    "Переведи на английский одним словом: «собака».",
    "Реши по шагам: ноутбук стоил 80 000 рублей. Цену подняли на 15%, затем на "
    "распродаже снизили на 25%, а при оплате картой дали ещё скидку 10%. "
    "Сколько заплатит покупатель картой? В конце назови итоговую сумму в рублях.",
    "Напиши одно предложение о пользе сна.",
)
LAPTOP_BASE_RUB = 80_000
LAPTOP_PRICE_RUB = 62_100
COMPARISON_MAX_NEW_TOKENS = 256

METHOD_FLASH = "Flash-attention"
METHOD_INT8 = "INT8"
METHOD_INT4 = "INT4"
METHOD_GPTQ_AWQ = "GPTQ или AWQ"


@dataclass(frozen=True)
class LessonChoice:
    """Вариант вопроса про качество после INT4."""

    text: str
    correct: bool
    reason: str


@dataclass(frozen=True)
class OptimizationCase:
    """Сценарий и метод, который закрывает его узкое место."""

    scenario: str
    method: str
    reason: str


QUALITY_QUIZ: tuple[LessonChoice, ...] = (
    LessonChoice(
        text="Качество не изменится: квантизация не влияет на способность рассуждать.",
        correct=False,
        reason=(
            "INT4 грубее представляет веса. На многошаговой математике "
            "погрешность накапливается от шага к шагу."
        ),
    ),
    LessonChoice(
        text=(
            "Качество может снизиться, иногда заметно: математические "
            "рассуждения чувствительны к INT4, поэтому результат нужно "
            "проверить бенчмарком."
        ),
        correct=True,
        reason=(
            "Насколько просядет ответ, зависит от модели, метода и задач. "
            "Это измеряют, а не предполагают."
        ),
    ),
    LessonChoice(
        text="Качество вырастет: сжатие действует как регуляризация и улучшает ответы.",
        correct=False,
        reason=(
            "Квантизация сокращает память. Она может сохранить качество "
            "близким к float16, но сама по себе его не повышает."
        ),
    ),
    LessonChoice(
        text="Модель полностью перестанет отвечать: INT4 ломает генерацию.",
        correct=False,
        reason=(
            "Корректно квантованная INT4-модель продолжает генерировать текст. "
            "Риск — точность на чувствительных задачах, а не полный отказ."
        ),
    ),
)

OPTIMIZATION_CASES: tuple[OptimizationCase, ...] = (
    OptimizationCase(
        scenario=(
            "Модель обрабатывает запросы с очень длинным контекстом, "
            "а узкое место — скорость вычисления attention"
        ),
        method=METHOD_FLASH,
        reason="Длинная матрица внимания упирается в поездки в HBM. Flash-attention убирает эти поездки.",
    ),
    OptimizationCase(
        scenario=(
            "Контекст вырос до десятков тысяч токенов, и attention стал "
            "упираться в память и время"
        ),
        method=METHOD_FLASH,
        reason="С ростом seq_len матрица seq × seq перестаёт помещаться в SRAM. Выигрыш flash-attention здесь как раз и появляется.",
    ),
    OptimizationCase(
        scenario=(
            "Нужно примерно вдвое сократить память под веса, сохранив "
            "качество максимально близким к FP16"
        ),
        method=METHOD_INT8,
        reason="INT8 — один байт вместо двух. Ступеней 256, для большинства задач сдвиг почти незаметен.",
    ),
    OptimizationCase(
        scenario=(
            "Памяти не хватает совсем немного, и рисковать качеством "
            "сложных ответов не хочется"
        ),
        method=METHOD_INT8,
        reason="Небольшой дефицит закрывается двукратным сжатием. INT4 здесь лишний риск для рассуждений.",
    ),
    OptimizationCase(
        scenario=(
            "Модель не помещается в видеопамять, поэтому нужно максимально "
            "сократить память под веса, допустив снижение качества"
        ),
        method=METHOD_INT4,
        reason="INT4 режет вес вчетверо относительно float16. Это способ влезть в карту, когда двукратного сжатия мало.",
    ),
    OptimizationCase(
        scenario=(
            "Для очень крупной модели нужно сократить число используемых GPU, "
            "и небольшая потеря качества допустима"
        ),
        method=METHOD_INT4,
        reason="Меньше памяти на веса — меньше карт. Потерю качества условие уже разрешает.",
    ),
    OptimizationCase(
        scenario=(
            "Нужна качественная посттренировочная INT4-квантизация, "
            "и есть возможность прогнать калибровочный набор"
        ),
        method=METHOD_GPTQ_AWQ,
        reason="GPTQ и AWQ как раз калибруют уже обученную модель, без нового обучения.",
    ),
    OptimizationCase(
        scenario=(
            "Нужно получить INT4-модель с меньшей потерей качества, "
            "используя статистику на калибровочных данных"
        ),
        method=METHOD_GPTQ_AWQ,
        reason=(
            "Оба метода минимизируют ошибку выхода, а не ошибку каждого веса. "
            "GPTQ компенсирует округление внутри слоя, AWQ защищает веса, "
            "на которые сильнее всего реагируют активации."
        ),
    ),
)


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} должен быть целым числом не меньше 1")


def attention_matrix_elements(seq_len: int) -> int:
    """Сколько чисел в полной матрице внимания seq × seq.

    Для 8 тысяч токенов это 64 миллиона. Именно эту матрицу обычный
    attention несколько раз читает и пишет в HBM.
    """
    _positive_int("seq_len", seq_len)
    return seq_len * seq_len


def attention_matrix_bytes(seq_len: int, dtype_bytes: int = 2) -> int:
    """Байты одной матрицы внимания.

    По умолчанию float16, два байта на число. Это одна матрица, без
    голов и без батча: реальный проход занимает ещё больше.
    """
    _positive_int("dtype_bytes", dtype_bytes)
    return attention_matrix_elements(seq_len) * dtype_bytes


def fits_in_sram(
    seq_len: int,
    sram_bytes: int = DEFAULT_SRAM_BYTES,
    dtype_bytes: int = 2,
) -> bool:
    """Помещается ли полная матрица в быструю память на чипе.

    Ядра считают только то, что уже лежит в SRAM. Если матрица больше,
    её приходится целиком держать в HBM и несколько раз оттуда возить.
    """
    _positive_int("sram_bytes", sram_bytes)
    return attention_matrix_bytes(seq_len, dtype_bytes) <= sram_bytes


def weight_memory_bytes(num_parameters: int, bits: int) -> int:
    """Байты под веса при заданной ширине числа.

    Float16 — 16 бит, INT8 — 8, INT4 — 4. Формула не учитывает масштабы
    блоков и слои, которые оставляют во float16: это нижняя оценка.
    """
    _positive_int("num_parameters", num_parameters)
    if bits not in {4, 8, 16, 32}:
        raise ValueError("bits должен быть 4, 8, 16 или 32")
    return num_parameters * bits // 8


def weight_memory_gib(num_parameters: int, bits: int) -> float:
    """Те же байты в гибибайтах, деление на 1024³."""
    return weight_memory_bytes(num_parameters, bits) / GIB


def quantization_scale(low: float, high: float, bits: int = INT4_BITS) -> float:
    """Шаг равномерной сетки.

    ``bits`` дают ``2**bits`` ступенек и на одну меньше промежутков.
    Урок делит размах 0.8 на 15 промежутков INT4 и получает около 0.053.
    """
    _positive_int("bits", bits)
    if high <= low:
        raise ValueError("high должен быть больше low")
    return (high - low) / ((2**bits) - 1)


def quantize_uniform(
    value: float,
    low: float,
    high: float,
    bits: int = INT4_BITS,
) -> tuple[int, float]:
    """Номер ближайшей равномерной ступеньки и восстановленное значение.

    Индекс лежит в ``0 ... 2**bits - 1``. Восстановленное число совпадает
    с исходным только если вес уже стоял на узле сетки. Разница — ошибка
    квантизации.

    Returns:
        Индекс ступеньки и значение после деквантизации.
    """
    scale = quantization_scale(low, high, bits)
    levels = 2**bits
    index = int(round((value - low) / scale))
    index = min(max(index, 0), levels - 1)
    return index, low + index * scale


def nearest_level(value: float, levels: tuple[float, ...] | list[float]) -> tuple[int, float]:
    """Ближайший уровень произвольной сетки, например NF4."""
    if not levels:
        raise ValueError("levels не должен быть пустым")
    index = min(range(len(levels)), key=lambda item: abs(levels[item] - value))
    return index, levels[index]


def laptop_price(base_rub: int = LAPTOP_BASE_RUB) -> int:
    """Итог задачи про ноутбук: +15%, затем −25%, затем −10%.

    80 000 × 1.15 = 92 000, × 0.75 = 69 000, × 0.90 = 62 100.
    Целочисленный путь не копит ошибку float.
    """
    _positive_int("base_rub", base_rub)
    after_raise = base_rub * 115 // 100
    after_sale = after_raise * 75 // 100
    return after_sale * 90 // 100


def mentions_amount(text: str, amount: int) -> bool:
    """Есть ли сумма в ответе, если пробелы и неразрывные пробелы убрать."""
    compact = (
        text.replace(" ", "")
        .replace("\u00a0", "")
        .replace("\u202f", "")
    )
    return str(amount) in compact


def nf4_stored_bytes(
    num_weights: int,
    block_size: int = NF4_BLOCK_SIZE,
    scale_group: int = NF4_SCALE_GROUP,
    double_quant: bool = True,
) -> int:
    """Оценка хранимых байт NF4: 4 бита на вес плюс масштабы блоков.

    Без double quant на каждый блок из ``block_size`` весов лежит один
    float32-scale. С double quant сами scale сжимаются до байта, а на
    группу таких scale остаётся один float32. Это схема идеи bitsandbytes,
    не раскладка его CUDA-структур по полям.
    """
    _positive_int("num_weights", num_weights)
    _positive_int("block_size", block_size)
    _positive_int("scale_group", scale_group)
    payload = (num_weights * 4 + 7) // 8
    n_blocks = (num_weights + block_size - 1) // block_size
    if not double_quant:
        return payload + n_blocks * 4
    n_groups = (n_blocks + scale_group - 1) // scale_group
    return payload + n_blocks + n_groups * 4


def weight_storage_report(
    model: torch.nn.Module,
    block_size: int = NF4_BLOCK_SIZE,
    scale_group: int = NF4_SCALE_GROUP,
) -> dict[str, int]:
    """Сколько байт занимают веса сейчас и сколько занял бы NF4.

    Сжимаются только матрицы ``nn.Linear``. Эмбеддинги, нормы и смещения
    остаются в исходном dtype — bitsandbytes тоже не упаковывает всё подряд.
    Отчёт считает хранение, а не ``memory_allocated``: аллокатор и кеш
    сюда не входят.
    """
    linear_ids = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, torch.nn.Linear)
    }
    seen: set[int] = set()
    float_bytes = 0
    other_bytes = 0
    linear_params = 0
    linear_bytes = 0
    for parameter in model.parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        nbytes = parameter.numel() * parameter.element_size()
        float_bytes += nbytes
        if identity in linear_ids:
            linear_params += parameter.numel()
            linear_bytes += nbytes
        else:
            other_bytes += nbytes
    if linear_params == 0:
        raise ValueError("в модели нет Linear-весов")
    linear_nf4 = nf4_stored_bytes(linear_params, block_size, scale_group, True)
    return {
        "linear_params": linear_params,
        "float_bytes": float_bytes,
        "linear_bytes": linear_bytes,
        "other_bytes": other_bytes,
        "nf4_bytes": linear_nf4 + other_bytes,
        "linear_nf4_bytes": linear_nf4,
    }


def _quantize_dequantize_scales(scales: torch.Tensor, group: int) -> torch.Tensor:
    """Сжимает масштабы блоков в INT8 и сразу разворачивает обратно."""
    flat = scales.reshape(-1)
    pad = (group - flat.numel() % group) % group
    padded = functional.pad(flat, (0, pad)) if pad else flat
    blocks = padded.view(-1, group)
    group_scale = blocks.amax(dim=1)
    safe = group_scale.clamp(min=1e-12)
    codes = torch.round(blocks / safe.unsqueeze(1) * 127).clamp(0, 127)
    restored = codes / 127 * group_scale.unsqueeze(1)
    return restored.reshape(-1)[: flat.numel()].view_as(scales)


def _nf4_roundtrip(
    weight: torch.Tensor,
    block_size: int,
    scale_group: int,
) -> torch.Tensor:
    """Округляет тензор к NF4 по блокам и возвращает float той же формы."""
    flat = weight.detach().float().reshape(-1)
    pad = (block_size - flat.numel() % block_size) % block_size
    padded = functional.pad(flat, (0, pad)) if pad else flat
    blocks = padded.view(-1, block_size)
    scales = blocks.abs().amax(dim=1)
    scales = _quantize_dequantize_scales(scales, scale_group)
    codebook = torch.tensor(NF4_CODEBOOK, device=blocks.device, dtype=torch.float32)
    restored = torch.empty_like(blocks)
    # Матрица расстояний на весь вес не влезает: ищем ближайший код кусками.
    chunk = 8192
    for start in range(0, blocks.shape[0], chunk):
        part = blocks[start : start + chunk]
        part_scale = scales[start : start + chunk]
        safe = part_scale.clamp(min=1e-8)
        normalized = part / safe.unsqueeze(1)
        codes = (normalized.unsqueeze(-1) - codebook).abs().argmin(dim=-1)
        restored[start : start + chunk] = codebook[codes] * part_scale.unsqueeze(1)
    trimmed = restored.reshape(-1)[: flat.numel()]
    return trimmed.view(weight.shape)


def round_linear_weights_to_nf4(
    model: torch.nn.Module,
    block_size: int = NF4_BLOCK_SIZE,
    scale_group: int = NF4_SCALE_GROUP,
) -> dict[str, int]:
    """Вносит в Linear-веса ту же ошибку, что NF4 с double quant.

    Восстановленные значения записываются обратно во float. Генерация
    после этого идёт обычной арифметикой и показывает цену округления,
    но не экономию памяти и не стоимость разворота весов на каждом
    умножении. Повторный вызов округляет уже испорченные веса ещё раз.
    """
    _positive_int("block_size", block_size)
    _positive_int("scale_group", scale_group)
    seen: set[int] = set()
    linears = 0
    weight_numel = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            pointer = module.weight.data_ptr()
            if pointer in seen or module.weight.numel() == 0:
                continue
            seen.add(pointer)
            restored = _nf4_roundtrip(module.weight.data, block_size, scale_group)
            module.weight.copy_(restored.to(dtype=module.weight.dtype))
            linears += 1
            weight_numel += module.weight.numel()
    if linears == 0:
        raise ValueError("в модели нет Linear-весов")
    return {"linears": linears, "weight_numel": weight_numel}


def flash_attention_available() -> bool:
    """Есть ли CUDA-ядра flash-attention 2, которые включает transformers.

    Пакет ``flash_attn`` собран под NVIDIA. На MPS и CPU его нет,
    даже если строка ``attn_implementation`` написана в коде загрузки.
    """
    return torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None


def int4_kernels_available() -> bool:
    """Можно ли загрузить NF4 через bitsandbytes на этой машине.

    Нужны и CUDA, и установленный пакет. На Apple Silicon условие ложно.
    """
    return torch.cuda.is_available() and importlib.util.find_spec("bitsandbytes") is not None


def load_model(model_name: str, use_int4: bool) -> torch.nn.Module:
    """Грузит модель как в задании: float16 либо NF4 bitsandbytes.

    INT4 требует ``quantization_config`` и ``device_map="cuda"``.
    Без ядер bitsandbytes функция отказывается, а не подменяет загрузку
    тихим float16. Ошибку округления на MPS показывает
    ``round_linear_weights_to_nf4``.

    Args:
        model_name: Имя модели на Hugging Face.
        use_int4: ``True`` — NF4, double quant, вычисления во float16.

    Returns:
        Модель в режиме eval.
    """
    from transformers import AutoModelForCausalLM

    if use_int4:
        if not int4_kernels_available():
            raise RuntimeError(
                "INT4 через bitsandbytes нужен CUDA и пакет bitsandbytes. "
                "На этой машине их нет."
            )
        from transformers import BitsAndBytesConfig

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=config,
            device_map="cuda",
        )
        model.eval()
        return model

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model


def accelerator_allocated_bytes(device: torch.device) -> int:
    """Сколько байт сейчас числится за CUDA или MPS.

    На CPU аллокатора ускорителя нет, функция возвращает 0. Для весов
    на CPU смотрите ``weight_storage_report``.
    """
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    return 0


def benchmark_chat(
    model: torch.nn.Module,
    tokenizer,
    questions: tuple[str, ...] | list[str] = COMPARISON_QUESTIONS,
    max_new_tokens: int = COMPARISON_MAX_NEW_TOKENS,
    label: str = "",
) -> dict[str, float | int | list[str]]:
    """Память, скорость и ответы на вопросы урока.

    Память снимается до прогрева. Прогрев на слове «Привет» не входит
    в таймер: первый вызов после загрузки платит за подготовку ядер.
    Вопросы идут через chat template, генерация жадная. Скорость — это
    новые токены в секунду, потому что две версии могут ответить разной
    длиной. Конец последовательности ответ обрезает, ``min_new_tokens``
    не ставится.

    Args:
        model: Уже загруженная causal-модель.
        tokenizer: Токенизатор с chat template.
        questions: Тексты пользовательских реплик.
        max_new_tokens: Потолок длины одного ответа.
        label: Префикс сообщений о прогрессе. Пустая строка молчит.

    Returns:
        Память ускорителя в байтах, секунды, число новых токенов,
        токены в секунду и тексты ответов.
    """
    _positive_int("max_new_tokens", max_new_tokens)
    if not questions:
        raise ValueError("questions не должен быть пустым")
    device = next(model.parameters()).device
    prefix = f"{label}: " if label else ""
    synchronize_device(device)
    memory_bytes = accelerator_allocated_bytes(device)

    if label:
        print(f"{prefix}прогрев", flush=True)
    warmup = tokenizer("Привет", return_tensors="pt")
    warmup = {key: value.to(device) for key, value in warmup.items()}
    with torch.inference_mode():
        model.generate(**warmup, max_new_tokens=5, do_sample=False)

    answers: list[str] = []
    total_new_tokens = 0
    synchronize_device(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for index, question in enumerate(questions, start=1):
            if label:
                print(f"{prefix}вопрос {index}/{len(questions)}", flush=True)
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            new_tokens = output[0][inputs["input_ids"].shape[1] :]
            total_new_tokens += int(new_tokens.shape[0])
            answers.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    synchronize_device(device)
    elapsed = time.perf_counter() - started
    if elapsed <= 0:
        raise RuntimeError("замер генерации остановился на нуле секунд")
    return {
        "memory_bytes": memory_bytes,
        "seconds": elapsed,
        "new_tokens": total_new_tokens,
        "tok_per_sec": total_new_tokens / elapsed,
        "answers": answers,
    }
