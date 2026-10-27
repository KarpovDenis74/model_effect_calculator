"""Разбор небольшой causal-модели: архитектура, параметры, память и скорость."""

import time
from functools import lru_cache

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def _ensure_safetensors_checkpoint(model_name: str) -> None:
    """Если в кеше только ``pytorch_model.bin``, конвертирует его в safetensors.

    transformers 5.x на torch < 2.6 отказывается читать pickle-веса из-за
    CVE-2025-32434. Для старых моделей вроде TinyStories в репозитории часто
    есть только ``.bin``. Конвертация один раз кладёт ``model.safetensors``
    рядом в локальный кеш Hugging Face.
    """
    from pathlib import Path

    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    from safetensors.torch import save_file

    cached_safe = try_to_load_from_cache(model_name, "model.safetensors")
    if isinstance(cached_safe, str) and Path(cached_safe).exists():
        return

    bin_path = hf_hub_download(model_name, "pytorch_model.bin")
    safe_path = Path(bin_path).with_name("model.safetensors")
    if safe_path.exists():
        return

    state_dict = torch.load(bin_path, map_location="cpu")
    save_file(state_dict, str(safe_path))


@lru_cache(maxsize=4)
def load_causal_model(
    model_name: str,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Загружает токенизатор и causal language model.

    Такая модель предсказывает следующий токен только по уже известному
    левому контексту. GPT-2 и distilgpt2 относятся к этому типу.
    После загрузки включается ``eval``: dropout не мешает генерации.
    Повторный вызов с тем же именем возвращает уже загруженные объекты.

    Args:
        model_name: Имя модели на Hugging Face.

    Returns:
        Токенизатор и модель.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except ValueError as error:
        if "torch.load" not in str(error) and "CVE-2025-32434" not in str(error):
            raise
        _ensure_safetensors_checkpoint(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_safetensors=True,
        )

    # У многих GPT-подобных моделей нет отдельного pad-токена.
    # Для generate() его подменяют концом последовательности.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return tokenizer, model


def architecture_summary(model: torch.nn.Module) -> dict[str, int]:
    """Достаёт из конфига масштаб GPT-подобной модели.

    ``n_layer`` — сколько блоков трансформера, ``n_head`` — сколько голов
    внимания в блоке, ``n_embd`` — длина вектора токена, ``vocab_size`` —
    сколько токенов модель может оценить на выходе, ``n_positions`` —
    сколько позиций умеет таблица позиционных эмбеддингов.

    Args:
        model: Загруженная causal-модель с конфигом GPT-2.

    Returns:
        Словарь с этими пятью числами.
    """
    config = model.config
    return {
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "vocab_size": config.vocab_size,
        "n_positions": config.n_positions,
    }


def count_parameters(model: torch.nn.Module) -> int:
    """Возвращает количество параметров модели."""
    return sum(parameter.numel() for parameter in model.parameters())


def estimate_model_weights_memory_mb(model: torch.nn.Module) -> float:
    """Оценивает память под веса модели в текущем формате хранения.

    Учитывает фактический dtype каждого параметра через ``element_size()``.
    Это только веса. Промежуточные активации, KV-cache и служебные структуры
    фреймворка сюда не входят.

    Args:
        model: Загруженная модель.

    Returns:
        Память под веса в мегабайтах.
    """
    memory_bytes = 0
    for parameter in model.parameters():
        memory_bytes += parameter.numel() * parameter.element_size()
    return memory_bytes / 1024**2


def estimate_memory_mb(num_params: int, bytes_per_param: int) -> float:
    """Оценивает память под веса модели для заданного формата параметров.

    Args:
        num_params: Количество параметров модели.
        bytes_per_param: Сколько байт занимает один параметр.
            Например, float32 — 4 байта, float16 — 2 байта, int8 — 1 байт.

    Returns:
        Оценка памяти под веса модели в мегабайтах.
    """
    memory_bytes = num_params * bytes_per_param
    memory_mb = memory_bytes / 1024**2
    return memory_mb


def generate_continuation(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 30,
) -> str:
    """Генерирует продолжение промпта жадным выбором следующего токена.

    ``do_sample=False`` отключает случайный выбор, поэтому повторный запуск
    на том же промпте даёт тот же текст. ``pad_token_id`` берётся из конца
    последовательности: у GPT-2 отдельного pad-токена нет.

    Args:
        model: Causal-модель в режиме ``eval``.
        tokenizer: Токенизатор этой модели.
        prompt: Начало текста.
        max_new_tokens: Сколько новых токенов дописать.

    Returns:
        Промпт вместе со сгенерированным продолжением.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def measure_generation_speed(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    num_runs: int = 3,
) -> dict:
    """Измеряет среднее время и скорость генерации модели.

    Первый вызов ``generate`` не учитывается: он медленнее из-за разовой
    инициализации кешей и служебных структур. Дальше функция ``num_runs`` раз
    замеряет генерацию и усредняет время и число новых токенов.

    Args:
        model: Модель для causal language modeling.
        tokenizer: Токенизатор, соответствующий модели.
        prompt: Текстовый промпт для генерации.
        max_new_tokens: Максимальное число новых токенов для генерации.
        num_runs: Количество запусков для усреднения времени.

    Returns:
        Словарь с временем генерации, числом новых токенов и скоростью
        в токенах в секунду.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    times = []
    generated_token_counts = []

    for _ in range(num_runs):
        start_time = time.perf_counter()

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        end_time = time.perf_counter()

        output_length = output_ids.shape[1]
        generated_tokens = output_length - input_length

        times.append(end_time - start_time)
        generated_token_counts.append(generated_tokens)

    avg_time_sec = sum(times) / len(times)
    avg_generated_tokens = sum(generated_token_counts) / len(generated_token_counts)
    tokens_per_sec = avg_generated_tokens / avg_time_sec

    return {
        "generation_time_sec": avg_time_sec,
        "generated_tokens": avg_generated_tokens,
        "tokens_per_sec": tokens_per_sec,
    }
