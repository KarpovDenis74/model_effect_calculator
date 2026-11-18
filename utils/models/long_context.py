"""Длинный контекст: цена внимания, TTFT и порог продукта.

Внимание сопоставляет каждый токен с каждым, поэтому память и вычисления
растут как n². Пока модель читает весь промпт, пользователь ждёт первый
токен: это TTFT. Стоимость запроса при коротком ответе почти целиком
задаётся входными токенами. Качество тоже страдает: факт из середины
длинного контекста модель использует хуже, чем факт с края.
"""

from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TextIteratorStreamer

from utils.cost import estimate_cost

CONTEXT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
FILLER_SENTENCE = (
    "Инференс языковых моделей в продакшене требует внимания "
    "к задержкам и стоимости. "
)
CONTEXT_QUESTION = "\n\nОдним предложением: о чём шла речь в тексте выше?"
DEFAULT_TARGET_LENGTHS = (512, 1024, 2048, 4096)

# Типовой тариф недорогой модели, доллар за 1 млн токенов.
LESSON_INPUT_PRICE_PER_MILLION = 0.15
LESSON_OUTPUT_PRICE_PER_MILLION = 0.60

# Ограничения продакт-менеджера из практики.
TTFT_LIMIT_SECONDS = 1.0
REQUEST_COST_LIMIT_USD = 0.0005

# До этой длины короткий запрос для окна в десятки тысяч токенов
# ещё не выглядит дорогим. Крупный документ эту границу переходит.
NOTICEABLE_CONTEXT_TOKENS = 2000

LOAD_OK = "Работает нормально"
LOAD_EXPENSIVE = "Заметно дороже и медленнее"
LOAD_OVERFLOW = "Отказ: контекст не помещается"

Placement = Literal["start", "middle", "end"]
_PLACEMENT_ORDER = {"start": 0, "middle": 1, "end": 2}


@dataclass(frozen=True)
class PromptPart:
    """Кусок промпта и место, где его стоит держать.

    ``start`` и ``end`` — края контекста, их модель использует лучше.
    ``middle`` — середина: сюда уходит фон, без которого ответ не ломается.
    """

    label: str
    text: str
    placement: Placement


def attention_matrix_cells(num_tokens: int) -> int:
    """Считает ячейки матрицы внимания для контекста из ``num_tokens``.

    Каждая ячейка — сравнение токена i с токеном j. Таких пар n * n.

    Args:
        num_tokens: Длина контекста.

    Returns:
        Число ячеек.
    """
    if num_tokens < 0:
        raise ValueError("num_tokens не может быть отрицательным")
    return num_tokens * num_tokens


def attention_scale_factor(short_tokens: int, long_tokens: int) -> float:
    """Во сколько раз вырастет матрица внимания при удлинении контекста.

    Отношение длин k даёт рост площади k². Удвоение контекста учетверяет
    матрицу.

    Args:
        short_tokens: Исходная длина.
        long_tokens: Новая длина.

    Returns:
        Во сколько раз новая матрица больше старой.
    """
    if short_tokens <= 0 or long_tokens < 0:
        raise ValueError("длины должны быть положительными")
    short_cells = attention_matrix_cells(short_tokens)
    return attention_matrix_cells(long_tokens) / short_cells


def describe_context_load(
    prompt_tokens: int,
    context_window: int,
    noticeable_from: int = NOTICEABLE_CONTEXT_TOKENS,
) -> str:
    """Оценивает, как модель с заданным окном переварит промпт.

    Сначала проверяется, помещается ли промпт в окно. Если помещается,
    короткие запросы остаются обычными, а длинные становятся заметно
    дороже и медленнее: матрица внимания растёт квадратом длины.

    Args:
        prompt_tokens: Длина входа.
        context_window: Окно модели в токенах.
        noticeable_from: С этой длины цена контекста уже ощутима,
            хотя запрос ещё входит в окно.

    Returns:
        Одна из подписей: нормально, дороже и медленнее, отказ.
    """
    if prompt_tokens < 0 or context_window <= 0 or noticeable_from < 0:
        raise ValueError("длины контекста должны быть неотрицательными")
    if prompt_tokens > context_window:
        return LOAD_OVERFLOW
    if prompt_tokens <= noticeable_from:
        return LOAD_OK
    return LOAD_EXPENSIVE


def order_prompt_parts(parts: list[PromptPart]) -> list[PromptPart]:
    """Ставит важное на края, а фон оставляет в середине.

    Порядок внутри одной зоны сохраняется: две части с ``placement="end"``
    идут в том порядке, в котором их передали.
    """
    return sorted(parts, key=lambda part: _PLACEMENT_ORDER[part.placement])


def render_prompt(parts: list[PromptPart]) -> str:
    """Собирает текст промпта после раскладки по lost in the middle."""
    return "\n\n".join(part.text for part in order_prompt_parts(parts))


def inference_device() -> torch.device:
    """Выбирает устройство для локального замера.

    На Apple Silicon это MPS, при видеокарте NVIDIA — CUDA, иначе CPU.
    Урок в ноутбуке оставляет модель на CPU. Здесь берётся ускоритель,
    если он есть: иначе лесенка до 4096 токенов занимает слишком много
    времени. Смысл сравнения длин от устройства не меняется.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_prompt(
    tokenizer: PreTrainedTokenizerBase,
    target_tokens: int,
    filler_sentence: str = FILLER_SENTENCE,
    question: str = CONTEXT_QUESTION,
) -> str:
    """Собирает промпт примерно нужной длины.

    Одно и то же задание стоит в конце. Длину набирает нейтральный
    наполнитель, поэтому сравниваются длина контекста и задержка,
    а не разные вопросы.

    Args:
        tokenizer: Токенизатор той модели, на которой будет замер.
        target_tokens: Нижняя граница длины наполнителя в токенах.
        filler_sentence: Повторяемый нейтральный текст.
        question: Вопрос, общий для всех длин.

    Returns:
        Наполнитель плюс вопрос. Фактическая длина может быть чуть
        больше ``target_tokens``: цикл останавливается на целом предложении.
    """
    if target_tokens < 1:
        raise ValueError("target_tokens должен быть не меньше 1")
    if not filler_sentence:
        raise ValueError("filler_sentence не должен быть пустым")

    text = ""
    steps = 0
    while len(tokenizer(text)["input_ids"]) < target_tokens:
        text += filler_sentence
        steps += 1
        if steps > target_tokens:
            raise RuntimeError(
                "Наполнитель не увеличивает число токенов, промпт не собрать"
            )
    return text + question


def estimate_context_cost(
    prompt_tokens: int,
    output_tokens: int,
    input_price_per_million: float = LESSON_INPUT_PRICE_PER_MILLION,
    output_price_per_million: float = LESSON_OUTPUT_PRICE_PER_MILLION,
) -> float:
    """Оценивает стоимость одного запроса по тарифу за миллион токенов.

    Локальная модель провайдеру ничего не платит. Цифра нужна, чтобы
    заранее увидеть счёт того же сценария на платном API. При коротком
    ответе счёт растёт почти линейно от длины промпта.

    Args:
        prompt_tokens: Токены входа.
        output_tokens: Токены ответа.
        input_price_per_million: Цена входа за 1 млн токенов.
        output_price_per_million: Цена выхода за 1 млн токенов.

    Returns:
        Стоимость одного запроса в долларах.
    """
    return estimate_cost(
        prompt_tokens,
        output_tokens,
        input_price_per_million,
        output_price_per_million,
    )


def measure_ttft(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 100,
) -> dict[str, float | int]:
    """Замеряет время до первого токена и полное время ответа.

    Генерация идёт в отдельном потоке, а основной поток читает стример.
    TTFT — момент первого куска текста. E2E — момент, когда стрим
    закончился. Число выходных токенов считается по кускам стримера:
    при ``skip_special_tokens=True`` это один токен на шаг.

    Перед замером вызывается сборщик мусора, чтобы предыдущий прогон
    меньше влиял на следующий.

    Args:
        model: Causal-модель. Тензоры входа переносятся на её устройство.
        tokenizer: Токенизатор этой модели.
        prompt: Полный текст запроса.
        max_new_tokens: Сколько новых токенов генерировать.

    Returns:
        ``prompt_tokens``, ``output_tokens``, ``ttft`` и ``e2e``.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens должен быть не меньше 1")

    gc.collect()
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_tokens = int(inputs["input_ids"].shape[1])
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=180,
    )
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        streamer=streamer,
    )

    def _generate() -> None:
        with torch.inference_mode():
            model.generate(**gen_kwargs)

    thread = threading.Thread(target=_generate)
    start = time.perf_counter()
    thread.start()

    ttft = None
    generated_tokens = 0
    for index, _chunk in enumerate(streamer):
        if index == 0:
            ttft = time.perf_counter() - start
        generated_tokens += 1

    e2e = time.perf_counter() - start
    thread.join()

    if ttft is None:
        raise RuntimeError("Модель не вернула ни одного токена")

    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": generated_tokens,
        "ttft": ttft,
        "e2e": e2e,
    }


def fits_product_limits(
    ttft: float,
    cost: float,
    ttft_limit: float = TTFT_LIMIT_SECONDS,
    cost_limit: float = REQUEST_COST_LIMIT_USD,
) -> bool:
    """Проверяет, что запрос уложился и во время, и в бюджет."""
    return ttft <= ttft_limit and cost <= cost_limit


def summarize_context_budget(
    rows: list[dict],
    ttft_limit: float = TTFT_LIMIT_SECONDS,
    cost_limit: float = REQUEST_COST_LIMIT_USD,
) -> dict:
    """Находит, до какой длины оба ограничения ещё выполняются подряд.

    Строки должны идти от короткого контекста к длинному. Порог — это
    конец начальной серии, где оба лимита ещё выполняются. Первая строка
    после неё показывает узкое место: TTFT, стоимость или оба лимита.
    Более длинная точка может случайно снова уложиться в лимит, если
    замер на границе скачет. В правило она не входит.

    Args:
        rows: Замеры с ключами ``prompt_tokens``, ``ttft``, ``cost``.
        ttft_limit: Максимальное время до первого токена, секунды.
        cost_limit: Максимальная цена одного запроса, доллары.

    Returns:
        Решение по каждой строке, длина на конце проходящей серии и имя
        ограничения, которое сломалось первым. ``later_fitting_tokens`` —
        длины, которые прошли уже после первого нарушения.
    """
    if not rows:
        raise ValueError("Список замеров не должен быть пустым")

    decisions = []
    binding_limit = None
    for row in rows:
        ttft_ok = row["ttft"] <= ttft_limit
        cost_ok = row["cost"] <= cost_limit
        fits = ttft_ok and cost_ok
        decisions.append({**row, "ttft_ok": ttft_ok, "cost_ok": cost_ok, "fits": fits})
        if fits or binding_limit is not None:
            continue
        if not ttft_ok and not cost_ok:
            binding_limit = "both"
        elif not ttft_ok:
            binding_limit = "ttft"
        else:
            binding_limit = "cost"

    prefix = []
    for row in decisions:
        if not row["fits"]:
            break
        prefix.append(row)
    longest = prefix[-1]["prompt_tokens"] if prefix else None
    later_fitting_tokens = [
        row["prompt_tokens"]
        for row in decisions
        if row["fits"] and (longest is None or row["prompt_tokens"] > longest)
    ]
    return {
        "rows": decisions,
        "longest_fitting_tokens": longest,
        "later_fitting_tokens": later_fitting_tokens,
        "binding_limit": binding_limit,
        "ttft_limit": ttft_limit,
        "cost_limit": cost_limit,
    }


def plot_context_tradeoff(
    rows: list[dict],
    path: str | Path,
    ttft_limit: float = TTFT_LIMIT_SECONDS,
    cost_limit: float = REQUEST_COST_LIMIT_USD,
) -> Path:
    """Строит TTFT и стоимость против длины контекста и сохраняет рисунок.

    Горизонтальные линии — ограничения продукта. Точка выше линии уже
    не проходит соответствующий лимит.

    Args:
        rows: Замеры с ``prompt_tokens``, ``ttft`` и ``cost``.
        path: Куда сохранить PNG.
        ttft_limit: Линия на графике времени.
        cost_limit: Линия на графике стоимости.

    Returns:
        Путь к сохранённому файлу.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("Список замеров не должен быть пустым")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lengths = [row["prompt_tokens"] for row in rows]
    ttfts = [row["ttft"] for row in rows]
    costs = [row["cost"] for row in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(lengths, ttfts, marker="o", color="#2e6db4")
    ax1.axhline(ttft_limit, color="#c0392b", linestyle="--", label=f"лимит {ttft_limit:.0f} с")
    ax1.set_title("TTFT vs длина контекста")
    ax1.set_xlabel("токенов на входе")
    ax1.set_ylabel("TTFT, с")
    ax1.legend()

    ax2.plot(lengths, costs, marker="o", color="#07ab4b")
    ax2.axhline(cost_limit, color="#c0392b", linestyle="--", label=f"лимит ${cost_limit:.4f}")
    ax2.set_title("Стоимость vs длина контекста")
    ax2.set_xlabel("токенов на входе")
    ax2.set_ylabel("стоимость запроса, $")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
