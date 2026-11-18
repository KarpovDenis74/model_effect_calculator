"""Токены, причина остановки и условная стоимость ответа."""

from utils.cost import estimate_cost

# Одинаковые учебные тарифы для сравнения моделей. Это не прайс OpenRouter:
# бесплатные модели фактически стоят 0, а здесь видна разница длины ответов.
INPUT_PRICE_PER_MILLION = 1.0
OUTPUT_PRICE_PER_MILLION = 4.0


def calculate_estimated_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_price_per_million: float = INPUT_PRICE_PER_MILLION,
    output_price_per_million: float = OUTPUT_PRICE_PER_MILLION,
) -> float:
    """Считает условную стоимость запроса в долларах.

    Вход и выход умножаются на разные цены за миллион токенов. Длинный
    ответ обычно дороже длинного промпта, потому что выходной тариф выше.

    Args:
        prompt_tokens: Токены всех сообщений, переданных модели.
        completion_tokens: Токены, которые модель сгенерировала.
        input_price_per_million: Цена миллиона входных токенов.
        output_price_per_million: Цена миллиона выходных токенов.

    Returns:
        Стоимость одного запроса.
    """
    return estimate_cost(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
    )


def token_usage(response) -> dict[str, int]:
    """Достаёт из ответа число входных, выходных и общих токенов.

    ``prompt_tokens`` включает весь список ``messages``: системную
    инструкцию, текущий запрос и предыдущие реплики. ``total_tokens``
    равен сумме входа и выхода.

    Args:
        response: Ответ Chat Completions API с заполненным ``usage``.

    Returns:
        Словарь с тремя счётчиками.

    Raises:
        RuntimeError: сервис не прислал статистику.
    """
    if response.usage is None:
        raise RuntimeError(f"Модель {response.model} не вернула usage")

    usage = response.usage
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def usage_is_consistent(response) -> bool:
    """Проверяет, что вход плюс выход равны общему числу токенов."""
    usage = token_usage(response)
    return usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]


def warn_if_truncated(finish_reason: str | None) -> bool:
    """Сообщает, если ответ упёрся в ``max_tokens``.

    ``stop`` означает, что модель закончила ответ сама. ``length`` означает,
    что генерация остановилась на лимите и текст мог оборваться.

    Args:
        finish_reason: Поле ``choices[0].finish_reason``.

    Returns:
        ``True``, если ответ мог быть обрезан.
    """
    if finish_reason == "length":
        print("Ответ мог быть обрезан. Увеличьте max_tokens или сократите запрос.")
        return True
    return False
