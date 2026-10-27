def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
    cached_tokens: int = 0,
    cached_price_per_million: float = 0.0,
) -> float:
    """Считает стоимость одного запроса по ценам за 1 миллион токенов.

    Вход и выход считаются отдельно, потому что у модели на них разные цены.
    Стоимость пропорциональна числу токенов: 500 токенов при цене 4.50
    за миллион стоят ``500 / 1_000_000 * 4.50``.

    Если начало промпта повторяется и попадает в кеш, эти токены считаются
    по ``cached_price_per_million``, а остальные входные — по обычной цене.
    Без кеша оба дополнительных аргумента можно не передавать.

    Args:
        input_tokens: Число токенов во входном запросе, включая кешированную часть.
        output_tokens: Число токенов в ожидаемом ответе.
        input_price_per_million: Цена обычных входных токенов за 1 миллион.
        output_price_per_million: Цена выходных токенов за 1 миллион.
        cached_tokens: Сколько входных токенов попадает в кеш.
        cached_price_per_million: Цена кешированных входных токенов за 1 миллион.

    Returns:
        Стоимость одного запроса в тех же единицах, что и переданные цены.
    """
    if cached_tokens < 0:
        raise ValueError("cached_tokens не может быть отрицательным")
    if cached_tokens > input_tokens:
        raise ValueError("cached_tokens не может быть больше input_tokens")

    fresh_input_tokens = input_tokens - cached_tokens
    input_cost = fresh_input_tokens / 1_000_000 * input_price_per_million
    cached_cost = cached_tokens / 1_000_000 * cached_price_per_million
    output_cost = output_tokens / 1_000_000 * output_price_per_million
    return input_cost + cached_cost + output_cost
