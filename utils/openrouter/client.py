"""Клиент OpenRouter: ключ, запрос и повторы временных ошибок.

OpenRouter принимает тот же Chat Completions, что и OpenAI. Поэтому
используется класс ``OpenAI``, но ``base_url`` указывает на OpenRouter.
Встроенные повторы библиотеки отключены: пауза и число попыток задаются
здесь, через exponential backoff.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Эти ошибки могут исчезнуть сами: лимит, обрыв связи, тайм-аут, 5xx.
# 400, 401 и 403 сюда не входят: повтор того же запроса их не исправит.
RETRYABLE_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)


def load_env_file(path: str | Path) -> None:
    """Подставляет переменные из файла, если их ещё нет в окружении.

    Файл не обязателен. Уже заданные переменные не перезаписываются.
    Значения не печатаются.

    Args:
        path: Путь к файлу вида ``KEY=value``.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def require_openrouter_api_key(env_file: str | Path | None = None) -> str:
    """Возвращает ключ OpenRouter из окружения.

    Если переменной ещё нет и передан ``env_file``, сначала читает его.
    Сам ключ наружу не печатается: вызывающий код узнаёт только то, что
    значение найдено, либо получает ошибку.

    Args:
        env_file: Необязательный файл с переменными окружения.

    Returns:
        Значение ``OPENROUTER_API_KEY``.

    Raises:
        RuntimeError: переменная не задана.
    """
    if env_file is not None:
        load_env_file(env_file)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Переменная окружения OPENROUTER_API_KEY не задана")
    return api_key


def create_openrouter_client(
    api_key: str,
    timeout: float = 30.0,
    max_retries: int = 0,
) -> OpenAI:
    """Создаёт клиент, который ходит в OpenRouter, а не в OpenAI напрямую.

    ``max_retries=0`` отключает скрытые повторы библиотеки. Иначе одна
    попытка ``create_chat_completion_with_retry`` могла бы внутри себя
    обратиться к API несколько раз.

    Args:
        api_key: Ключ OpenRouter.
        timeout: Сколько секунд ждать ответ одного запроса.
        max_retries: Сколько раз библиотека повторяет запрос сама.

    Returns:
        Настроенный клиент.
    """
    if timeout <= 0:
        raise ValueError("timeout должен быть больше 0")
    if max_retries < 0:
        raise ValueError("max_retries не может быть отрицательным")

    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        max_retries=max_retries,
        timeout=timeout,
    )


def calculate_retry_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> float:
    """Считает паузу перед следующей попыткой.

    Формула урока: ``min(max_delay, base_delay * 2 ** (attempt - 1) + jitter)``.
    ``jitter`` — случайное смещение от 0 до ``base_delay``, чтобы несколько
    клиентов не повторили запрос в одну и ту же секунду.

    Args:
        attempt: Номер неудачной попытки, начиная с 1.
        base_delay: Начальная задержка в секундах.
        max_delay: Потолок ожидания в секундах.

    Returns:
        Задержку в секундах.
    """
    if attempt < 1:
        raise ValueError("attempt должен быть не меньше 1")
    if base_delay < 0 or max_delay < 0:
        raise ValueError("задержки не могут быть отрицательными")

    exponential_delay = base_delay * 2 ** (attempt - 1)
    jitter = random.uniform(0, base_delay)
    return min(exponential_delay + jitter, max_delay)


def log_response_usage(
    response,
    usage_logger: logging.Logger | None = None,
) -> float | None:
    """Пишет в лог модель, входные и выходные токены и условную стоимость.

    Стоимость считается учебными тарифами из ``calculate_estimated_cost``:
    1$ за миллион входных токенов и 4$ за миллион выходных. Для бесплатной
    модели фактическое списание OpenRouter может быть нулём. Учебная цена
    нужна, чтобы сравнивать длину запросов и ответов.

    Args:
        response: Ответ ``chat.completions.create``.
        usage_logger: Куда писать строку. По умолчанию логгер этого модуля.

    Returns:
        Условную стоимость или ``None``, если сервис не прислал ``usage``.
    """
    from utils.openrouter.usage import calculate_estimated_cost

    active_logger = logger if usage_logger is None else usage_logger
    if response.usage is None:
        active_logger.warning(
            "Модель %s не вернула информацию об использовании",
            response.model,
        )
        return None

    estimated_cost = calculate_estimated_cost(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    )
    active_logger.info(
        (
            "model=%s prompt_tokens=%d completion_tokens=%d "
            "estimated_cost_usd=%.8f"
        ),
        response.model,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        estimated_cost,
    )
    return estimated_cost


def create_chat_completion_with_retry(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 150,
    max_attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    usage_logger: logging.Logger | None = None,
):
    """Отправляет запрос и повторяет его только после временных ошибок.

    Повторяются ``RateLimitError``, ``APITimeoutError``,
    ``APIConnectionError`` и ``InternalServerError``. ``BadRequestError``
    и ошибки доступа пробрасываются сразу: тот же запрос вернёт их снова.

    ``max_attempts`` считает все обращения к API, включая первое.
    После последней неудачи исключение выбрасывается наружу, пустой
    ответ не подменяется.

    Args:
        client: Клиент из ``create_openrouter_client``.
        model: Идентификатор модели или маршрутизатор, например
            ``openrouter/free``.
        messages: Сообщения ``system``, ``user`` и ``assistant`` по порядку.
        temperature: Случайность выбора токенов.
        max_tokens: Верхняя граница длины ответа.
        max_attempts: Сколько раз можно обратиться к API.
        base_delay: Начальная пауза между попытками.
        max_delay: Потолок паузы.
        usage_logger: Логгер для статистики успешного ответа.

    Returns:
        Успешный ответ Chat Completions API.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts должен быть не меньше 1")

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            log_response_usage(response, usage_logger=usage_logger)
            return response
        except RETRYABLE_ERRORS as error:
            if attempt == max_attempts:
                logger.exception(
                    "Запрос не выполнен после %d попыток",
                    max_attempts,
                )
                raise

            delay = calculate_retry_delay(
                attempt=attempt,
                base_delay=base_delay,
                max_delay=max_delay,
            )
            logger.warning(
                "Ошибка %s. Повтор %d из %d через %.2f с",
                type(error).__name__,
                attempt + 1,
                max_attempts,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError("Запрос не был выполнен")
