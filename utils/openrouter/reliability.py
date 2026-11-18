"""JSON mode, корректирующие повторы и метрики технической надёжности.

Технический повтор отправляет тот же запрос после 429, тайм-аута или 5xx.
Корректирующий повтор нужен, когда API уже ответил, но ответ не прошёл
контракт: в историю кладутся предыдущий текст модели и описание ошибки.
Fallback — это безопасное действие после последней неудачи. Он не
увеличивает success_rate.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from utils.openrouter.contract import (
    ALLOWED_CATEGORIES,
    ALLOWED_PRIORITIES,
    MAX_SUMMARY_LENGTH,
    SUPPORT_SYSTEM_PROMPT,
    Message,
    ResponseValidationError,
    parse_and_validate,
)

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
TEMPERATURE = 0.7
MAX_TOKENS = 512

RequestFn = Callable[[list[Message]], str]


def build_support_messages(user_request: str) -> list[Message]:
    """Собирает system и user для разбора одного обращения.

    Args:
        user_request: Текст обращения. Инструкции внутри него остаются
            текстом пользователя и не заменяют системный промпт.

    Returns:
        Два сообщения для Chat Completions.
    """
    return [
        {"role": "system", "content": SUPPORT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Обращение пользователя:\n{user_request}"},
    ]


def correction_message(error: ResponseValidationError) -> str:
    """Описывает нарушение контракта так, чтобы модель могла его исправить."""
    return (
        "Предыдущий ответ не прошёл проверку. "
        f"Причина: {error}. "
        "Исправь ответ и верни только JSON-объект "
        "с полями category, priority и summary. "
        f"Допустимые category: {ALLOWED_CATEGORIES}. "
        f"Допустимые priority: {ALLOWED_PRIORITIES}. "
        f"Длина summary: не более {MAX_SUMMARY_LENGTH} символов."
    )


def collect_seeded_answers(
    client: OpenAI,
    model: str,
    messages: list[Message],
    seed: int,
    temperature: float = 0.7,
    runs: int = 2,
) -> list[str]:
    """Отправляет один и тот же запрос несколько раз с одним seed.

    Одинаковые модель, сообщения, температура и seed повышают шанс
    получить близкие ответы. Совпадение не гарантировано: провайдер
    может обновить модель или считать недетерминированно.

    Args:
        client: Клиент OpenRouter.
        model: Идентификатор модели с поддержкой ``seed``.
        messages: Сообщения запроса.
        seed: Начальное состояние генератора.
        temperature: Температура генерации.
        runs: Сколько раз повторить запрос.

    Returns:
        Тексты ответов в порядке запуска.
    """
    if runs < 1:
        raise ValueError("runs должен быть не меньше 1")

    answers = []
    for _ in range(runs):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            seed=seed,
        )
        answers.append(response.choices[0].message.content or "")
    return answers


def request_json(
    client: OpenAI,
    model_id: str,
    messages: list[Message],
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """Запрашивает ответ в JSON mode и возвращает его текст.

    JSON mode требует синтаксически корректный JSON. Набор полей он
    не проверяет: это делает ``parse_and_validate``. Пустой ответ и
    обрыв по ``max_tokens`` считаются ошибкой валидации, чтобы их
    можно было отправить в корректирующий повтор.

    Args:
        client: Клиент OpenRouter.
        model_id: Идентификатор модели.
        messages: Текущая история диалога.
        temperature: Температура генерации.
        max_tokens: Верхняя граница длины ответа.

    Returns:
        Непустой текст ответа.

    Raises:
        ResponseValidationError: ответ пустой или оборван лимитом токенов.
    """
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    choice = response.choices[0]
    content = choice.message.content

    if response.usage is not None:
        logger.info(
            (
                "API response: model=%s finish_reason=%s "
                "prompt_tokens=%d completion_tokens=%d"
            ),
            model_id,
            choice.finish_reason,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
    else:
        logger.info(
            "API response: model=%s finish_reason=%s usage=unknown",
            model_id,
            choice.finish_reason,
        )

    if choice.finish_reason == "length":
        raise ResponseValidationError(
            "Ответ оборвался из-за ограничения max_tokens"
        )
    if not content:
        raise ResponseValidationError("Модель вернула пустой ответ")
    return content


def get_json_response(
    user_request: str,
    request_fn: RequestFn,
    request_id: int | str,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[dict[str, str], int]:
    """Получает ответ, который проходит контракт, за ограниченное число попыток.

    Первая попытка отправляет исходные сообщения. Если проверка не
    прошла, в историю добавляются предыдущий ответ модели и просьба
    исправить конкретную ошибку. После последней неудачи исключение
    уходит наружу: подставлять выдуманные поля нельзя.

    Args:
        user_request: Обращение пользователя.
        request_fn: Функция, которая по истории сообщений возвращает текст.
        request_id: Идентификатор обращения в логах.
        max_attempts: Сколько раз можно спрашивать модель.

    Returns:
        Проверенный словарь и номер успешной попытки, начиная с 1.

    Raises:
        ResponseValidationError: все попытки закончились ошибкой.
        ValueError: ``max_attempts`` меньше 1.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts должен быть не меньше 1")

    messages = build_support_messages(user_request)
    last_error: ResponseValidationError | None = None

    for attempt in range(1, max_attempts + 1):
        content: str | None = None
        logger.info(
            "request_id=%s attempt=%d/%d",
            request_id,
            attempt,
            max_attempts,
        )
        try:
            content = request_fn(messages)
            data = parse_and_validate(content)
            logger.info(
                (
                    "request_id=%s validation=success "
                    "attempt=%d category=%s priority=%s"
                ),
                request_id,
                attempt,
                data["category"],
                data["priority"],
            )
            return data, attempt
        except ResponseValidationError as error:
            last_error = error
            logger.warning(
                "request_id=%s validation=failed attempt=%d reason=%s",
                request_id,
                attempt,
                error,
            )
            if attempt == max_attempts:
                break
            if content is not None:
                messages.append({"role": "assistant", "content": content})
            messages.append(
                {"role": "user", "content": correction_message(error)}
            )

    raise ResponseValidationError(
        "Не удалось получить корректный ответ " f"за {max_attempts} попытки"
    ) from last_error


def run_correction_smoke_test() -> None:
    """Проверяет корректирующие повторы на трёх заготовленных строках.

    Первая строка — сломанный JSON, вторая — JSON без обязательного поля,
    третья — корректный ответ. Тест убеждается, что перед второй и третьей
    попытками модели показывают её предыдущий текст и причину отказа.
    К API эта функция не обращается.
    """
    scripted_responses = [
        '{"category": "delivery", "priority": "high",}',
        '{"category": "delivery", "priority": "urgent"}',
        (
            '{"category": "delivery", "priority": "high", '
            '"summary": "Курьер опаздывает"}'
        ),
    ]
    response_iterator = iter(scripted_responses)
    message_histories: list[list[Message]] = []

    def scripted_request(messages: list[Message]) -> str:
        message_histories.append([message.copy() for message in messages])
        return next(response_iterator)

    data, attempt = get_json_response(
        user_request="Курьер опаздывает.",
        request_fn=scripted_request,
        request_id="smoke_test",
        max_attempts=3,
    )

    if attempt != 3:
        raise AssertionError(f"ожидалась 3-я попытка, получена {attempt}")
    if data["category"] != "delivery" or data["priority"] != "high":
        raise AssertionError(f"неверный разобранный ответ: {data}")
    if len(message_histories) != 3:
        raise AssertionError("история должна сохраниться для каждой попытки")

    second_attempt = message_histories[1]
    if second_attempt[-2]["role"] != "assistant":
        raise AssertionError("перед повтором нет ответа модели")
    if second_attempt[-2]["content"] != scripted_responses[0]:
        raise AssertionError("в историю попал не первый ответ модели")
    if second_attempt[-1]["role"] != "user":
        raise AssertionError("описание ошибки должно идти от пользователя")
    if "Ответ не является корректным JSON" not in second_attempt[-1]["content"]:
        raise AssertionError("вторая попытка не объясняет синтаксическую ошибку")

    third_attempt = message_histories[2]
    if third_attempt[-2]["content"] != scripted_responses[1]:
        raise AssertionError("в историю попал не второй ответ модели")
    if "Отсутствуют обязательные поля" not in third_attempt[-1]["content"]:
        raise AssertionError("третья попытка не объясняет пропущенное поле")

    logger.info("Локальная проверка пройдена: ответ исправлен за %d попытки", attempt)


def evaluate_reliability(
    requests: list[str],
    request_fn: RequestFn,
    max_attempts: int = MAX_ATTEMPTS,
) -> dict[str, Any]:
    """Считает, какая доля обращений получила валидный JSON.

    ``success_rate`` после попытки n включает все успехи с 1-й по n-ю
    попытку, поэтому он не уменьшается. Обращение, ушедшее в fallback,
    в эту долю не входит. ``total_generation_attempts`` считает и
    успешные, и полностью неуспешные попытки.

    Args:
        requests: Тексты обращений.
        request_fn: Функция запроса к модели или заготовленным ответам.
        max_attempts: Лимит попыток на одно обращение.

    Returns:
        Метрики: успехи по попыткам, накопительный success_rate,
        число исправлений, неуспехи и общее число генераций.
    """
    if not requests:
        raise ValueError("Список requests не должен быть пустым")

    success_by_attempt: Counter[int] = Counter()
    failed_requests: list[dict[str, Any]] = []

    for request_id, user_request in enumerate(requests, start=1):
        try:
            data, attempt = get_json_response(
                user_request=user_request,
                request_fn=request_fn,
                request_id=request_id,
                max_attempts=max_attempts,
            )
            success_by_attempt[attempt] += 1
            logger.info(
                "request_id=%d result=success attempt=%d summary=%s",
                request_id,
                attempt,
                data["summary"],
            )
        except ResponseValidationError as error:
            failed_requests.append(
                {
                    "request_id": request_id,
                    "request": user_request,
                    "error": str(error),
                }
            )
            logger.error(
                "request_id=%d result=failed attempts=%d reason=%s",
                request_id,
                max_attempts,
                error,
            )

    total_requests = len(requests)
    cumulative_successes = 0
    success_rate_by_attempt: dict[int, float] = {}

    logger.info("Накопительный success_rate")
    for attempt in range(1, max_attempts + 1):
        cumulative_successes += success_by_attempt[attempt]
        success_rate = cumulative_successes / total_requests
        success_rate_by_attempt[attempt] = success_rate
        logger.info(
            "После попытки %d: %.2f (%d/%d)",
            attempt,
            success_rate,
            cumulative_successes,
            total_requests,
        )

    recovered_after_correction = sum(
        success_by_attempt[attempt] for attempt in range(2, max_attempts + 1)
    )
    failed_count = len(failed_requests)
    final_success_rate = cumulative_successes / total_requests
    total_generation_attempts = sum(
        attempt * success_by_attempt[attempt]
        for attempt in range(1, max_attempts + 1)
    )
    total_generation_attempts += failed_count * max_attempts

    logger.info("Всего обращений: %d", total_requests)
    logger.info("Успешно с первой попытки: %d", success_by_attempt[1])
    logger.info(
        "Исправлено после корректирующих повторов: %d",
        recovered_after_correction,
    )
    logger.info("Перешло к fallback-сценарию: %d", failed_count)
    logger.info("Всего попыток генерации: %d", total_generation_attempts)
    logger.info("Итоговый success_rate: %.2f", final_success_rate)

    return {
        "total_requests": total_requests,
        "success_by_attempt": {
            attempt: success_by_attempt[attempt]
            for attempt in range(1, max_attempts + 1)
        },
        "success_rate_by_attempt": success_rate_by_attempt,
        "recovered_after_correction": recovered_after_correction,
        "failed_count": failed_count,
        "failed_requests": failed_requests,
        "total_generation_attempts": total_generation_attempts,
        "final_success_rate": final_success_rate,
    }
