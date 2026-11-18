"""Пример: seed, контракт JSON и техническая надёжность ответа.

Урок разбирает, как сравнивать инструкции модели и как принимать
её ответ в сервис. Пример вызывает функции из ``utils.openrouter``.

1. Одинаковый seed, модель, сообщения и температура уменьшают
   случайный разброс. Совпадение текстов не гарантировано.
2. Системный промпт описывает поля. JSON mode требует корректный
   JSON, но не проверяет названия и значения полей.
3. ``parse_and_validate`` и Pydantic-модель ``SupportRequest``
   проверяют контракт уже после ответа модели.
4. Корректирующий повтор показывает модели её предыдущий ответ
   и причину отказа. Это не технический retry и не fallback.
5. ``evaluate_reliability`` считает накопительный success_rate.
   Передача обращения оператору в эту долю не входит.

Сначала всё проверяется на заготовленных строках, без API.
Потом два запроса с ``seed=42`` и разбор восьми обращений
через бесплатную модель OpenRouter.

Запуск из корня проекта:

    python example_json_reliability.py
"""

import importlib.util
import logging
import sys
from functools import partial

from pydantic import ValidationError

if importlib.util.find_spec("openai") is None:
    sys.exit(
        "Для этого примера нужен пакет openai.\n"
        "Поставьте зависимости из requirements.txt."
    )

from utils.openrouter import (
    SUPPORT_TEST_REQUESTS,
    ResponseValidationError,
    SupportRequest,
    collect_seeded_answers,
    create_openrouter_client,
    evaluate_reliability,
    get_free_json_models,
    parse_and_validate,
    request_json,
    require_openrouter_api_key,
    run_correction_smoke_test,
    select_json_model,
    select_seed_model,
    structured_response_format,
    support_request_schema,
    validate_support_json,
)

VALID_JSON = (
    '{"category": "delivery", "priority": "high", '
    '"summary": "Курьер опаздывает"}'
)
SEED_MESSAGES = [
    {
        "role": "system",
        "content": (
            "Определи категорию обращения пользователя. "
            "Выбери одно значение: delivery, payment, "
            "product, account или other. "
            "Верни только название категории."
        ),
    },
    {
        "role": "user",
        "content": (
            "Заказ должен был приехать вчера, "
            "но курьер так и не появился."
        ),
    },
]


def section(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def show_validation(title: str, content: str) -> None:
    print(f"\n{title}")
    print(content.strip())
    try:
        print("контракт пройден:", parse_and_validate(content))
    except ResponseValidationError as error:
        print("контракт не пройден:", error)


def lesson_outcome_pattern() -> list[str]:
    """Повторяет разбор из урока: 14, 4, 1 успех и 1 полный отказ.

    ``ok`` — валидный JSON. ``bad`` — JSON с недопустимым приоритетом.
    Число элементов равно числу вызовов модели, а не числу обращений.
    """
    return (
        ["ok"] * 14
        + ["bad", "ok"] * 4
        + ["bad", "bad", "ok"]
        + ["bad", "bad", "bad"]
    )


def scripted_request_fn(pattern: list[str]):
    """Подменяет модель очередью заранее известных ответов."""
    queue = iter(pattern)

    def request_fn(messages: list[dict[str, str]]) -> str:
        del messages
        if next(queue) == "ok":
            return VALID_JSON
        return (
            '{"category": "delivery", "priority": "urgent", '
            '"summary": "Заказ не доставлен в срок"}'
        )

    return request_fn


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    section("1. Контракт: синтаксис JSON ещё не значит, что ответ можно использовать")
    print(
        "Сервис ждёт ровно три поля. category выбирает очередь, "
        "priority — срочность, summary видит оператор. "
        "Допустимые категории: delivery, payment, product, account, other. "
        "Допустимые приоритеты: low, medium, high. "
        "summary — непустая строка не длиннее 120 символов."
    )
    show_validation("Корректный ответ", VALID_JSON)
    show_validation(
        "Лишняя запятая ломает синтаксис",
        '{"category": "delivery", "priority": "high",}',
    )
    show_validation(
        "urgent нет среди приоритетов",
        (
            '{"category": "delivery", "priority": "urgent", '
            '"summary": "Заказ не доставлен в срок"}'
        ),
    )
    show_validation(
        "Нет обязательного summary",
        '{"category": "delivery", "priority": "high"}',
    )

    section("2. Тот же контракт в Pydantic и JSON Schema")
    print(
        "Literal ограничивает категории и приоритеты. "
        "extra=forbid запрещает лишние поля. "
        "Field и field_validator ограничивают summary."
    )
    accepted = SupportRequest(
        category="delivery",
        priority="high",
        summary="  Заказ не доставлен в срок  ",
    )
    print("после проверки:", accepted.model_dump())
    try:
        SupportRequest(
            category="delivery",
            priority="urgent",
            summary="Заказ не доставлен в срок",
        )
    except ValidationError as error:
        print("urgent отклонён:", error.error_count(), "ошибка")

    schema = support_request_schema()
    print("обязательные поля схемы:", schema["required"])
    print("category:", schema["properties"]["category"]["enum"])
    print("priority:", schema["properties"]["priority"]["enum"])
    print(
        "Structured Outputs передаёт эту схему в response_format "
        "с strict=True. Тип запроса:",
        structured_response_format()["type"],
    )
    print(
        "JSON mode и схема отвечают за форму ответа. "
        "validate_support_json всё равно проверяет результат локально:"
    )
    print(validate_support_json(VALID_JSON).model_dump())

    section("3. Корректирующий повтор на заготовленных ответах")
    print(
        "Первая строка — сломанный JSON. Вторая — объект без summary. "
        "Третья — корректный ответ. Перед каждой новой попыткой в историю "
        "попадают предыдущий текст модели и описание ошибки. "
        "К OpenRouter этот шаг не обращается."
    )
    run_correction_smoke_test()
    print("Локальная проверка корректирующих повторов пройдена.")

    section("4. Накопительный success_rate на числах из урока")
    print(
        "Из 20 обращений 14 проходят сразу, 4 — со второй попытки, "
        "1 — с третьей, 1 остаётся неуспешным. "
        "success_rate после попытки включает все предыдущие успехи. "
        "Fallback не добавляет обращение в эту долю."
    )
    metrics = evaluate_reliability(
        requests=[f"обращение {index}" for index in range(1, 21)],
        request_fn=scripted_request_fn(lesson_outcome_pattern()),
        max_attempts=3,
    )
    print("success_by_attempt:", metrics["success_by_attempt"])
    print(
        "success_rate по попыткам:",
        {
            attempt: f"{rate:.2f}"
            for attempt, rate in metrics["success_rate_by_attempt"].items()
        },
    )
    print("исправлено повторами:", metrics["recovered_after_correction"])
    print("ушло в fallback:", metrics["failed_count"])
    print("всего генераций:", metrics["total_generation_attempts"])
    print("итоговый success_rate:", f"{metrics['final_success_rate']:.2f}")
    expected_rates = {1: 0.70, 2: 0.90, 3: 0.95}
    for attempt, expected in expected_rates.items():
        actual = metrics["success_rate_by_attempt"][attempt]
        if abs(actual - expected) > 1e-9:
            raise AssertionError(
                f"после попытки {attempt} ожидалось {expected}, получено {actual}"
            )
    if metrics["recovered_after_correction"] != 5:
        raise AssertionError("исправленных ответов должно быть 5")
    if metrics["total_generation_attempts"] != 28:
        raise AssertionError("всего генераций должно быть 28")
    if metrics["failed_count"] != 1:
        raise AssertionError("неуспешным должно остаться одно обращение")

    section("5. Один и тот же запрос с seed=42")
    print(
        "Сравнивать две инструкции можно, только если модель, сообщения, "
        "температура, seed и остальные параметры совпадают. "
        "Два запуска ниже меняют только случайность генерации, "
        "и seed фиксирует её начальное состояние."
    )
    api_key = require_openrouter_api_key(".env")
    client = create_openrouter_client(
        api_key,
        timeout=90.0,
        max_retries=2,
    )
    seed_model = select_seed_model(api_key)
    print("модель с поддержкой seed:", seed_model)
    answers = collect_seeded_answers(
        client=client,
        model=seed_model,
        messages=SEED_MESSAGES,
        seed=42,
        temperature=0.7,
        runs=2,
    )
    print("ответ 1:", answers[0])
    print("ответ 2:", answers[1])
    print("тексты совпали:", answers[0] == answers[1])
    print(
        "Если тексты различаются, это не ошибка эксперимента. "
        "Провайдер может обновить модель или считать недетерминированно."
    )

    section("6. Восемь обращений через JSON mode")
    print(
        "Каждое обращение получает не больше трёх попыток. "
        "Пустой ответ и обрыв по max_tokens тоже идут в корректирующий повтор. "
        "После третьей ошибки обращение считается неуспешным: "
        "код не подставляет категорию и summary сам."
    )
    json_models = get_free_json_models(api_key)
    model_id = select_json_model(json_models)
    print("модель JSON mode:", model_id)
    print("обращений:", len(SUPPORT_TEST_REQUESTS))
    live_metrics = evaluate_reliability(
        requests=SUPPORT_TEST_REQUESTS,
        request_fn=partial(request_json, client, model_id),
        max_attempts=3,
    )
    print("успехи по попыткам:", live_metrics["success_by_attempt"])
    print(
        "накопительный success_rate:",
        {
            attempt: f"{rate:.2f}"
            for attempt, rate in live_metrics["success_rate_by_attempt"].items()
        },
    )
    print("исправлено повторами:", live_metrics["recovered_after_correction"])
    print("неуспешных:", live_metrics["failed_count"])
    print("всего генераций:", live_metrics["total_generation_attempts"])
    print(
        "Итоговый success_rate:",
        f"{live_metrics['final_success_rate']:.2f}",
    )
    print(
        "Эта доля показывает только техническую пригодность JSON. "
        "Правильность категории и приоритета по смыслу обращения "
        "нужно проверять отдельно, на примерах с известным ответом."
    )
    if live_metrics["failed_requests"]:
        print("Неуспешные обращения:")
        for item in live_metrics["failed_requests"]:
            print(f"  #{item['request_id']}: {item['error']}")


if __name__ == "__main__":
    main()
