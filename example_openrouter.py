"""Пример: запросы к моделям через OpenRouter.

Один клиент OpenAI SDK отправляет Chat Completions не в OpenAI, а на
``https://openrouter.ai/api/v1``. Ключ берётся из ``OPENROUTER_API_KEY``
и в вывод не попадает. Если переменной нет, читается локальный ``.env``.

Дальше пример показывает четыре части урока.

1. Сообщение ``system`` задаёт правила, ``user`` — сам вопрос.
   ``temperature`` и ``max_tokens`` управляют генерацией.
   Маршрутизатор ``openrouter/free`` сам выбирает бесплатную модель,
   поэтому для сравнения он не используется.
2. Из ответа читаются текст, идентификатор модели, ``finish_reason``
   и счётчики токенов. ``length`` значит, что ответ упёрся в лимит.
   Условная цена считается тарифами 1$ и 4$ за миллион входных и
   выходных токенов. Это учебные числа, не прайс выбранной модели.
3. Перед экспериментом каталог сообщает, поддерживает ли модель
   ``temperature``, ``max_tokens`` и ``seed``.
4. Временные ошибки 429, тайм-аут, обрыв связи и 5xx повторяются
   с растущей паузой. Ошибки запроса и доступа не повторяются.
   Две бесплатные модели разных разработчиков получают одни и те же
   три промпта. Сравниваются текст, число токенов и учебная стоимость.

Бесплатные модели могут отвечать медленно, поэтому клиент в примере
ждёт один запрос до 90 секунд. В уроке для иллюстрации указано 30.

Запуск из корня проекта:

    python example_openrouter.py
"""

import importlib.util
import logging
import sys

if importlib.util.find_spec("openai") is None:
    sys.exit(
        "Для этого примера нужен пакет openai.\n"
        "Поставьте зависимости из requirements.txt."
    )

from utils.openrouter import (
    calculate_retry_delay,
    check_supported_parameters,
    compare_models,
    create_chat_completion_with_retry,
    create_openrouter_client,
    get_model_info,
    get_free_text_models,
    require_openrouter_api_key,
    select_two_models,
    summarize_comparison,
    token_usage,
    usage_is_consistent,
    warn_if_truncated,
)


def section(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    section("1. Ключ и клиент")
    print(
        "os.getenv читает OPENROUTER_API_KEY. В консоль попадает только "
        "факт, что значение есть. base_url отправляет запросы в OpenRouter. "
        "max_retries=0 отключает повторы библиотеки: их делает "
        "create_chat_completion_with_retry."
    )
    api_key = require_openrouter_api_key(".env")
    print("Ключ OpenRouter найден")
    client = create_openrouter_client(api_key, timeout=90.0, max_retries=0)
    print("Адрес API:", client.base_url)
    print()

    section("2. Пауза между повторами")
    print(
        "После временной ошибки ждут base_delay * 2 ** (попытка - 1) "
        "плюс случайный jitter, но не дольше max_delay. "
        "400, 401 и 403 так не повторяют."
    )
    for attempt in range(1, 5):
        delay = calculate_retry_delay(attempt)
        print(f"Попытка {attempt}: ожидание {delay:.2f} с")
    print()

    section("3. Первый запрос: роли, температура и лимит токенов")
    print(
        "system задаёт стиль на весь диалог. user содержит вопрос. "
        "Порядок в messages — это порядок реплик. temperature=0.2 делает "
        "выбор токенов спокойнее. max_tokens=150 ограничивает длину, "
        "но модель может закончить раньше."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Ты объясняешь технические термины простыми словами. "
                "Отвечай не более чем в двух предложениях."
            ),
        },
        {
            "role": "user",
            "content": "Что такое API?",
        },
    ]
    response = create_chat_completion_with_retry(
        client=client,
        model="openrouter/free",
        messages=messages,
        temperature=0.2,
        max_tokens=150,
    )
    answer = response.choices[0].message.content or ""
    finish_reason = response.choices[0].finish_reason
    usage = token_usage(response)

    print()
    print("Ответ:")
    print(answer)
    print()
    print("Использованная модель:", response.model)
    print("Причина завершения:", finish_reason)
    print("Входные токены:", usage["prompt_tokens"])
    print("Выходные токены:", usage["completion_tokens"])
    print("Всего токенов:", usage["total_tokens"])
    print("Сумма входа и выхода сходится:", usage_is_consistent(response))
    warn_if_truncated(finish_reason)
    print(
        "Входные токены считают весь messages, включая system. "
        "Следующий ход диалога нужно снова положить в этот список: "
        "API сам историю не хранит."
    )
    print()

    section("4. Какие параметры принимает выбранная модель")
    print(
        "response.model — конкретная модель, которую выбрал openrouter/free. "
        "Карточка каталога показывает supported_parameters. Код ниже "
        "проверяет три параметра, которые нужны эксперименту."
    )
    model_info = get_model_info(api_key, response.model)
    check_supported_parameters(
        model_info=model_info,
        required_parameters=["temperature", "max_tokens", "seed"],
    )
    print()

    section("5. Две модели на одних и тех же трёх задачах")
    print(
        "Для сравнения берётся фиксированный id, а не openrouter/free. "
        "Иначе каждый запрос может уйти в другую модель. "
        "select_two_models оставляет двух разных разработчиков."
    )
    free_model_ids = get_free_text_models(api_key)
    model_ids = select_two_models(free_model_ids)
    print("Выбранные модели:")
    for model_id in model_ids:
        print(f"- {model_id}")

    comparison = compare_models(client=client, model_ids=model_ids)
    summary = summarize_comparison(comparison)

    print()
    print("=" * 80)
    print("Итоговая статистика по моделям")
    print("=" * 80)
    print(summary.to_string(index=False))
    print()
    print(
        "Учебный тариф: 1$ за миллион входных токенов и 4$ за миллион "
        "выходных. Больше выходных токенов обычно сильнее увеличивает цену, "
        "чем более длинный промпт. Число токенов показывает расход, "
        "а не то, какая модель точнее выполнила требование."
    )


if __name__ == "__main__":
    main()
