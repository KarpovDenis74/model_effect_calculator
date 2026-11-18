"""Одинаковые промпты на двух моделях: ответы, токены и учебная цена."""

from __future__ import annotations

import pandas as pd
from openai import OpenAI

from utils.openrouter.client import create_chat_completion_with_retry
from utils.openrouter.usage import calculate_estimated_cost

SYSTEM_PROMPT = (
    "Точно выполняй требования пользователя. "
    "Не добавляй информацию, о которой не просили. "
    "Соблюдай указанный формат ответа."
)

COMPARISON_PROMPTS = [
    {
        "prompt_id": "extraction",
        "prompt": (
            "Извлеки номер заказа, город доставки и дату из сообщения: "
            "'Заказ 18452 нужно доставить в Казань 17 августа'. "
            "Верни только JSON с полями order_id, city и delivery_date."
        ),
    },
    {
        "prompt_id": "summary",
        "prompt": (
            "Сократи текст до двух предложений, сохранив все основные "
            "результаты:\n\n"
            "Команда добавила к сервису кэширование ответов. "
            "После этого среднее время ответа уменьшилось с 1.8 до 0.7 "
            "секунды. Доля ошибок не изменилась, а нагрузка на основную "
            "базу данных сократилась на 35 процентов."
        ),
    },
    {
        "prompt_id": "code",
        "prompt": (
            "Найди ошибку в функции и верни исправленный код:\n\n"
            "def average(values):\n"
            "    return sum(values) / len(value)\n\n"
            "После кода добавь одно предложение с объяснением ошибки."
        ),
    },
]


def print_response_result(
    prompt_id: str,
    model_id: str,
    answer: str,
    prompt_tokens: int,
    completion_tokens: int,
    estimated_cost: float,
    finish_reason: str | None = None,
) -> None:
    """Печатает ответ модели и статистику одного запроса.

    Args:
        prompt_id: Короткое имя задачи.
        model_id: Идентификатор модели.
        answer: Текст ответа.
        prompt_tokens: Входные токены.
        completion_tokens: Выходные токены.
        estimated_cost: Условная стоимость в долларах.
        finish_reason: Почему генерация остановилась. ``length`` значит,
            что ответ упёрся в ``max_tokens``.
    """
    print()
    print("=" * 80)
    print(f"Задача: {prompt_id}")
    print(f"Модель: {model_id}")
    print("-" * 80)
    print(answer)
    print("-" * 80)
    if finish_reason is not None:
        print(f"Причина завершения: {finish_reason}")
    print(f"Входные токены: {prompt_tokens}")
    print(f"Выходные токены: {completion_tokens}")
    print(f"Расчётная стоимость: ${estimated_cost:.8f}")


def compare_models(
    client: OpenAI,
    model_ids: list[str],
    prompts: list[dict[str, str]] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.2,
    max_tokens: int = 220,
) -> pd.DataFrame:
    """Запускает каждую модель на одних и тех же промптах.

    История между задачами не переносится: у каждого запроса свой список
    из системной инструкции и одного сообщения пользователя. Так модели
    сравниваются на одинаковом входе.

    Args:
        client: Клиент OpenRouter.
        model_ids: Какие модели вызвать.
        prompts: Список словарей с ``prompt_id`` и ``prompt``.
        system_prompt: Общая инструкция для всех запросов.
        temperature: Температура генерации.
        max_tokens: Лимит длины ответа.

    Returns:
        Таблицу с ответом, токенами и условной стоимостью каждой пары.
    """
    tasks = COMPARISON_PROMPTS if prompts is None else prompts
    records = []

    for prompt_data in tasks:
        for model_id in model_ids:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_data["prompt"]},
            ]
            response = create_chat_completion_with_retry(
                client=client,
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if response.usage is None:
                raise RuntimeError(f"Модель {model_id} не вернула usage")

            answer = response.choices[0].message.content or ""
            estimated_cost = calculate_estimated_cost(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
            print_response_result(
                prompt_id=prompt_data["prompt_id"],
                model_id=model_id,
                answer=answer,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                estimated_cost=estimated_cost,
                finish_reason=response.choices[0].finish_reason,
            )
            records.append(
                {
                    "prompt_id": prompt_data["prompt_id"],
                    "model": model_id,
                    "answer": answer,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "estimated_cost_usd": estimated_cost,
                }
            )

    return pd.DataFrame(records)


def summarize_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Суммирует токены и учебную стоимость по каждой модели.

    Args:
        comparison: Таблица из ``compare_models``.

    Returns:
        По одной строке на модель.
    """
    return comparison.groupby("model", as_index=False).agg(
        requests=("prompt_id", "count"),
        prompt_tokens=("prompt_tokens", "sum"),
        completion_tokens=("completion_tokens", "sum"),
        estimated_cost_usd=("estimated_cost_usd", "sum"),
    )
