"""Пример: один шаг генерации и ручной цикл greedy / sampling.

Модель не пишет ответ целиком. На каждом шаге она получает текущий контекст,
делает forward pass и возвращает logits по всему словарю. Softmax превращает
эти сырые оценки в вероятности. Затем выбирается один токен и дописывается
к входу. Следующий шаг уже работает с обновлённым текстом.

Цепочка одного шага:

    текст → tokenizer → input_ids → model → logits последней позиции
         → softmax → probs → выбор следующего токена

``logits`` — сырые оценки. Они могут быть отрицательными и больше единицы,
их сумма ничему не равна. ``probs`` после softmax лежат от 0 до 1 и в сумме
дают 1. Softmax сохраняет порядок: самый большой логит станет самой большой
вероятностью. Поэтому greedy можно брать и по логитам, и по вероятностям.

Две базовые стратегии:

- greedy — всегда берёт максимум. На одном промпте результат стабилен, но
  уже выбранные токены могут закрепить повторяющийся шаблон.
- sampling — выбирает токен случайно с учётом вероятностей. Текст становится
  разнообразнее и менее предсказуемым. Готовый ``generate(do_sample=True)``
  по умолчанию ещё и режет словарь top-k=50; для сравнения с ручным
  sampling из полного распределения ставят ``top_k=0``.

Длина ответа влияет на latency напрямую: 20 новых токенов — примерно
20 шагов, 200 — примерно 200. Поэтому API ограничивают ``max_new_tokens``.

В примере используется ``gpt2``. Она небольшая и обучена в основном на
английском, поэтому на русских промптах продолжения могут быть слабее.
Смысл разбора шага генерации от этого не меняется.

Запуск из корня проекта:

    python example_generation.py
"""

import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    sys.exit(
        "Для этого примера нужен torch: модель gpt2 хранит веса в нём.\n"
        "Поставьте зависимости из requirements.txt: на этом Intel Mac это torch==2.2.2."
    )

from utils.models import (
    build_greedy_trace,
    diagnose_generation_checks,
    generate_greedy_with_trace,
    generate_sample_with_full_distribution,
    generate_with_strategy,
    load_causal_model,
    show_next_token_candidates,
)

PROMPT_CAPITALS = (
    "Столица Италии — Рим. "
    "Столица Германии — Берлин. "
    "Столица Франции —"
)
PROMPT_REPEAT = "Это очень очень"
REPEATED_PHRASE = "я не буду"


def main() -> None:
    print("1. Один шаг: top-5 следующих токенов после промпта")
    tokenizer, model = load_causal_model("gpt2")
    show_next_token_candidates(
        model=model,
        tokenizer=tokenizer,
        prompt=PROMPT_CAPITALS,
        top_k=5,
    )
    for prompt in (
        "Машинное обучение — это",
        "Сегодня я хочу",
        "Лучший способ выучить Python —",
    ):
        print()
        show_next_token_candidates(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            top_k=5,
        )

    print("\n2. Ручной greedy-цикл с печатью кандидатов на каждом шаге")
    generated = generate_greedy_with_trace(
        model=model,
        tokenizer=tokenizer,
        prompt="Жил-был",
        max_new_tokens=8,
        top_k=5,
    )
    print("\nИтог:", generated)

    print("\n3. Greedy может закрепить повтор, если шаблон уже попал в контекст")
    repeated = generate_greedy_with_trace(
        model=model,
        tokenizer=tokenizer,
        prompt=PROMPT_REPEAT,
        max_new_tokens=35,
        top_k=5,
    )
    print("\nИтог:", repeated)

    print("\n4. Тот же промпт: greedy стабилен, sampling даёт разные тексты")
    for run in range(3):
        text = generate_with_strategy(
            model=model,
            tokenizer=tokenizer,
            prompt=PROMPT_REPEAT,
            max_new_tokens=35,
            strategy="greedy",
        )
        print(f"\nGreedy запуск {run + 1}")
        print(text)

    for seed in (1, 2, 3):
        text = generate_with_strategy(
            model=model,
            tokenizer=tokenizer,
            prompt=PROMPT_REPEAT,
            max_new_tokens=35,
            strategy="sample",
            seed=seed,
        )
        print(f"\nSample seed {seed}")
        print(text)

    print("\n5. Диагностический отчёт: таблица greedy и сравнение с sampling")
    greedy_trace, greedy_text = build_greedy_trace(
        model=model,
        tokenizer=tokenizer,
        prompt=PROMPT_REPEAT,
        max_new_tokens=35,
        top_k=5,
    )
    print(greedy_trace.to_string(index=False))
    print("\nGreedy text:")
    print(greedy_text)

    sample_text = generate_sample_with_full_distribution(
        model=model,
        tokenizer=tokenizer,
        prompt=PROMPT_REPEAT,
        max_new_tokens=35,
        seed=42,
    )
    print("\nSample text:")
    print(sample_text)

    checks = diagnose_generation_checks(
        greedy_trace,
        greedy_text,
        sample_text,
        repeated_phrase=REPEATED_PHRASE,
    )
    print("\nПроверки:")
    for name, value in checks.items():
        print(f"{name}: {bool(value)}")


if __name__ == "__main__":
    main()
