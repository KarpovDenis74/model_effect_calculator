"""Пример: top-k, top-p, температура и штрафы за повтор.

После forward pass у модели есть логиты следующего токена. Дальше их
обрабатывают отдельными функциями, и только потом выбирают токен.

Порядок такой:

    логиты
      → apply_repetition_penalty или apply_frequency_penalty
      → apply_temperature
      → apply_top_k
      → apply_top_p
      → choose_greedy_token или sample_tokens

Температура меняет остроту распределения и не выбрасывает токены.
top-k оставляет фиксированное число лучших. top-p оставляет минимальный
набор, чья суммарная вероятность не меньше порога. repetition_penalty
один раз снижает логит уже встреченного id. frequency_penalty вычитает
штраф за каждое предыдущее появление.

Сами функции лежат в ``utils.models``. Этот файл только запускает разбор
урока: таблицы на заданных логитах и прогон gpt2. Модель английская,
поэтому промпт тоже английский — на нём виден цикл из урока. При первом
запуске веса скачиваются с Hugging Face.

Запуск из корня проекта:

    python example_decoding_controls.py
"""

import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    sys.exit(
        "Для этого примера нужен torch.\n"
        "Поставьте зависимости из requirements.txt."
    )

from utils.models import (
    demo_cases,
    demo_flat_cap,
    demo_frequency_penalty,
    demo_gpt2,
    demo_penalty_symptoms,
    demo_pipeline,
    demo_repetition_penalty,
    demo_temperature,
    demo_top_k,
    demo_top_p,
    demo_which_parameter,
)


def main() -> None:
    print("Функции урока вызываются по отдельности.")
    print("Температура готовит распределение, фильтры режут кандидатов,")
    print("штрафы уменьшают оценки уже встреченных токенов.")
    print()

    demo_top_k()
    demo_top_p()
    demo_temperature()
    demo_flat_cap()
    demo_which_parameter()
    demo_repetition_penalty()
    demo_frequency_penalty()
    demo_penalty_symptoms()
    demo_pipeline()
    demo_cases()
    demo_gpt2()


if __name__ == "__main__":
    main()
