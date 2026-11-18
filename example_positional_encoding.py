"""Пример: RoPE, ALiBi, YaRN и предел контекстного окна.

Внимание не различает порядок. «Мать любит дочь» и «Дочь любит мать»
состоят из тех же слов, но описывают разные события. Позиционное
кодирование сообщает модели место токена и тем самым задаёт,
насколько длинный вход она способна принять.

Дальше пример показывает четыре схемы на числах, сопоставляет их
с формулировками урока и выбирает YaRN для контракта на 60 тысяч
токенов при окне 8 тысяч и ограниченном бюджете.

Практика прячет три кодовых числа в короткий и длинный документы.
TinyLlama с окном 2048 получает оба, Qwen2.5 с широким окном — только
длинный. Попадание считается найденным, если число есть в ответе.
При первом запуске TinyLlama скачивается, это около 2 ГБ. Qwen уже
используется в примере про цену контекста.

Запуск из корня проекта тем интерпретатором, где стоят зависимости:

    python example_positional_encoding.py
"""

import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    sys.exit(
        "Для этого примера нужен torch.\n"
        "Поставьте зависимости из requirements.txt в тот же интерпретатор, "
        "которым запускаете скрипт."
    )

from transformers import AutoTokenizer

from utils.models import (
    BEACON_EXPECTED,
    METHOD_IDEAS,
    QWEN_INSTRUCT_MODEL,
    TINYLLAMA_CONTEXT_WINDOW,
    TINYLLAMA_MODEL,
    absolute_position_defined,
    alibi_penalty,
    ask_about_beacons,
    beacon_marks,
    count_text_tokens,
    make_beacon_document,
    recommend_context_extension,
    rope_relative_angle,
    rope_rotation_angle,
    yarn_scaled_position,
)

BASE_ANGLE = 1.0
TRAINED_WINDOW = 8_000
EXTENDED_WINDOW = 64_000
CONTRACT_TOKENS = 60_000
ROPE_MODEL_WINDOW = 8_000


def section(title: str) -> None:
    print("=" * 80, flush=True)
    print(title, flush=True)
    print("=" * 80, flush=True)


def main() -> None:
    section("1. Одинаковые слова в разном порядке — разный смысл")
    print("Мать любит дочь")
    print("Дочь любит мать")
    print(
        "Набор слов один и тот же. Без позиции модель не знает, "
        "кто стоит раньше, и не отличает, кто кого любит."
    )

    section("2. Абсолютные метки заканчиваются на длине обучения")
    trained_length = TINYLLAMA_CONTEXT_WINDOW
    for position in (1, trained_length, trained_length + 1):
        defined = absolute_position_defined(position, trained_length)
        state = "метка есть" if defined else "метки нет"
        print(f"позиция {position:>5}: {state}")
    print(
        "Дальше этой границы абсолютное кодирование не продолжается: "
        "лишние токены некуда посадить."
    )

    section("3. RoPE: абсолютный поворот кодирует относительное расстояние")
    print(
        "Позиция умножается на угол шага θ. "
        f"Здесь θ = {BASE_ANGLE:.0f}, чтобы разница была видна без дробей."
    )
    for position in (1, 2, 5):
        print(
            f"позиция {position}: угол {rope_rotation_angle(position, BASE_ANGLE):.0f}θ"
        )
    near = rope_relative_angle(5, 4, BASE_ANGLE)
    far = rope_relative_angle(101, 100, BASE_ANGLE)
    print(f"позиции 5 и 4 отличаются на {near:.0f}θ")
    print(f"позиции 101 и 100 отличаются на {far:.0f}θ")
    print(
        "Разница одна и та же: модель видит расстояние, а не только номер. "
        "За обучающей длиной углы становятся незнакомыми, и качество падает. "
        "Так устроены Llama, Qwen и Mistral."
    )

    section("4. ALiBi: штраф растёт вместе с расстоянием")
    print("Наклон в этом примере равен 1. У разных голов внимания он разный.")
    for distance in (1, 5, 100):
        penalty = alibi_penalty(distance, slope=1)
        print(f"расстояние {distance:>3}: штраф {penalty:.0f}")
    print(
        "Формула не хранит таблицу позиций, поэтому то же произведение "
        "считается и для 10 000 токенов. Чем дальше токен, тем слабее связь. "
        "Так устроены MPT и BLOOM."
    )

    section("5. YaRN укладывает длинную позицию в знакомый диапазон")
    scaled = yarn_scaled_position(EXTENDED_WINDOW, TRAINED_WINDOW, EXTENDED_WINDOW)
    print(
        f"Модель училась до {TRAINED_WINDOW} токенов. "
        f"Нужно принять {EXTENDED_WINDOW}."
    )
    print(
        f"Позиция {EXTENDED_WINDOW} после масштабирования становится {scaled:.0f}: "
        "угол остаётся в диапазоне обучения."
    )
    print(
        "Полный YaRN сильнее сжимает медленные пары вектора, почти не трогает "
        "быстрые и затем донастраивает модель на длинных примерах. "
        "Это дешевле, чем учить модель с нуля."
    )

    section("6. Какая идея к какому подходу относится")
    print(
        "Фраза «за пределом обучающей длины дополнительным токенам не хватает "
        "позиционных меток» относится к абсолютному кодированию. "
        "У RoPE метки не кончаются: кончается знакомый диапазон углов."
    )
    for method, ideas in METHOD_IDEAS:
        print(f"\n{method}")
        for idea in ideas:
            print(f"  — {idea}")

    section("7. Контракт на 60 тысяч токенов и окно 8 тысяч")
    print(
        "Есть проверенная RoPE-модель с окном 8000. "
        "Полное обучение на 60 тысячах токенов бюджет не позволяет."
    )
    choice = recommend_context_extension(
        document_tokens=CONTRACT_TOKENS,
        model_window=ROPE_MODEL_WINDOW,
        uses_rope=True,
        can_fully_retrain=False,
    )
    print("Рекомендация:", choice)
    print(
        "Передать 60 тысяч в окно 8000 нельзя: хвост документа не будет "
        "обработан корректно. Обрезать контракт до 8000 тоже нельзя: "
        "юридически важные условия окажутся за бортом. "
        "Обучение с нуля решает задачу, но выходит за бюджет."
    )

    section("8. Маячки: окно модели против длины документа")
    print(
        f"Длину отмеряет токенизатор {TINYLLAMA_MODEL}. "
        f"Его окно — {TINYLLAMA_CONTEXT_WINDOW} токенов. "
        "Короткий документ должен в него войти, длинный — выйти."
    )
    tokenizer = AutoTokenizer.from_pretrained(TINYLLAMA_MODEL)
    document_short = make_beacon_document(tokenizer, 600)
    document_long = make_beacon_document(tokenizer, 1400)
    short_tokens = count_text_tokens(tokenizer, document_short)
    long_tokens = count_text_tokens(tokenizer, document_long)
    print(f"Короткий документ: {short_tokens} токенов")
    print(f"Длинный документ:  {long_tokens} токенов")
    print("Маячки:", BEACON_EXPECTED)

    scenarios = [
        ("TinyLlama, короткий документ", TINYLLAMA_MODEL, document_short, short_tokens),
        ("TinyLlama, длинный документ", TINYLLAMA_MODEL, document_long, long_tokens),
        ("Qwen2.5, длинный документ", QWEN_INSTRUCT_MODEL, document_long, long_tokens),
    ]

    print(flush=True)
    print(f"{'сценарий':<32}{'АЛЬФА':<8}{'БЕТА':<8}{'ГАММА':<8}")
    results = []
    for title, model_name, document, document_tokens in scenarios:
        print(f"\nЗапуск: {title}", flush=True)
        result = ask_about_beacons(model_name, document)
        results.append((title, document_tokens, result))
        marks = beacon_marks(result["answers"])
        for name in ("АЛЬФА", "БЕТА", "ГАММА"):
            marks.setdefault(name, "нет")
        print(
            f"{title:<32}{marks['АЛЬФА']:<8}{marks['БЕТА']:<8}{marks['ГАММА']:<8}",
            flush=True,
        )
        print(
            f"окно модели {result['context_window']}, "
            f"документ {document_tokens} токенов, устройство {result['device']}"
        )
        for answer in result["answers"]:
            print(f"  {answer.name}: ожидалось {answer.expected}, ответ: {answer.answer}")
        if result["error"]:
            print("Генерация остановилась:", result["error"])

    small_short = results[0][2]["answers"]
    small_long = results[1][2]["answers"]
    large_long = results[2][2]["answers"]
    print()
    print(
        "TinyLlama на коротком документе нашла "
        f"{sum(item.found for item in small_short)} из {len(small_short)}, "
        "на длинном — "
        f"{sum(item.found for item in small_long)} из {len(small_long)}."
    )
    print(
        "Qwen2.5 на том же длинном документе нашла "
        f"{sum(item.found for item in large_long)} из {len(large_long)}."
    )
    print(
        "Если документ длиннее окна, модель либо не принимает хвост, "
        "либо резко теряет маячки. Широкое окно даёт документу поместиться. "
        "Этого мало: модель ещё должна доставать факт из начала, середины "
        "и конца, а не только формально принимать длину. "
        "RoPE надёжен внутри обучающего окна, ALiBi лучше выходит за него "
        "без дообучения, YaRN расширяет уже готовую RoPE-модель."
    )


if __name__ == "__main__":
    main()
