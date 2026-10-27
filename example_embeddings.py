"""Пример: таблица входных эмбеддингов и ближайшие токены.

После токенизации у каждого фрагмента есть числовой id. Этот id — номер строки
в таблице эмбеддингов. Строка — вещественный вектор фиксированной длины, который
модель выучила вместе с остальными весами. Отдельное число в векторе обычно
нечего «перевести на русский»: модель использует вектор целиком.

Входной эмбеддинг ещё не знает предложение. Один и тот же token_id всегда
достаёт одну и ту же строку, даже если слово в двух фразах значит разное.
«Замок» на холме и «замок» на двери совпадут. Различать значения модель
начинает позже, когда эти векторы проходят через слои трансформера.

Таблица при этом уже устроена как пространство. Если привести векторы к длине 1,
скалярное произведение становится косинусным сходством: оно сравнивает
направления. Ближайшие соседи часто оказываются формами того же слова или
словами одной темы, но не словарём синонимов. Сосед может попасть рядом просто
потому, что модель видела эти токены в похожих контекстах.

Модель в примере — ``cointegrated/rubert-tiny2``. Она небольшая и русскоязычная,
поэтому обычные русские слова чаще остаются одним токеном. Размер таблицы у неё
``[83828, 312]``: 83828 строк словаря и вектор длины 312.

Запуск из корня проекта:

    python example_embeddings.py
"""

import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    sys.exit(
        "Для этого примера нужен torch: таблица эмбеддингов — это веса модели.\n"
        "Поставьте зависимости из requirements.txt: на этом Intel Mac это torch==2.2.2."
    )

from utils.embeddings import (
    find_nearest_tokens,
    get_embedding_matrix,
    get_single_token_id,
    get_token_embedding,
    same_input_embedding,
    tokenize_without_special,
)

CANDIDATE_TOKENS = [
    "улица",
    "город",
    "река",
    "король",
    "музыка",
    "школа",
    "машина",
    "книга",
]


def main() -> None:
    print("1. Таблица эмбеддингов: строка на токен, длина вектора одинаковая")
    matrix = get_embedding_matrix()
    print(tuple(matrix.shape))

    print("\n2. Слово из одного токена — одна строка таблицы")
    word = "замок"
    tokens, token_ids = tokenize_without_special(word)
    token_id = get_single_token_id(word)
    embedding = get_token_embedding(word)
    print(tokens, token_ids)
    print(word, token_id, tuple(embedding.shape))
    print(embedding[:10])

    print("\n3. Тот же token_id в разных предложениях даёт тот же вектор")
    texts = [
        "Старый замок стоял на холме.",
        "Дверной замок сломался.",
    ]
    for text in texts:
        text_tokens, text_ids = tokenize_without_special(text)
        print(text)
        print(text_tokens)
        print(text_ids)
        print()
    print("векторы совпали:", same_input_embedding("замок", texts[0], texts[1]))

    print("\n4. Ближайшие к «замок»: тема рядом, но это ещё не синонимы")
    for token, score in find_nearest_tokens("замок", top_k=10):
        print(token, score)

    print("\n5. Ищем соседа, который не является формой слова или синонимом")
    for candidate in CANDIDATE_TOKENS:
        print(f"\n{candidate}")
        try:
            nearest = find_nearest_tokens(candidate, top_k=10)
        except ValueError as error:
            print(error)
            continue
        for token, score in nearest:
            print(token, score)

    # Исследованный токен: улица.
    # Ближайшие токены на rubert-tiny2 обычно такие: улицы, улиц, улице,
    # дорога, Улица, шоссе, улицами, бульвар, река, переулок.
    # Формы того же слова: улицы, улиц, улице, улицами, Улица.
    # Близкая городская тема: дорога, шоссе, бульвар, переулок.
    # Неочевидный сосед: река. Это не синоним улицы.
    # Сблизить их могли похожие контексты описания места: и улица, и река
    # «проходят через город», бывают «вдоль» чего-то и встречаются в одних
    # и тех же географических фразах. До слоёв трансформера модель этого
    # различия ещё не делает.
    street_neighbors = find_nearest_tokens("улица", top_k=10)
    print("\nВыбранный пример — улица:")
    for token, score in street_neighbors:
        print(token, score)


if __name__ == "__main__":
    main()
