"""Разбор урока про top-k, top-p и штрафы за повтор.

Функции печатают учебные таблицы и, в конце, прогон gpt2. Механизмы,
которыми они пользуются, лежат в ``decoding``: фильтры, температура и штрафы.
"""

from __future__ import annotations

import pandas as pd
import torch

from utils.models.causal import load_causal_model
from utils.models.decoding import (
    apply_frequency_penalty,
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    choose_greedy_token,
    describe_logits,
    diagnose_decoding_controls,
    generate_greedy_text,
    kept_names,
    kept_probability_mass,
    kept_token_ids,
    logits_from_probabilities,
    probability_rows,
)

TRANSPORT_TOKENS = [
    "поезд",
    "автобус",
    "самолёт",
    "велосипед",
    "корабль",
    "карандаш",
]
TRANSPORT_PROBS = [0.35, 0.25, 0.18, 0.10, 0.07, 0.05]

PEAKED_TOKENS = ["Париж", "Лондон", "Берлин", "Рим", "дерево", "ложка"]
PEAKED_PROBS = [0.70, 0.15, 0.08, 0.04, 0.02, 0.01]

FLAT_TOKENS = [
    "Париж",
    "Лондон",
    "Берлин",
    "Рим",
    "Мадрид",
    "Вена",
    "Прага",
    "Осло",
    "остальные",
]
FLAT_PROBS = [0.18, 0.16, 0.14, 0.12, 0.11, 0.10, 0.08, 0.06, 0.05]

CAT_TOKENS = ["кот", "он", "прыгнул", "уснул"]
CAT_LOGITS = torch.tensor([3.0, 1.8, 2.7, 2.2])
CAT_SEEN = {0, 1}

# Логиты убывают ровно, поэтому при top_p=0.9 низкая температура
# оставляет 3 кандидата, а высокая — 12. Это иллюстрация из урока.
TEMPERATURE_LOGITS = torch.tensor(
    [3.2 - 0.28 * index for index in range(20)],
    dtype=torch.float32,
)

# Почти ровное распределение на 100 токенах: top-p набирает массу долго,
# top-k заранее ограничивает ширину выбора.
WIDE_LOGITS = torch.linspace(2.0, 1.9, 100)

SITUATIONS = [
    (
        "Диалог: где-то продолжений мало, где-то несколько десятков.",
        "top-p",
        "Число естественных вариантов меняется от шага к шагу.",
    ),
    (
        "Рассказ: отрезать маловероятный хвост и расширять набор там, "
        "где модель не уверена.",
        "top-p",
        "Порог задаёт долю массы, а не число токенов.",
    ),
    (
        "Сохранить токены с суммарной вероятностью не меньше 0.9, "
        "сколько бы их ни было.",
        "top-p",
        "Это прямое определение top_p=0.9.",
    ),
    (
        "Автодополнение выбирает не более чем из пяти лучших токенов.",
        "top-k",
        "Нужен жёсткий верхний предел числа кандидатов.",
    ),
    (
        "Короткие подсказки: предсказуемость важнее разнообразия, "
        "в выборе только двадцать лучших.",
        "top-k",
        "Фиксированное k делает подсказки стабильнее.",
    ),
    (
        "Небольшая модель иногда размазывает вероятность по сотне токенов. "
        "Слишком широкий выбор нужно запретить.",
        "top-k",
        "При ровном распределении top-p оставляет почти всю сотню.",
    ),
]

PENALTY_ACTIONS = [
    (
        "Одна фраза повторяется по кругу",
        "Умеренно увеличить repetition_penalty",
    ),
    (
        "Ответ стал разнообразнее и сохранил смысл",
        "Оставить подобранное значение",
    ),
    (
        "Модель избегает нужного термина",
        "Уменьшить repetition_penalty",
    ),
    (
        "В коде подменяются имена переменных",
        "Уменьшить или отключить штраф",
    ),
    (
        "Ломаются повторяющиеся поля JSON",
        "Отключить штраф, формат контролировать отдельно",
    ),
    (
        "В стихотворении пропадает задуманный повтор",
        "Уменьшить или отключить штраф",
    ),
]

GPT2_PROMPT = "This is very very"
GPT2_PENALTIES = [1.0, 1.05, 1.1, 1.2, 1.3, 2.0, 10.0]


def show_table(title: str, rows: list[dict], float_digits: int = 4) -> None:
    """Печатает список словарей таблицей с короткими дробями."""
    print(title)
    frame = pd.DataFrame(rows)
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value, digits=float_digits: f"{value:.{digits}f}"
            )
    print(formatted.to_string(index=False))
    print()


def section(title: str) -> None:
    """Печатает заголовок раздела урока."""
    print("=" * 80)
    print(title)
    print("=" * 80)


def demo_top_k() -> None:
    """Показывает top-k на распределении транспорта из урока."""
    section("1. top-k: три самых вероятных токена из шести")
    print(
        "Таблица урока про транспорт. top_k=3 оставляет поезд, автобус "
        "и самолёт. Их исходные вероятности дают 0.35 + 0.25 + 0.18 = 0.78. "
        "После нормализации это около 0.45, 0.32 и 0.23."
    )
    print()

    logits = logits_from_probabilities(TRANSPORT_PROBS)
    snapshot = logits.clone()
    filtered = apply_top_k(logits, top_k=3)

    show_table(
        "Вероятности до фильтра и после нормализации",
        probability_rows(TRANSPORT_TOKENS, logits, filtered),
        float_digits=2,
    )
    print(
        f"Сумма исходных вероятностей оставшихся токенов: "
        f"{kept_probability_mass(logits, filtered):.2f}"
    )
    print("Оставшиеся токены:", ", ".join(kept_names(filtered, TRANSPORT_TOKENS)))
    print("Исходный тензор не изменился:", bool(torch.equal(snapshot, logits)))

    only_best = apply_top_k(logits, top_k=1)
    print(
        "При top_k=1 остаётся один токен "
        f"«{kept_names(only_best, TRANSPORT_TOKENS)[0]}» "
        "с вероятностью 1. Такой выбор совпадает с greedy."
    )
    print()


def demo_top_p() -> None:
    """Сравнивает один и тот же top-p на остром и ровном распределении."""
    section("2. top-p: одно и то же 0.9 на остром и ровном распределении")
    print(
        "top_p=0.9 означает минимальный набор самых вероятных токенов, "
        "чья суммарная вероятность не меньше 0.9. Это не 90% словаря "
        "и не гарантия правильного токена."
    )
    print()

    peaked = logits_from_probabilities(PEAKED_PROBS)
    peaked_p = apply_top_p(peaked, top_p=0.9)
    peaked_k = apply_top_k(peaked, top_k=5)

    print("Модель почти уверена: первые три токена уже содержат 0.93 массы.")
    show_table(
        "Острое распределение, top_p=0.9",
        probability_rows(PEAKED_TOKENS, peaked, peaked_p),
        float_digits=2,
    )
    print(
        "Оставлены:",
        ", ".join(kept_names(peaked_p, PEAKED_TOKENS)),
        f"(масса {kept_probability_mass(peaked, peaked_p):.2f})",
    )
    print(
        "Тот же шаг при top_k=5 дополнительно сохраняет",
        ", ".join(kept_names(peaked_k, PEAKED_TOKENS)[3:]),
        "— у дерева вероятность всего 0.02.",
    )
    print()

    flat = logits_from_probabilities(FLAT_PROBS)
    flat_p = apply_top_p(flat, top_p=0.9)
    flat_k = apply_top_k(flat, top_k=5)
    print(
        "Модель не уверена. Семь городов дают только 0.89, поэтому порог 0.9 "
        "добирается восьмым токеном, Осло. Масса набора становится 0.95."
    )
    show_table(
        "Ровное распределение, top_p=0.9",
        probability_rows(FLAT_TOKENS, flat, flat_p),
        float_digits=2,
    )
    print(
        f"top-p оставил {len(kept_token_ids(flat_p))} токенов, "
        f"top-k=5 оставил {len(kept_token_ids(flat_k))} "
        f"с массой {kept_probability_mass(flat, flat_k):.2f}. "
        "Фиксированная пятёрка отрезает Вену, Прагу и Осло, хотя они "
        "правдоподобны."
    )
    print()


def demo_temperature() -> None:
    """Показывает, как температура меняет число кандидатов при том же top-p."""
    section("3. Температура меняет, сколько токенов наберёт тот же top-p")
    print(
        "Логиты делятся на температуру до фильтрации. Здесь 20 токенов "
        "с логитами 3.2, 2.92, 2.64, ... При top_p=0.9:"
    )
    print()

    rows = []
    for temperature in (0.3, 1.0, 1.5):
        scaled = apply_temperature(TEMPERATURE_LOGITS, temperature)
        filtered = apply_top_p(scaled, top_p=0.9)
        rows.append(
            {
                "temperature": temperature,
                "kept_tokens": len(kept_token_ids(filtered)),
                "greedy_token_id": choose_greedy_token(scaled),
                "kept_mass": kept_probability_mass(scaled, filtered),
            }
        )

    show_table("Один top_p=0.9 при разной температуре", rows, float_digits=2)
    print(
        "При 0.3 остаются 3 токена, при 1.5 — 12. Жадный выбор оба раза "
        "указывает на токен 0: температура не переставляет кандидатов, "
        "она только перераспределяет массу, и top-p видит уже новую форму."
    )
    print()


def demo_flat_cap() -> None:
    """Показывает, зачем при ровном распределении ставить потолок top-k."""
    section("4. Когда ровное распределение лучше ограничить через top-k")
    print(
        "Сто токенов с логитами от 2.0 до 1.9. Вероятность почти равномерная, "
        "как у небольшой модели при высокой температуре."
    )
    print()

    by_p = apply_top_p(WIDE_LOGITS, top_p=0.9)
    by_k = apply_top_k(WIDE_LOGITS, top_k=20)
    # Сначала потолок по числу, затем доля массы уже среди этих двадцати.
    both = apply_top_p(by_k, top_p=0.9)

    rows = [
        {
            "filter": "top_p=0.9",
            "kept_tokens": len(kept_token_ids(by_p)),
            "original_mass": kept_probability_mass(WIDE_LOGITS, by_p),
        },
        {
            "filter": "top_k=20",
            "kept_tokens": len(kept_token_ids(by_k)),
            "original_mass": kept_probability_mass(WIDE_LOGITS, by_k),
        },
        {
            "filter": "top_k=20, затем top_p=0.9",
            "kept_tokens": len(kept_token_ids(both)),
            "original_mass": kept_probability_mass(WIDE_LOGITS, both),
        },
    ]
    show_table("Почти равномерные 100 токенов", rows, float_digits=2)
    print(
        "Один top-p оставляет 90 кандидатов, включая слабые продолжения. "
        "top-k=20 режет выбор до двадцати лучших. Если после этого ещё раз "
        "применить top-p, порог считается заново по этим двадцати: остаётся 18. "
        "В выбор попадают только токены, прошедшие оба фильтра. На первой "
        "настройке меняйте один фильтр за раз, иначе не видно, кто повлиял."
    )
    print()


def demo_which_parameter() -> None:
    """Раскладывает ситуации урока по top-p и top-k."""
    section("5. Какой фильтр ставить по задаче")
    print(
        "top-p сохраняет самую вероятную часть распределения: размер набора "
        "сам подстраивается под уверенность модели. top-k заранее ограничивает "
        "выбор заданным числом токенов."
    )
    print()
    for parameter in ("top-p", "top-k"):
        print(parameter)
        for text, chosen, reason in SITUATIONS:
            if chosen != parameter:
                continue
            print(f"- {text}")
            print(f"  {reason}")
        print()


def demo_repetition_penalty() -> None:
    """Показывает, как repetition penalty меняет логиты и жадный выбор."""
    section("6. repetition_penalty снижает логит уже встреченных токенов")
    print(
        "Штраф смотрит на id, а не на смысл слова. Положительный логит "
        "делится на penalty, отрицательный умножается. При 1.0 логиты те же."
    )
    print()

    raw_choice = CAT_TOKENS[choose_greedy_token(CAT_LOGITS)]
    penalized = apply_repetition_penalty(CAT_LOGITS, CAT_SEEN, penalty=1.2)
    penalized_choice = CAT_TOKENS[choose_greedy_token(penalized)]

    rows = []
    for index, token in enumerate(CAT_TOKENS):
        rows.append(
            {
                "token": token,
                "seen": index in CAT_SEEN,
                "logit_before": float(CAT_LOGITS[index]),
                "logit_after": float(penalized[index]),
            }
        )
    show_table("penalty = 1.2", rows, float_digits=2)
    print(f"До штрафа greedy выбирает «{raw_choice}».")
    print(
        f"После штрафа максимум у «{penalized_choice}»: "
        "3.0 / 1.2 = 2.5, а логит «прыгнул» остался 2.7."
    )
    print(
        "Повтор не запрещён. Если встреченный токен всё ещё лучший по "
        "контексту, модель возьмёт его снова."
    )
    print()

    negative = torch.tensor([-2.0])
    negative_after = apply_repetition_penalty(negative, {0}, penalty=1.2)
    print(
        f"Отрицательный логит -2.0 умножается на 1.2 и становится "
        f"{negative_after.item():.2f}. Деление дало бы {-2.0 / 1.2:.2f}: "
        "оценка выросла бы, и повтор стал бы вероятнее."
    )
    print()

    same_logit = torch.tensor([3.0])
    once = apply_repetition_penalty(same_logit, {0}, penalty=1.2)
    print(
        "Множество seen хранит факт появления, не счётчик. "
        f"Логит 3.0 и при одном, и при пяти вхождениях становится "
        f"{once.item():.2f}: деление выполняется один раз."
    )
    print()


def demo_frequency_penalty() -> None:
    """Сравнивает частотный штраф при одном и при пяти повторах."""
    section("7. frequency_penalty усиливается с каждым повтором")
    print("Формула урока: z' = z - alpha * count. Здесь alpha = 0.3.")
    print()

    logit = torch.tensor([3.0, 2.2])
    tokens = ["Париж", "город"]
    rows = []
    for count in (1, 5):
        processed = apply_frequency_penalty(logit, {0: count}, alpha=0.3)
        described = describe_logits(processed, tokens)
        rows.append(
            {
                "count": count,
                "paris_logit": described[0]["logit"],
                "city_logit": described[1]["logit"],
                "greedy": tokens[choose_greedy_token(processed)],
            }
        )
    show_table("Один и тот же исходный логит 3.0", rows, float_digits=2)
    print(
        "После пяти появлений логит Парижа падает до 1.5, и greedy "
        "переключается на «город». repetition_penalty оставил бы 2.5 "
        "и при пяти повторах. В transformers.generate() основной рычаг — "
        "repetition_penalty; частотный штраф бывает в отдельных API."
    )
    print()


def demo_penalty_symptoms() -> None:
    """Печатает, в какую сторону двигать штраф по симптому генерации."""
    section("8. В какую сторону двигать штраф")
    print(
        "Штраф не отличает цикл от нужного повтора. Имена переменных, "
        "ключи JSON, термины и рефрен стиха для него такие же id, как "
        "и зациклившаяся фраза."
    )
    print()
    show_table(
        "Симптом и действие",
        [
            {"symptom": symptom, "action": action}
            for symptom, action in PENALTY_ACTIONS
        ],
    )


def demo_pipeline() -> None:
    """Проводит один вектор логитов через штраф, температуру и оба фильтра."""
    section("9. Все шаги на одном векторе")
    tokens = ["кот", "он", "прыгнул", "уснул", "сел", "лампа"]
    logits = torch.tensor([3.0, 1.8, 2.7, 2.2, 0.5, -0.4])
    seen = {0, 1}

    stages: list[tuple[str, torch.Tensor]] = [("исходные логиты", logits)]
    stages.append((
        "repetition_penalty=1.2",
        apply_repetition_penalty(stages[-1][1], seen, penalty=1.2),
    ))
    stages.append((
        "temperature=0.7",
        apply_temperature(stages[-1][1], temperature=0.7),
    ))
    stages.append(("top_k=4", apply_top_k(stages[-1][1], top_k=4)))
    stages.append(("top_p=0.9", apply_top_p(stages[-1][1], top_p=0.9)))

    rows = []
    for name, stage_logits in stages:
        chosen = choose_greedy_token(stage_logits)
        rows.append(
            {
                "stage": name,
                "kept": len(kept_token_ids(stage_logits)),
                "greedy": tokens[chosen],
                "candidates": ", ".join(kept_names(stage_logits, tokens)),
            }
        )
    show_table("Порядок обработки", rows)
    print(
        "Штраф переключает greedy с «кот» на «прыгнул». Температура "
        "сохраняет этого победителя и меняет только форму распределения. "
        "Дальше top-k и top-p сужают кандидатов, из которых его выбирают."
    )
    print()


def demo_cases() -> None:
    """Подбирает минимально достаточные top-k, top-p и repetition penalty."""
    section("10. Три кейса: подобрать минимально достаточное значение")
    print(
        "В каждом кейсе параметр увеличивают, пока симптом не исчезнет. "
        "Для top-p наоборот: берут наибольший порог, при котором проблемный "
        "токен уже выпал. Так ограничение остаётся самым мягким из рабочих."
    )
    print()

    case_1, case_2, case_3, diagnosis = diagnose_decoding_controls()
    print("Кейс 1. Уже встреченные id 0 и 1 снова выигрывают greedy.")
    print(case_1.to_string(index=False))
    print()
    print(
        "При 1.05 логит « with» падает с 5.0 до 5.0 / 1.05 ≈ 4.76. "
        "« First» остаётся 4.8, и цикл на этом шаге прекращается. "
        "top-k и top-p историю токенов не видят и могут оставить повтор в наборе."
    )
    print()

    print("Кейс 2. Пять seed: 11, 22, 33, 44, 55.")
    print(case_2.to_string(index=False))
    print()
    print(
        "При top_k=1 во всех seed остаётся « take». Температура здесь не "
        "помогла бы: после фильтра кандидат всё равно один. Три разных "
        "токена впервые появляются при top_k=3."
    )
    print()

    print("Кейс 3. Слабый токен « DNS» сидит в хвосте с вероятностью 0.01.")
    print(case_3.to_string(index=False))
    print()
    print(
        "При top_p=1.0 фильтр выключен, DNS доступен. При 0.95 первые восемь "
        "токенов набирают 0.97, а « read» и « DNS» выпадают. Пороги 0.9 и 0.8 "
        "тоже его убирают, но режут сильнее. Подходит максимальный из рабочих: "
        "0.95."
    )
    print()
    print("Итоговая диагностика")
    print(diagnosis.to_string(index=False))
    print()


def demo_gpt2() -> None:
    """Снимает цикл greedy-генерации gpt2 штрафом за повтор."""
    section("11. gpt2: тот же цикл, что в уроке, и штраф внутри generate")
    print(
        "do_sample=False, поэтому на каждом шаге берётся максимум логита. "
        "top-k и top-p при greedy не используются. repetition_penalty "
        "по-прежнему меняет логиты до выбора. Промпт:",
        GPT2_PROMPT,
    )
    print("Загружаю gpt2. При первом запуске веса скачиваются.")
    print()

    tokenizer, model = load_causal_model("gpt2")
    previous_text = None
    for penalty in GPT2_PENALTIES:
        text = generate_greedy_text(
            model=model,
            tokenizer=tokenizer,
            prompt=GPT2_PROMPT,
            repetition_penalty=penalty,
            max_new_tokens=35,
        )
        same_as_previous = text == previous_text
        print("-" * 80)
        print(f"repetition_penalty = {penalty}")
        if same_as_previous:
            print("Текст совпал с предыдущим значением штрафа.")
        print(text)
        print()
        previous_text = text

    print(
        "1.0 не штрафует встреченные токены, и greedy может закрепить одну "
        "и ту же фразу. Небольшой рост, например до 1.05, иногда меняет "
        "один максимум — дальше вся траектория другая. Несколько разных "
        "штрафов могут дать один текст: greedy смотрит на то, какой логит "
        "максимален, а не на то, насколько сильно он уменьшился. Штраф 10 "
        "здесь может выглядеть безобидно и в другом промпте сломать нужные "
        "повторы. Цикл уже снял 1.05, дальше без нового симптома его не поднимают."
    )
    print()
