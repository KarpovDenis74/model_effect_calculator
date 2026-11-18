"""Пример: почему длинный контекст дорожает быстрее, чем растёт.

Внимание строит матрицу «токен на токен». Удвоение входа увеличивает
эту матрицу в четыре раза, поэтому растут и память, и время prefill.
TTFT — время до первого токена — как раз и есть этот prefill. Генерация
самого ответа идёт уже токен за токеном и от длины входа зависит слабо.

Отдельно страдает качество. Модель лучше держит факты в начале и в конце
контекста и чаще теряет то, что лежит в середине. Это lost in the middle:
инструкцию, вопрос и нужный фрагмент базы знаний ставят на края.

Практика считает TTFT локально на Qwen2.5-0.5B-Instruct и оценивает
стоимость по тарифу $0.15 / $0.60 за миллион входных и выходных токенов.
Ограничения продукта: первый токен не позже 1 секунды и не дороже
$0.0005 за запрос. При первом запуске модель скачивается, это около 1 ГБ.
Если есть MPS или CUDA, замер идёт на нём.

Запуск из корня проекта:

    python example_long_context.py
"""

import importlib.util
import sys

missing = [
    name
    for name in ("torch", "matplotlib")
    if importlib.util.find_spec(name) is None
]
if missing:
    sys.exit(
        "Для этого примера не хватает пакетов: "
        + ", ".join(missing)
        + ".\nПоставьте зависимости из requirements.txt в тот же интерпретатор, "
        "которым запускаете скрипт."
    )

from utils.models import (
    CONTEXT_MODEL_NAME,
    CONTEXT_QUESTION,
    DEFAULT_TARGET_LENGTHS,
    FILLER_SENTENCE,
    LESSON_INPUT_PRICE_PER_MILLION,
    LESSON_OUTPUT_PRICE_PER_MILLION,
    REQUEST_COST_LIMIT_USD,
    TTFT_LIMIT_SECONDS,
    PromptPart,
    attention_matrix_cells,
    attention_scale_factor,
    describe_context_load,
    estimate_context_cost,
    inference_device,
    load_causal_model,
    make_prompt,
    measure_ttft,
    order_prompt_parts,
    plot_context_tradeoff,
    render_prompt,
    summarize_context_budget,
)

BINDING_TEXT = {
    "ttft": "время до первого токена",
    "cost": "стоимость запроса",
    "both": "и время, и стоимость",
    None: "ни один замер лимит не нарушил",
}

QUIZ_CASES = [
    ("Короткий вопрос с фрагментом текста", 1000, 32_000),
    ("Пара реплик и системная инструкция", 800, 32_000),
    ("Крупный документ целиком", 8000, 32_000),
    ("Несколько документов внутри окна 32k", 12_000, 32_000),
    ("Документ длиннее окна", 200_000, 32_000),
]

CARSHARING_PARTS = [
    PromptPart(
        label="Системная инструкция",
        placement="start",
        text=(
            "Ты — ассистент службы поддержки каршеринга. "
            "Отвечай кратко и только на основе документов ниже. "
            "Если ответа нет в документах, прямо скажи об этом."
        ),
    ),
    PromptPart(
        label="Тарифы, не по вопросу",
        placement="middle",
        text="Фрагмент базы знаний: два абзаца про тарифные планы — не по теме вопроса.",
    ),
    PromptPart(
        label="Старая переписка про тарифы",
        placement="middle",
        text=(
            "История переписки: шесть предыдущих реплик пользователя и бота "
            "по другому вопросу — тарифам, не связанным с текущим."
        ),
    ),
    PromptPart(
        label="Залог, не по вопросу",
        placement="middle",
        text=(
            "Фрагмент базы знаний: три абзаца про политику возврата залога "
            "при повреждении автомобиля — к текущему вопросу не относится."
        ),
    ),
    PromptPart(
        label="Текущий вопрос",
        placement="end",
        text="Не получается зайти в аккаунт, доступа к почте нет, что делать?",
    ),
    PromptPart(
        label="Фрагмент с точным ответом",
        placement="end",
        text=(
            "Чтобы сбросить пароль без доступа к почте, подтвердите личность "
            "номером договора аренды в чате поддержки."
        ),
    ),
]


def section(title: str) -> None:
    print("=" * 80, flush=True)
    print(title, flush=True)
    print("=" * 80, flush=True)


def main() -> None:
    section("1. Матрица внимания растёт как квадрат длины")
    print(
        "Каждый токен сравнивается с каждым. Контекст из n токенов "
        "даёт n * n ячеек. Удвоили длину — ячеек стало в четыре раза больше."
    )
    for num_tokens in (4, 8, 1000, 8000, 32_000):
        print(
            f"{num_tokens:>6} токенов → "
            f"{attention_matrix_cells(num_tokens):>15,} ячеек"
        )
    print(
        "От 1000 до 32 000 токенов длина выросла в "
        f"{32_000 / 1000:.0f} раз, а матрица — в "
        f"{attention_scale_factor(1000, 32_000):.0f} раз."
    )

    section("2. Как окно в 32k переваривает разную длину")
    print(
        "Короткий вопрос модель принимает как обычный запрос. "
        "Документ на тысячи токенов уже заметно дороже и медленнее, "
        "хотя в окно он ещё входит. Текст длиннее окна модель не обработает."
    )
    for title, prompt_tokens, window in QUIZ_CASES:
        verdict = describe_context_load(prompt_tokens, window)
        print(f"{prompt_tokens:>7} / окно {window:<7}  {verdict:<32}  {title}")

    section("3. Важное ставим на края, фон оставляем в середине")
    print(
        "Инструкция открывает промпт. Вопрос пользователя и фрагмент "
        "с точным ответом закрывают его. Тарифы, старая переписка и залог "
        "к вопросу не относятся, поэтому остаются между краями."
    )
    for index, part in enumerate(order_prompt_parts(CARSHARING_PARTS), start=1):
        print(f"{index}. [{part.placement}] {part.label}")
    print()
    print(render_prompt(CARSHARING_PARTS))

    section("4. TTFT и стоимость на Qwen2.5-0.5B-Instruct")
    print(
        f"Тариф для оценки: ${LESSON_INPUT_PRICE_PER_MILLION:.2f} за 1 млн входных "
        f"и ${LESSON_OUTPUT_PRICE_PER_MILLION:.2f} за 1 млн выходных токенов. "
        f"Лимиты продукта: TTFT ≤ {TTFT_LIMIT_SECONDS:.0f} с и "
        f"≤ ${REQUEST_COST_LIMIT_USD:.4f} за запрос."
    )
    print(
        "Промпты отличаются только числом повторов нейтрального предложения. "
        "Вопрос в конце один и тот же, поэтому сравнивается длина, а не задача."
    )
    print("Наполнитель:", FILLER_SENTENCE.strip())
    print("Вопрос:", CONTEXT_QUESTION.strip())

    device = inference_device()
    print("Устройство замера:", device)
    tokenizer, model = load_causal_model(CONTEXT_MODEL_NAME)
    model.to(device)
    print("Модель:", CONTEXT_MODEL_NAME)

    print("Прогрев не входит в таблицу: первый запуск прогревает кеши.")
    measure_ttft(model, tokenizer, "Привет", max_new_tokens=5)

    rows = []
    for target in DEFAULT_TARGET_LENGTHS:
        prompt = make_prompt(tokenizer, target)
        measured = measure_ttft(model, tokenizer, prompt)
        cost = estimate_context_cost(measured["prompt_tokens"], measured["output_tokens"])
        row = {**measured, "cost": cost}
        rows.append(row)
        decode_seconds = measured["e2e"] - measured["ttft"]
        print(
            f"контекст {measured['prompt_tokens']:>5} ток | "
            f"TTFT {measured['ttft']:.2f} с | "
            f"генерация {decode_seconds:.2f} с | "
            f"E2E {measured['e2e']:.2f} с | "
            f"${cost:.6f}",
            flush=True,
        )

    report = summarize_context_budget(rows)
    print()
    for row in report["rows"]:
        status = "укладывается" if row["fits"] else "не укладывается"
        print(
            f"{row['prompt_tokens']:>5} токенов: "
            f"TTFT {row['ttft']:.2f} с ({'ок' if row['ttft_ok'] else 'выше лимита'}), "
            f"${row['cost']:.6f} ({'ок' if row['cost_ok'] else 'выше лимита'}) — "
            f"{status}"
        )

    longest = report["longest_fitting_tokens"]
    if longest is None:
        print(
            "Уже самая короткая измеренная длина не проходит оба ограничения. "
            "Безопасный порог лежит ниже первой точки лесенки."
        )
    else:
        print(
            "Оба ограничения держатся до "
            f"{longest} токенов. Следующая измеренная длина их нарушает."
        )
    if report["later_fitting_tokens"]:
        print(
            "После первого нарушения снова уложились длины: "
            f"{report['later_fitting_tokens']}. "
            "На границе секунды TTFT скачет от запуска к запуску, "
            "поэтому порог ставится по первому срыву, а не по отдельной удачной точке."
        )
    print("Первым ломается:", BINDING_TEXT[report["binding_limit"]])
    print(
        "Конкретная граница зависит от устройства и тарифа. Метод тот же: "
        "измерить длины, наложить лимиты и дальше этой границы сокращать "
        "историю или базу знаний."
    )

    plot_path = plot_context_tradeoff(rows, "outputs/long_context_tradeoff.png")
    print("Графики:", plot_path)


if __name__ == "__main__":
    main()
