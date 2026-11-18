"""Пример: зачем авторегрессивной генерации KV-cache и сколько он стоит.

Без кеша K и V уже прочитанных токенов пересчитываются на каждом шаге.
Сумма таких шагов квадратична по длине промпта и ответа. С кешем prefill
читает промпт один раз, а каждый новый токен добавляет только себя.
Память кеша растёт линейно с контекстом, со слоями и с размером батча.
Prefix caching сохраняет ещё и общий префикс разных запросов.

Замер в конце повторяет авторский бенчмарк: длинный промпт, жадная
генерация, 100 новых токенов, сравнение ``use_cache=True`` и ``False``.
Урок запускает Qwen2.5-3B-Instruct на CUDA. Здесь CUDA нет, поэтому
та же схема идёт на уже скачанной Qwen2.5-0.5B-Instruct и на MPS, если
он доступен. Секунды из-за этого другие. Формулы памяти и prefix caching
от устройства не зависят.

Запуск из корня проекта:

    python example_kv_cache.py
"""

import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    sys.exit(
        "Для этого примера не хватает torch.\n"
        "Поставьте зависимости из requirements.txt в тот же интерпретатор, "
        "которым запускаете скрипт."
    )

from utils.inference import (
    CACHE_QUIZ,
    FAQ_PREFIX_TOKENS,
    FAQ_REQUESTS,
    FAQ_UNIQUE_TOKENS,
    FLOAT16_BYTES,
    LESSON_KV_MODEL,
    LESSON_NEW_TOKENS,
    LESSON_REFERENCE_CACHED_SECONDS,
    LESSON_REFERENCE_NOCACHE_SECONDS,
    LESSON_REFERENCE_PROMPT_TOKENS,
    LESSON_WARMUP_TOKENS,
    QUIZ_BATCH_SIZE,
    QUIZ_CACHE_MEGABYTES,
    QUIZ_HEAD_DIM,
    QUIZ_KV_HEADS,
    QUIZ_LAYERS,
    QUIZ_PREFIX_PREFILL_TOKENS,
    QUIZ_PREFIX_TOKENS,
    QUIZ_REQUESTS,
    QUIZ_SEQ_LEN,
    QUIZ_UNIQUE_TOKENS,
    QWEN25_3B_ATTENTION_HEADS,
    QWEN25_3B_KV_HEADS,
    attention_layout,
    benchmark_generation,
    cache_speedup,
    cached_attention_cells,
    cached_token_work,
    gqa_cache_factor,
    kv_cache_megabytes,
    naive_attention_cells,
    naive_token_work,
    prefill_tokens,
    prefix_savings_factor,
    repeat_prompt,
)
from utils.models import (
    QWEN_INSTRUCT_MODEL,
    inference_device,
    load_chat_model,
    release_chat_model,
)


def section(title: str) -> None:
    print("=" * 80, flush=True)
    print(title, flush=True)
    print("=" * 80, flush=True)


def print_work(prompt_tokens: int, new_tokens: int) -> None:
    plain = naive_token_work(prompt_tokens, new_tokens)
    cached = cached_token_work(prompt_tokens, new_tokens)
    plain_cells = naive_attention_cells(prompt_tokens, new_tokens)
    cached_cells = cached_attention_cells(prompt_tokens, new_tokens)
    print(
        f"Промпт {prompt_tokens}, ответ {new_tokens}: "
        f"токены без кеша {plain}, с кешем {cached}, "
        f"отношение {plain / cached:.1f}x"
    )
    print(
        f"Ячейки внимания: без кеша {plain_cells}, "
        f"с кешем {cached_cells}, отношение {plain_cells / cached_cells:.1f}x"
    )


def main() -> None:
    section("1. Без кеша одни и те же K и V считаются на каждом шаге")
    print(
        "Промпт «Привет, как» — три токена. На первом шаге модель считает "
        "Q, K и V для каждого и предсказывает четвёртый, например «дела». "
        "На втором шаге токенов уже четыре, и наивный проход заново считает "
        "K и V первых трёх, хотя токен «Привет» не изменился. На третьем "
        "шаге пересчёт повторяется уже для пяти позиций."
    )
    print(
        "Если промпт длиной P, а ответ длиной L, длины наивных шагов — "
        "это P, P + 1, ..., P + L - 1. Сумма этих длин порядка (P + L)². "
        "С кешем prefill читает P один раз, а остальные L - 1 шагов читают "
        "по одному новому токену. Сложность генерации становится линейной."
    )
    print_work(prompt_tokens=3, new_tokens=3)
    print(
        "На длинном промпте авторского замера разрыв уже не учебный. "
        f"Промпт {LESSON_REFERENCE_PROMPT_TOKENS} токенов и ответ "
        f"{LESSON_NEW_TOKENS}:"
    )
    print_work(LESSON_REFERENCE_PROMPT_TOKENS, LESSON_NEW_TOKENS)
    print(
        "Отношение по токенам — верхняя оценка повторной работы. На стенде "
        "оно не превращается в такое же отношение секунд: прогон без кеша "
        "загружает GPU широкими матрицами, и один токен там дешевле, чем "
        "в узком шаге с кешем. Секунд всё равно больше, потому что токенов "
        "через модель проходит намного больше."
    )

    section("2. Что именно лежит в кеше")
    print(
        "Профилирование длинного промпта на 10 000 токенов показывает "
        "резкий рост памяти после первого шага и дальше медленный линейный "
        "прирост. Ниже варианты задания и почему они верные или нет."
    )
    for statement in CACHE_QUIZ:
        mark = "верно" if statement.correct else "неверно"
        print(f"- [{mark}] {statement.text}")
        print(f"  {statement.reason}")
    per_token = kv_cache_megabytes(
        QUIZ_LAYERS,
        QUIZ_KV_HEADS,
        QUIZ_HEAD_DIM,
        seq_len=1,
        batch_size=QUIZ_BATCH_SIZE,
        dtype_bytes=FLOAT16_BYTES,
    )
    prefill_mb = kv_cache_megabytes(
        QUIZ_LAYERS,
        QUIZ_KV_HEADS,
        QUIZ_HEAD_DIM,
        seq_len=10_000,
        batch_size=QUIZ_BATCH_SIZE,
        dtype_bytes=FLOAT16_BYTES,
    )
    print(
        f"На модели задания 2 один токен кеша весит {per_token:.1f} МБ. "
        f"Prefill 10 000 токенов занимает {prefill_mb:.0f} МБ, затем каждый "
        "новый токен добавляет те же "
        f"{per_token:.1f} МБ. Веса в этот прирост не входят."
    )

    section("3. Сколько памяти занимает кеш")
    print(
        "Размер = 2 × слои × KV-головы × d_head × seq_len × batch × байт. "
        "Двойка — это K и V. seq_len растёт во время генерации, batch "
        "умножает кеш, потому что у каждого запроса он свой."
    )
    quiz_mb = kv_cache_megabytes(
        QUIZ_LAYERS,
        QUIZ_KV_HEADS,
        QUIZ_HEAD_DIM,
        QUIZ_SEQ_LEN,
        QUIZ_BATCH_SIZE,
        FLOAT16_BYTES,
    )
    print(
        f"Задание 2: {QUIZ_LAYERS} слоёв, {QUIZ_KV_HEADS} KV-голов, "
        f"d_head {QUIZ_HEAD_DIM}, {QUIZ_SEQ_LEN} токенов, batch "
        f"{QUIZ_BATCH_SIZE}, float16. Кеш одного запроса: {quiz_mb:.0f} МБ."
    )
    if quiz_mb != QUIZ_CACHE_MEGABYTES:
        raise SystemExit(
            f"Ожидались {QUIZ_CACHE_MEGABYTES} МБ, получилось {quiz_mb}."
        )
    batch_mb = kv_cache_megabytes(
        QUIZ_LAYERS,
        QUIZ_KV_HEADS,
        QUIZ_HEAD_DIM,
        QUIZ_SEQ_LEN,
        batch_size=8,
        dtype_bytes=FLOAT16_BYTES,
    )
    print(
        f"Батч из восьми таких запросов — уже {batch_mb:.0f} МБ только под "
        "кеш, то есть 16 ГБ. На длинном контексте кеш становится узким "
        "местом рядом с весами."
    )
    factor = gqa_cache_factor(QWEN25_3B_ATTENTION_HEADS, QWEN25_3B_KV_HEADS)
    print(
        f"GQA уменьшает этот объём. У {LESSON_KV_MODEL} {QWEN25_3B_ATTENTION_HEADS} "
        f"голов запроса и {QWEN25_3B_KV_HEADS} группы K/V: кеш меньше в "
        f"{factor} раз, чем был бы при отдельной паре K/V на каждую голову. "
        "Память всё равно растёт линейно с длиной, её по-прежнему надо "
        "закладывать."
    )

    section("4. Prefix caching переиспользует общий префикс")
    print(
        "Обычный KV-cache живёт внутри одной генерации. Prefix caching "
        "оставляет K и V неизменного начала для следующих запросов. "
        "Выгода тем больше, чем длиннее общий префикс, чем короче "
        "уникальный хвост и чем чаще этот префикс встречается. Если общего "
        "начала нет, сохранять нечего, и оба режима обрабатывают одинаковый "
        "объём."
    )
    faq_plain = prefill_tokens(
        FAQ_PREFIX_TOKENS,
        FAQ_UNIQUE_TOKENS,
        FAQ_REQUESTS,
        reuse_prefix=False,
    )
    faq_cached = prefill_tokens(
        FAQ_PREFIX_TOKENS,
        FAQ_UNIQUE_TOKENS,
        FAQ_REQUESTS,
        reuse_prefix=True,
    )
    print(
        f"FAQ-бот: префикс {FAQ_PREFIX_TOKENS}, вопрос {FAQ_UNIQUE_TOKENS}, "
        f"{FAQ_REQUESTS} запросов. Без переиспользования prefill читает "
        f"{faq_plain} токенов. С ним — {faq_cached}. Это меньше примерно "
        f"в {prefix_savings_factor(FAQ_PREFIX_TOKENS, FAQ_UNIQUE_TOKENS, FAQ_REQUESTS):.0f} раз."
    )
    quiz_cached = prefill_tokens(
        QUIZ_PREFIX_TOKENS,
        QUIZ_UNIQUE_TOKENS,
        QUIZ_REQUESTS,
        reuse_prefix=True,
    )
    quiz_plain = prefill_tokens(
        QUIZ_PREFIX_TOKENS,
        QUIZ_UNIQUE_TOKENS,
        QUIZ_REQUESTS,
        reuse_prefix=False,
    )
    print(
        f"Задание 4: префикс {QUIZ_PREFIX_TOKENS}, вопрос {QUIZ_UNIQUE_TOKENS}, "
        f"{QUIZ_REQUESTS} запросов. С prefix caching prefill обрабатывает "
        f"{quiz_cached} токенов."
    )
    if quiz_cached != QUIZ_PREFIX_PREFILL_TOKENS:
        raise SystemExit(
            f"Ожидались {QUIZ_PREFIX_PREFILL_TOKENS} токенов, получилось {quiz_cached}."
        )
    print(
        f"Без него было бы {quiz_plain} токенов, примерно в "
        f"{prefix_savings_factor(QUIZ_PREFIX_TOKENS, QUIZ_UNIQUE_TOKENS, QUIZ_REQUESTS):.0f} "
        "раз больше. На тысячах запросов в день стоимость общего префикса "
        "размазывается ещё сильнее."
    )

    section("5. Замер на локальной модели")
    device = inference_device()
    print(
        f"Авторский код грузит {LESSON_KV_MODEL} во float16 на CUDA, "
        "прогревает коротким вызовом и сравнивает 100 жадных токенов. "
        f"На GPU урока промпт вышел {LESSON_REFERENCE_PROMPT_TOKENS} токенов: "
        f"{LESSON_REFERENCE_CACHED_SECONDS:.2f} с с кешем, "
        f"{LESSON_REFERENCE_NOCACHE_SECONDS:.2f} с без кеша, "
        f"ускорение {LESSON_REFERENCE_NOCACHE_SECONDS / LESSON_REFERENCE_CACHED_SECONDS:.1f}x."
    )
    print(
        f"Здесь устройство {device.type}. Модель — {QWEN_INSTRUCT_MODEL}: "
        "она уже есть в кеше и помещается в память. Схема замера та же, "
        "включая синхронизацию MPS или CUDA до и после generate. "
        "Первый вызов — прогрев, его время в сравнение не входит."
    )
    tokenizer, model = load_chat_model(QWEN_INSTRUCT_MODEL)
    try:
        prompt = repeat_prompt()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        print(f"Длина промпта в токенах: {prompt_tokens}", flush=True)
        layout = attention_layout(model.config)
        measured_mb = kv_cache_megabytes(
            layout["num_layers"],
            layout["num_kv_heads"],
            layout["head_dim"],
            prompt_tokens + LESSON_NEW_TOKENS,
            batch_size=1,
            dtype_bytes=2 if device.type in {"cuda", "mps"} else 4,
        )
        print(
            f"Конфиг: {layout['num_layers']} слоёв, "
            f"{layout['num_attention_heads']} голов запроса, "
            f"{layout['num_kv_heads']} KV-голов, d_head {layout['head_dim']}. "
            f"GQA уменьшает кеш в "
            f"{gqa_cache_factor(layout['num_attention_heads'], layout['num_kv_heads'])} раз. "
            f"Кеш на промпт и {LESSON_NEW_TOKENS} новых токенов — "
            f"{measured_mb:.1f} МБ."
        )
        print_work(prompt_tokens, LESSON_NEW_TOKENS)

        print("Прогрев, 5 токенов с кешем.", flush=True)
        benchmark_generation(
            model,
            inputs,
            use_cache=True,
            max_new_tokens=LESSON_WARMUP_TOKENS,
        )
        print(f"Замер с кешем, {LESSON_NEW_TOKENS} токенов.", flush=True)
        cached = benchmark_generation(
            model,
            inputs,
            use_cache=True,
            max_new_tokens=LESSON_NEW_TOKENS,
        )
        print(f"Замер без кеша, {LESSON_NEW_TOKENS} токенов.", flush=True)
        plain = benchmark_generation(
            model,
            inputs,
            use_cache=False,
            max_new_tokens=LESSON_NEW_TOKENS,
        )
    finally:
        release_chat_model(model)

    speedup = cache_speedup(float(plain["seconds"]), float(cached["seconds"]))
    print(f"С кешем:    {cached['seconds']:.2f} сек")
    print(f"Без кеша:   {plain['seconds']:.2f} сек")
    print(f"Ускорение:  {speedup:.1f}x")
    print(
        f"Новых токенов: с кешем {cached['new_tokens']}, "
        f"без кеша {plain['new_tokens']}."
    )
    print(
        "С кешем каждый из 100 шагов после prefill обрабатывает один новый "
        "токен, а K и V длинного префикса читаются из памяти. Без кеша "
        "каждый шаг заново прогоняет внимание по всей последовательности, "
        "и она ещё растёт. На коротком тексте разрыв был бы скромным. "
        "В рабочей генерации кеш поэтому не отключают: режим без него нужен, "
        "чтобы увидеть цену повторного счёта."
    )


if __name__ == "__main__":
    main()
