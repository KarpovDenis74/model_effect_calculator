"""Пример: когда поток запросов уже не тянет transformers.pipeline.

Pipeline берёт запросы по одному. vLLM — отдельный сервер: continuous
batching и PagedAttention встроены в движок, а снаружи он выглядит как
OpenAI API. На одном запросе за раз они близки. Когда десятки запросов
приходят одновременно, pipeline копит очередь, а vLLM уплотняет батч.

Сервер ``vllm serve`` требует NVIDIA. Здесь CUDA нет, поэтому команда
запуска и асинхронный клиент собраны, но процесс не стартует. Вместо
него на Qwen2.5-0.5B-Instruct меряется pipeline по одному и статические
батчи 4 и 16 одинаковых промптов. Для равной длины ответов такая волна
похожа на пачку, которую continuous batching забрал бы разом. График
ложится в ``outputs/vllm_throughput.png``.

Запуск из корня проекта:

    python example_vllm.py
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    sys.exit(
        "Для этого примера не хватает torch.\n"
        "Поставьте зависимости из requirements.txt в тот же интерпретатор, "
        "которым запускаете скрипт."
    )

from utils.inference import (
    LESSON_VLLM_MODEL,
    LESSON_VLLM_SAMPLE,
    PAGE_DEMO_LENGTHS,
    PAGE_DEMO_PAGE_SIZE,
    PAGE_DEMO_RESERVED,
    VLLM_ADVICE,
    VLLM_CLAIMS,
    VLLM_CONCURRENCIES,
    VLLM_FLAG_CARDS,
    VLLM_MAX_NEW_TOKENS,
    VLLM_PROMPT,
    VLLM_READY_URL,
    VLLM_TOTAL_REQUESTS,
    benchmark_batch,
    benchmark_pipeline,
    build_vllm_command,
    burst_metrics,
    chat_prompt,
    contiguous_reserved_tokens,
    load_test_vllm,
    load_text_pipeline,
    make_vllm_client,
    paged_reserved_tokens,
    plot_throughput,
    release_pipeline,
    reservation_waste,
    server_is_ready,
    vllm_runtime_available,
)
from utils.inference.kv_cache import synchronize_device
from utils.models import QWEN_INSTRUCT_MODEL
from transformers import GenerationConfig


def section(title: str) -> None:
    print("=" * 80, flush=True)
    print(title, flush=True)
    print("=" * 80, flush=True)


def print_burst(title: str, result: dict) -> None:
    print(
        f"{title} | concurrency={result['concurrency']:>2} | "
        f"{result['time_sec']:6.1f} с | {result['throughput']:.2f} запр/с | "
        f"ожидание {result['avg_user_latency']:5.2f} с | "
        f"обслуживание {result['avg_service']:.2f} с"
    )


def main() -> None:
    section("1. Сервер — это не pipeline за HTTP")
    print(
        "Пока запрос один, generate и pipeline справляются. Когда пользователей "
        "много, GPU простаивает между короткими ответами, если брать их по "
        "очереди. Continuous batching освобождает слот сразу и ставит туда "
        "следующий запрос. Pipeline так не умеет: следующий ждёт, пока "
        "закончится текущий вызов."
    )
    print(
        "vLLM принимает запросы по сети и держит их в одном непрерывном батче. "
        "Это другой движок, а не обёртка, которая внутри вызывает pipeline()."
    )
    for claim in VLLM_CLAIMS:
        mark = "верно" if claim.correct else "неверно"
        print(f"- [{mark}] {claim.text}")
        print(f"  {claim.reason}")

    section("2. PagedAttention раздаёт KV-cache страницами")
    useful = sum(PAGE_DEMO_LENGTHS)
    contiguous = contiguous_reserved_tokens(len(PAGE_DEMO_LENGTHS), PAGE_DEMO_RESERVED)
    paged = paged_reserved_tokens(PAGE_DEMO_LENGTHS, PAGE_DEMO_PAGE_SIZE)
    plain = reservation_waste(contiguous, useful)
    pages = reservation_waste(paged, useful)
    print(
        "Длина ответа заранее неизвестна. Сплошная схема бронирует на каждый "
        f"запрос {PAGE_DEMO_RESERVED} токенов кеша, даже если ответ короткий. "
        "Отдать дырку соседу нельзя: кусок уже непрерывный и занят."
    )
    print(
        f"Четыре ответа {list(PAGE_DEMO_LENGTHS)} занимают {useful} токенов "
        f"полезного кеша. Сплошная бронь — {plain['reserved']} токенов, "
        f"пусто {plain['wasted']} ({plain['waste_share']:.0%}). "
        f"Страницы по {PAGE_DEMO_PAGE_SIZE}: занято {pages['reserved']}, "
        f"пусто только хвосты страниц, {pages['wasted']} ({pages['waste_share']:.0%})."
    )
    if plain["wasted"] != 332 or pages["wasted"] != 44:
        raise SystemExit(
            f"Ожидались пустоты 332 и 44, получилось {plain['wasted']} и {pages['wasted']}."
        )
    print(
        "Как и виртуальная память ОС, PagedAttention не выдаёт один сплошной "
        "кусок физической памяти. Законченный запрос сразу возвращает страницы "
        "в пул. На той же карте помещается больше одновременных KV-cache, "
        "батч становится плотнее, throughput растёт. Одному запросу это "
        "не обещает кратного ускорения: делить батч не с кем."
    )

    section("3. При какой нагрузке переход оправдан")
    print(
        "На одном запросе за раз объединять некого, и vLLM с pipeline идут "
        "рядом. Pipeline и дальше обрабатывает пачку цепочкой: время "
        "складывается. vLLM считает одновременные запросы вместе, и его "
        "throughput растёт, пока карта была недогружена одиночным проходом."
    )
    print("На GPU урока пачка из 16 одинаковых запросов к vLLM выглядела так:")
    for sample in LESSON_VLLM_SAMPLE:
        print_burst("vLLM урока", sample)
    print(
        "При concurrency=1 среднее ожидание около 30 с, а обслуживание около "
        "3.5 с: запросы отправили разом, и почти всё время они стоят в очереди. "
        "При concurrency=16 очереди нет, ожидание и обслуживание сходятся. "
        "Пользователь жалуется на ожидание, сервер видит только обслуживание."
    )
    for choice in VLLM_ADVICE:
        mark = "верно" if choice.correct else "неверно"
        print(f"- [{mark}] {choice.text}")
        print(f"  {choice.reason}")
    print(
        "Учебный прогон смотрит throughput и средние задержки. Продакшн "
        "добавляет время до первого токена, время на один токен генерации "
        "и хвосты p95/p99, а не среднее по короткой пачке."
    )

    section("4. Флаги запуска и клиент")
    print("Команда урока:")
    print(" ".join(build_vllm_command()))
    print(
        "Prefix caching в неё не входит. Все 16 запросов одинаковые, и общий "
        "префикс подарил бы vLLM фору: pipeline считает его каждый раз заново. "
        "Для FAQ-бота с длинным system-промптом --enable-prefix-caching как раз "
        "самый полезный флаг."
    )
    for card in VLLM_FLAG_CARDS:
        print(f"- {card.flag}: {card.text}")
        print(f"  {card.reason}")
    print(
        "Клиент — обычный OpenAI SDK. Меняется base_url на "
        "http://localhost:8000/v1, api_key любая непустая строка. "
        "temperature=0 совпадает с жадным do_sample=False у baseline."
    )
    if vllm_runtime_available():
        print("Исполняемый файл vllm и CUDA на месте, сервер можно поднимать.")
    else:
        print(
            "Исполняемого vllm и CUDA здесь нет. Процесс не запускается, "
            "лог vllm_server.log не пишется. Ниже — замер pipeline и волн "
            "батча на локальной модели."
        )
    if server_is_ready(VLLM_READY_URL, timeout=1):
        print(f"{VLLM_READY_URL} уже отвечает, прогоняю клиент урока.")
        run_live_server()
    else:
        print(f"{VLLM_READY_URL} молчит. Живой нагрузочный тест клиента пропущен.")

    section("5. Pipeline по одному и волны батча")
    print(
        f"Урок грузит {LESSON_VLLM_MODEL} во float16 на CUDA, 16 запросов, "
        f"до {VLLM_MAX_NEW_TOKENS} новых токенов, промпт «{VLLM_PROMPT}». "
        f"Здесь та же схема на {QWEN_INSTRUCT_MODEL}. Прогрев — один полный "
        "ответ той же длины, в таблицу он не входит: короткого прогрева на "
        "5 токенов мало, первый длинный вызов на MPS всё равно холодный."
    )
    text_pipe = load_text_pipeline(QWEN_INSTRUCT_MODEL)
    try:
        prompt = chat_prompt(text_pipe.tokenizer, VLLM_PROMPT)
        device = text_pipe.model.device
        warmup = GenerationConfig(
            max_new_tokens=VLLM_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=text_pipe.tokenizer.pad_token_id,
        )
        print("Прогрев одного ответа.", flush=True)
        synchronize_device(device)
        warmed = text_pipe(prompt, generation_config=warmup)
        synchronize_device(device)
        preview = warmed[0]["generated_text"]
        print("Фрагмент прогрева:", preview[-240:].replace("\n", " "))
        sequential = benchmark_pipeline(text_pipe, prompt)
        wave_seconds = {1: float(sequential["avg_service"])}
        for width in (4, 16):
            print(f"волна из {width}", flush=True)
            wave_seconds[width] = benchmark_batch(text_pipe, prompt, width)
    finally:
        release_pipeline(text_pipe)
        del text_pipe

    print(
        f"pipeline | {sequential['total']} запросов подряд | "
        f"{sequential['time_sec']:.1f} с | "
        f"{sequential['throughput']:.2f} запр/с | "
        f"обслуживание {sequential['avg_service']:.2f} с/запрос"
    )
    print(
        "Дальше те же 16 запросов считаются отправленными разом. Волна "
        "размера 1 берёт среднее время одного запроса из цикла выше. Волны "
        "4 и 16 — измеренные статические батчи."
    )
    points: list[tuple[int, float]] = []
    for concurrency in VLLM_CONCURRENCIES:
        burst = burst_metrics(
            VLLM_TOTAL_REQUESTS,
            concurrency,
            wave_seconds[concurrency],
        )
        points.append((concurrency, float(burst["throughput"])))
        print_burst("локальная волна", burst)
    plot_path = plot_throughput(
        float(sequential["throughput"]),
        points,
        Path("outputs/vllm_throughput.png"),
    )
    print(f"График: {plot_path}")
    rates = dict(points)
    ceiling = float(sequential["throughput"])
    print(
        f"Потолок запросов по одному — {ceiling:.2f} запр/с. "
        f"Волна из 4 — {rates[4]:.2f} запр/с ({rates[4] / ceiling:.2f}x к потолку), "
        f"из 16 — {rates[16]:.2f} запр/с ({rates[16] / ceiling:.1f}x). "
        "На GPU урока vLLM от 0.27 запр/с при одном запросе доходил до 5.07 "
        "при шестнадцати. Здесь это статический батч transformers, не сервер "
        "vLLM: равные ответы он считает разом, состав батча по ходу генерации "
        "не меняет. Повторный запуск сдвинет секунды."
    )


def run_live_server() -> None:
    """Нагрузочный тест урока, если /v1/models уже отвечает."""
    client = make_vllm_client(async_client=True)

    async def scenario():
        for concurrency in VLLM_CONCURRENCIES:
            result = await load_test_vllm(concurrency, client)
            print_burst("vLLM", result)

    asyncio.run(scenario())


if __name__ == "__main__":
    main()
