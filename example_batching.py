"""Пример: почему один запрос недогружает GPU и чем continuous batching
отличается от static.

Latency — время одного ответа, его чувствует пользователь. Throughput —
сколько запросов система успевает за секунду. Батч поднимает throughput
и часто увеличивает latency: запрос ждёт соседей.

Static batching собирает пачку и держит её до конца самого длинного
ответа. Короткие запросы после EOS продолжают занимать слот. Continuous
batching сразу ставит в освободившийся слот следующий запрос из очереди.
Выигрыш появляется, только если очередь длиннее батча и длинные запросы
приходят вперемешку с короткими.

Модель и GPU здесь не нужны: запрос задаётся числом шагов генерации.

Запуск из корня проекта:

    python example_batching.py
"""

from utils.inference import (
    compare_batching,
    format_timeline,
    fresh_skewed,
    fresh_skewed_interleaved,
    fresh_skewed_long,
    fresh_uniform,
    fresh_uniform_long,
    make_requests,
    simulate_continuous_batching,
    simulate_static_batching,
    static_batch_waste,
)


def section(title: str) -> None:
    print("=" * 80, flush=True)
    print(title, flush=True)
    print("=" * 80, flush=True)


def print_run(title: str, result: dict) -> None:
    print(
        f"{title:<12}{result['wall_steps']:>5} шагов, "
        f"утилизация {result['utilization']:.1%}"
    )


def main() -> None:
    section("1. Один запрос оставляет GPU полупустой")
    print(
        "GPU считает тысячи ядер параллельно. При батче размера 1 матрицы "
        "узкие, и загрузка около 30% — обычная картина, а не поломка. "
        "Размер модели и temperature на это не влияют. Чтобы занять ядра, "
        "в один проход нужно отправить несколько запросов."
    )
    print(
        "Пока запросов мало, ответы быстрые, но throughput низкий. "
        "Когда запросы собирают в батч, GPU занят плотнее, а каждый "
        "пользователь дополнительно ждёт соседей по пачке."
    )

    section("2. Static batching: короткие запросы ждут самого длинного")
    waste = static_batch_waste([50] * 7 + [2000])
    print(
        "Восемь запросов попали в один батч: семь ответов по 50 токенов "
        "и один на 2000. Батч живёт 2000 шагов, и на каждом шаге заняты "
        "все восемь слотов."
    )
    print(f"Полезная работа:     {waste['useful_slot_steps']} слото-шагов")
    print(f"Всего слото-шагов:   {waste['total_slot_steps']}")
    print(f"Впустую:             {waste['wasted_slot_steps']}")
    print(f"Доля простоя:        {waste['waste_share']:.0%}")

    section("3. Двенадцать запросов разной длины, батч из 4")
    print("Static. Точка — слот уже закончившегося запроса.")
    static_result = simulate_static_batching(make_requests(), batch_size=4)
    print(format_timeline(static_result["timeline"]))
    print()
    print(f"Шагов всего:       {static_result['wall_steps']}")
    print(f"Полезных слотов:   {static_result['useful_slot_steps']}")
    print(f"Всего слот-клеток: {static_result['total_slot_steps']}")
    print(f"Утилизация:        {static_result['utilization']:.1%}")

    print()
    print("Continuous. Новый запрос занимает слот сразу после EOS.")
    continuous_result = simulate_continuous_batching(make_requests(), batch_size=4)
    print(format_timeline(continuous_result["timeline"]))
    print()
    print(f"Шагов всего:  {continuous_result['wall_steps']}")
    print(f"Утилизация:   {continuous_result['utilization']:.1%}")
    print()
    print(
        f"Static:     {static_result['wall_steps']} шагов, "
        f"утилизация {static_result['utilization']:.1%}"
    )
    print(
        f"Continuous: {continuous_result['wall_steps']} шагов, "
        f"утилизация {continuous_result['utilization']:.1%}"
    )
    print(
        "У C в static после двух букв идут точки до конца батча. "
        "В continuous на его место сразу встаёт E. Простой остаётся "
        "только в хвосте, когда очередь уже пуста."
    )

    section("4. Очередь короче батча: выигрыша нет")
    print(
        "Суммарная работа одинаковая, 800 шагов. В однородной очереди "
        "все ответы по 100. В скошенной — семь по 25 и один на 625. "
        "Все восемь запросов входят в батч сразу, подставлять некого."
    )
    for title, factory in (
        ("Однородная", fresh_uniform),
        ("Скошенная", fresh_skewed),
    ):
        compared = compare_batching(factory, batch_size=8)
        print(f"\n{title} нагрузка (8 запросов):")
        print_run("Static:", compared["static"])
        print_run("Continuous:", compared["continuous"])
        print(f"Выигрыш continuous: {compared['speedup']:.2f}x")

    section("5. Длинная очередь, но короткие идут пачкой")
    print(
        "32 запроса, батч 8. В скошенной очереди сначала 28 коротких, "
        "потом 4 длинных. Слоты освобождаются одновременно и снова "
        "забирают одинаковых соседей, поэтому continuous не обгоняет static."
    )
    for title, factory in (
        ("Однородная", fresh_uniform_long),
        ("Скошенная", fresh_skewed_long),
    ):
        compared = compare_batching(factory, batch_size=8)
        print(f"\n{title} нагрузка (32 запроса):")
        print_run("Static:", compared["static"])
        print_run("Continuous:", compared["continuous"])
        print(f"Выигрыш:    {compared['speedup']:.2f}x")

    section("6. Длинные запросы стоят среди коротких")
    mixed = compare_batching(fresh_skewed_interleaved, batch_size=8)
    print("Скошенная с равномерным распределением (32 запроса):")
    print_run("Static:", mixed["static"])
    print_run("Continuous:", mixed["continuous"])
    print(f"Выигрыш:    {mixed['speedup']:.2f}x")
    print(
        "В static каждый батч тащит один длинный запрос, а семь коротких "
        "после EOS ещё сотни шагов занимают слоты. В continuous длинные "
        "остаются на своих местах, а через свободные слоты идут короткие. "
        "Утилизация не 100%: когда очередь кончается, хвост снова пустеет. "
        "Выигрыш набран раньше, пока в очереди ещё было кого подставить."
    )


if __name__ == "__main__":
    main()
