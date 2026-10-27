"""Пример: смотрим архитектуру distilgpt2 и сравниваем её с gpt2.

Causal-модель предсказывает следующий токен только по уже написанному слева
тексту. distilgpt2 меньше 100 миллионов параметров, поэтому её можно
разбирать на CPU. По качеству она не соперничает с большими LLM: здесь важно
увидеть знакомые части и понять, чего стоит запуск.

Как читать ``print(model)`` сверху вниз:

- ``wte`` превращает ``token_id`` в вектор. У distilgpt2 словарь — 50257
  токенов, длина вектора — 768.
- ``wpe`` добавляет позицию токена. Позиций до 1024, вектор тоже длины 768.
- ``drop`` — dropout после сложения этих двух векторов. В ``model.eval()``
  он выключен.
- ``h`` — шесть блоков трансформера, запись ``(0-5): 6 x GPT2Block``.
  Внутри блока: LayerNorm, attention, остаточное сложение, снова LayerNorm,
  MLP и ещё одно остаточное сложение. Остаточные связи в ``print(model)``
  не видны: это сложение внутри прямого прохода, а не отдельный слой.
- ``c_attn`` с выходом 2304 — это не свёртка из компьютерного зрения.
  В GPT-2 так названа линейная проекция: 2304 = 3 × 768, сразу Q, K и V.
- ``mlp`` обрабатывает каждый токен отдельно: 768 → 3072 → 768.
- ``lm_head`` переводит финальный вектор длины 768 в оценки по всему словарю,
  то есть в 50257 чисел.

Число параметров — ориентир для памяти и скорости, не оценка качества.
Один параметр в float32 занимает 4 байта, в float16 — 2, в int8 — 1.
У distilgpt2 это примерно 81.91M параметров и 312.47 MB в float32,
156.24 MB в float16 и 78.12 MB в int8. Квантование экономит память и может
ухудшить качество. Оценка не включает активации, KV-cache и сам фреймворк.

Сравнение в примере: distilgpt2 и gpt2, генерация на 10 и на 30 новых
токенов. gpt2 крупнее, поэтому весит больше и на CPU обычно медленнее.
Переход с 10 токенов на 30 увеличивает общее время: шагов генерации больше.
Если память ограничена, разумный выбор — distilgpt2. Брать модель только
по числу параметров нельзя: важны ещё качество на задаче, скорость и
доступные ресурсы.

Запуск из корня проекта:

    python example_models.py
"""

import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    sys.exit(
        "Для этого примера нужен torch: модель distilgpt2 хранит веса в нём.\n"
        "Поставьте зависимости из requirements.txt: на этом Intel Mac это torch==2.2.2."
    )

import pandas as pd

from utils.models import (
    architecture_summary,
    count_parameters,
    estimate_memory_mb,
    estimate_model_weights_memory_mb,
    generate_continuation,
    load_causal_model,
    measure_generation_speed,
)

MODELS_TO_COMPARE = ["distilgpt2", "gpt2"]
MAX_NEW_TOKENS_VALUES = [10, 30]
PROMPT = "В этом уроке внимание помогает модели"


def main() -> None:
    print("1. Структура distilgpt2 и числа из конфига")
    tokenizer, model = load_causal_model("distilgpt2")
    print(model)
    summary = architecture_summary(model)
    print("Количество блоков трансформера:", summary["n_layer"])
    print("Количество голов attention:", summary["n_head"])
    print("Размерность скрытого представления:", summary["n_embd"])
    print("Размер словаря:", summary["vocab_size"])
    print("Позиций в wpe:", summary["n_positions"])

    print("\n2. Параметры и память под веса")
    num_params = count_parameters(model)
    first_param = next(model.parameters())
    print(f"Количество параметров: {num_params:,}")
    print(f"Количество параметров в миллионах: {num_params / 1_000_000:.2f}M")
    print(first_param.dtype)
    print(first_param.element_size(), "байт на параметр")
    print(
        "Память под веса в текущем формате: "
        f"{estimate_model_weights_memory_mb(model):.2f} MB"
    )
    for name, nbytes in (("float32", 4), ("float16", 2), ("int8", 1)):
        print(f"{name}: {estimate_memory_mb(num_params, nbytes):.2f} MB")

    print("\n3. Короткое продолжение, выбор следующего токена жадный")
    print(generate_continuation(model, tokenizer, PROMPT, max_new_tokens=30))

    print("\n4. distilgpt2 и gpt2: 10 и 30 новых токенов")
    results = []
    for model_name in MODELS_TO_COMPARE:
        print(f"Analyzing {model_name}...")
        model_tokenizer, causal_model = load_causal_model(model_name)
        model_params = count_parameters(causal_model)
        memory_fp16_mb = estimate_memory_mb(model_params, bytes_per_param=2)
        for max_new_tokens in MAX_NEW_TOKENS_VALUES:
            speed_stats = measure_generation_speed(
                model=causal_model,
                tokenizer=model_tokenizer,
                prompt=PROMPT,
                max_new_tokens=max_new_tokens,
                num_runs=3,
            )
            results.append(
                {
                    "model_name": model_name,
                    "max_new_tokens": max_new_tokens,
                    "num_params_m": model_params / 1_000_000,
                    "memory_fp16_mb": memory_fp16_mb,
                    "generation_time_sec": speed_stats["generation_time_sec"],
                    "tokens_per_sec": speed_stats["tokens_per_sec"],
                }
            )

    df = pd.DataFrame(results)
    df["num_params_m"] = df["num_params_m"].round(2)
    df["memory_fp16_mb"] = df["memory_fp16_mb"].round(2)
    df["generation_time_sec"] = df["generation_time_sec"].round(3)
    df["tokens_per_sec"] = df["tokens_per_sec"].round(2)
    print(df.to_string(index=False))

    largest = df.loc[df["num_params_m"].idxmax()]
    at_30 = df[df["max_new_tokens"] == 30]
    fastest = at_30.loc[at_30["tokens_per_sec"].idxmax()]
    print()
    print(f"Больше параметров и памяти в float16: {largest['model_name']}.")
    print(f"Быстрее на 30 новых токенах: {fastest['model_name']}.")
    for model_name, group in df.groupby("model_name"):
        short = group.loc[group["max_new_tokens"] == 10, "generation_time_sec"].item()
        long = group.loc[group["max_new_tokens"] == 30, "generation_time_sec"].item()
        print(
            f"{model_name}: время с 10 до 30 токенов меняется "
            f"с {short:.3f} до {long:.3f} сек."
        )
    print(
        "Если память ограничена, берут меньшую модель: она компактнее и "
        "обычно быстрее. Число параметров само по себе качество не обещает."
    )


if __name__ == "__main__":
    main()
