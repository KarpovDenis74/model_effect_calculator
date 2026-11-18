"""Пример: flash-attention и квантизация весов.

Flash-attention не меняет формулу внимания. Он не пишет матрицу seq × seq
в медленную память видеокарты и за счёт этого быстрее на длинном
контексте. Квантизация уменьшает сами веса: float16 — два байта на
параметр, INT8 — один, INT4 — полбайта. За сжатие платят ошибкой
округления. На многошаговой математике она заметнее, чем на коротком
факте.

Урок грузит Qwen2.5-3B-Instruct дважды, во float16 и в NF4 через
bitsandbytes, и мерит память CUDA. Здесь нет NVIDIA-ядер, поэтому
flash-attn и bitsandbytes не запускаются. Память 3B считается по формуле,
а ошибка NF4 вносится в уже скачанную Qwen2.5-0.5B-Instruct: Linear-веса
округляются и сразу разворачиваются обратно во float. Скорость этого
прогона — не скорость упакованного INT4: умножения остаются float16.
На GPU урока упакованный INT4 занял 1.93 ГБ против 5.75 и оказался
медленнее на коротких запросах.

Запуск из корня проекта:

    python example_flash_quantization.py
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
    ATTENTION_DEMO_LENGTHS,
    COMPARISON_QUESTIONS,
    DEFAULT_SRAM_BYTES,
    EXAMPLE_WEIGHT,
    EXAMPLE_WEIGHT_HIGH,
    EXAMPLE_WEIGHT_LOW,
    FLASH_ATTENTION_IMPLEMENTATION,
    FLOAT16_BITS,
    INT4_BITS,
    INT8_BITS,
    LAPTOP_PRICE_RUB,
    LESSON_3B_PARAMETERS,
    LESSON_70B_PARAMETERS,
    LESSON_GPU_FP16_MEMORY_GB,
    LESSON_GPU_FP16_SECONDS,
    LESSON_GPU_FP16_TOKENS,
    LESSON_GPU_FP16_TPS,
    LESSON_GPU_INT4_MEMORY_GB,
    LESSON_GPU_INT4_SECONDS,
    LESSON_GPU_INT4_TOKENS,
    LESSON_GPU_INT4_TPS,
    LESSON_QUANT_MODEL,
    LESSON_ROUNDED_GB,
    NF4_CODEBOOK,
    OPTIMIZATION_CASES,
    QUALITY_QUIZ,
    attention_matrix_bytes,
    attention_matrix_elements,
    benchmark_chat,
    weight_memory_bytes,
    fits_in_sram,
    flash_attention_available,
    int4_kernels_available,
    laptop_price,
    load_model,
    mentions_amount,
    nearest_level,
    quantization_scale,
    quantize_uniform,
    round_linear_weights_to_nf4,
    weight_memory_gib,
    weight_storage_report,
)
from utils.models import (
    QWEN_INSTRUCT_MODEL,
    load_chat_model,
    release_chat_model,
)


def section(title: str) -> None:
    print("=" * 80, flush=True)
    print(title, flush=True)
    print("=" * 80, flush=True)


def gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def answer_note(index: int, text: str) -> str:
    if index == 0:
        folded = text.lower()
        found = "париж" in folded or "paris" in folded or "巴黎" in text
        return "столица названа" if found else "столица в ответе не найдена"
    if index == 1:
        found = "dog" in text.lower()
        return "перевод dog есть" if found else "перевода dog нет"
    if index == 2:
        if mentions_amount(text, LAPTOP_PRICE_RUB):
            return f"итоговая сумма {LAPTOP_PRICE_RUB} есть"
        steps = [
            label
            for label, amount in (("92 000", 92_000), ("69 000", 69_000))
            if mentions_amount(text, amount)
        ]
        if steps:
            return "итога 62 100 нет, по пути есть " + " и ".join(steps)
        return "итога 62 100 нет, промежуточных сумм нет"
    return "ответ непустой" if text.strip() else "пустой ответ"


def print_run(title: str, result: dict, memory_gb: float) -> None:
    print(
        f"{title:<22}{memory_gb:<12.2f}{result['seconds']:<12.2f}"
        f"{result['new_tokens']:<10}{result['tok_per_sec']:<8.1f}"
    )


def main() -> None:
    section("1. Обычное внимание упирается в медленную память")
    print(
        "У видеокарты две памяти. HBM — те самые гигабайты VRAM: там лежат "
        "веса, KV-cache и большие промежуточные матрицы, но для ядер она "
        "медленная. SRAM стоит на чипе рядом с ядрами и работает гораздо "
        "быстрее, зато её единицы или десятки мегабайт. Ядро считает только "
        "то, что уже лежит в SRAM. Пока данные едут из HBM, ядра ждут."
    )
    elements = attention_matrix_elements(8000)
    matrix_bytes = attention_matrix_bytes(8000)
    print(
        f"Обычный attention для 8000 токенов строит матрицу {elements} чисел, "
        f"это {gib(matrix_bytes) * 1024:.0f} МиБ во float16. В SRAM на "
        f"{DEFAULT_SRAM_BYTES // (1024 * 1024)} МиБ она не входит, поэтому "
        "её целиком держат в HBM, читают для softmax, пишут обратно и читают "
        "ещё раз для умножения на V. Перевозка занимает больше времени, чем "
        "сами умножения. Такой шаг называют memory-bound."
    )
    if elements != 64_000_000:
        raise SystemExit(f"Ожидалось 64 миллиона ячеек, получилось {elements}.")
    print(
        f"{'Длина':>8}{'Матрица, МиБ':>16}{'Влезает в 20 МиБ':>20}"
    )
    for length in ATTENTION_DEMO_LENGTHS:
        size_mib = attention_matrix_bytes(length) / (1024 * 1024)
        print(f"{length:>8}{size_mib:>16.2f}{str(fits_in_sram(length)):>20}")
    print(
        "На коротком запросе матрица маленькая, возить почти нечего. "
        "С нескольких тысяч токенов выигрыш flash-attention становится "
        "заметным и по времени, и по памяти."
    )

    section("2. Flash-attention считает то же самое другими поездками")
    print(
        "Flash-attention — не новая модель и не флаг качества. Это те же Q, K, "
        "V и тот же softmax, разложенные блоками по SRAM. Полная матрица в HBM "
        "не создаётся. Softmax по кускам строки собирается online softmax: "
        "ядро помнит текущий максимум и сумму и поправляет уже посчитанное, "
        "когда приезжает следующий блок. В медленную память уходит только "
        "готовый выход внимания."
    )
    print(
        "Ядра написала группа Три Дао, в ходу flash-attention 2. В transformers "
        f"их включает attn_implementation={FLASH_ATTENTION_IMPLEMENTATION!r} "
        "при загрузке модели. Пакет называется flash-attn."
    )
    if flash_attention_available():
        print("На этой машине ядра flash-attention доступны.")
    else:
        print(
            "На этой машине ядер нет: flash-attn собран под NVIDIA, CUDA здесь "
            "не видна. Формула внимания от этого не меняется, меняется только "
            "возможность включить быструю реализацию."
        )

    section("3. Сколько весят float16, INT8 и INT4")
    print(
        "Квантизация уменьшает не матрицу внимания, а сами веса. Float16 — "
        "два байта на параметр. INT8 — один байт, INT4 — полбайта. Перед "
        "умножением в режиме bitsandbytes блок весов разворачивают обратно "
        "во float16: арифметика остаётся обычной, это weight-only квантизация. "
        "Память выигрывает всегда. Скорость — нет: на коротких запросах "
        "разворот может оказаться дороже экономии на перевозке весов."
    )
    for bits, title in ((FLOAT16_BITS, "float16"), (INT8_BITS, "INT8"), (INT4_BITS, "INT4")):
        exact = weight_memory_gib(LESSON_3B_PARAMETERS, bits)
        print(
            f"{title:<8}{LESSON_3B_PARAMETERS} параметров × {bits} бит "
            f"= {exact:.2f} ГиБ. Урок округляет этот ряд до "
            f"{LESSON_ROUNDED_GB[bits]} ГБ."
        )
    seventy_gb = weight_memory_bytes(LESSON_70B_PARAMETERS, FLOAT16_BITS) / 1_000_000_000
    print(
        f"70 миллиардов параметров во float16 — {seventy_gb:.0f} ГБ, если считать "
        "гигабайт как 10⁹ байт. В одну видеокарту это не влезает."
    )
    scale = quantization_scale(EXAMPLE_WEIGHT_LOW, EXAMPLE_WEIGHT_HIGH, INT4_BITS)
    index, restored = quantize_uniform(
        EXAMPLE_WEIGHT,
        EXAMPLE_WEIGHT_LOW,
        EXAMPLE_WEIGHT_HIGH,
        INT4_BITS,
    )
    print(
        f"Равномерный INT4 на отрезке [{EXAMPLE_WEIGHT_LOW}, {EXAMPLE_WEIGHT_HIGH}]: "
        f"16 ступеней, 15 промежутков, шаг {scale:.3f}. Вес {EXAMPLE_WEIGHT} "
        f"попадает на ступень {index} и разворачивается в {restored:.4f}. "
        f"Зазор {restored - EXAMPLE_WEIGHT:.4f} — ошибка округления."
    )
    if abs(scale - 0.8 / 15) > 1e-12 or index != 8:
        raise SystemExit("Равномерная сетка разошлась с разбором урока.")
    uniform_gap = quantization_scale(-1.0, 1.0, INT4_BITS)
    zero_index, _zero = nearest_level(0.0, NF4_CODEBOOK)
    next_positive = NF4_CODEBOOK[zero_index + 1]
    print(
        "NF4 расставляет те же 16 уровней по нормальному распределению: гуще "
        f"около нуля. Следующий положительный уровень после нуля — {next_positive:.4f}, "
        f"а у равномерной сетки на [-1, 1] шаг {uniform_gap:.4f}. Края NF4 "
        f"реже: от {NF4_CODEBOOK[-2]:.4f} до {NF4_CODEBOOK[-1]:.1f}. "
        "Плюс свой scale на блок весов и double quant самих масштабов. "
        "Поэтому в конфиге урока стоят nf4 и bnb_4bit_use_double_quant=True."
    )

    section("4. Где качество просядет и какой метод выбрать")
    print(
        "INT8 обычно почти неотличим от float16. INT4 на коротком факте, "
        "переводе и одном предложении чаще держится рядом, а на цепочке "
        "процентов ошибается чаще. Решение принимают по замеру на своей задаче."
    )
    for choice in QUALITY_QUIZ:
        mark = "верно" if choice.correct else "неверно"
        print(f"- [{mark}] {choice.text}")
        print(f"  {choice.reason}")
    price = laptop_price()
    print(
        f"Эталон математической задачи: 80 000 × 1.15 = 92 000, "
        f"× 0.75 = 69 000, × 0.90 = {price} рублей."
    )
    if price != LAPTOP_PRICE_RUB:
        raise SystemExit(f"Ожидалась сумма {LAPTOP_PRICE_RUB}, получилось {price}.")
    print()
    print("Сценарии задания раскладываются так:")
    for case in OPTIMIZATION_CASES:
        print(f"- {case.method}: {case.scenario}")
        print(f"  {case.reason}")
    print(
        "GPTQ идёт по слоям и компенсирует ошибку одних весов правкой других, "
        "по небольшому калибровочному набору. AWQ по активациям находит "
        "маленькую долю важных весов и защищает их. Оба метода — post-training: "
        "модель заново не учат. Берут тот формат, который уже есть для модели "
        "и который умеет инференс-сервер."
    )

    section("5. Float16 и NF4 на локальной модели")
    memory_ratio = LESSON_GPU_FP16_MEMORY_GB / LESSON_GPU_INT4_MEMORY_GB
    print(
        f"Авторский замер, {LESSON_QUANT_MODEL} на GPU: float16 "
        f"{LESSON_GPU_FP16_MEMORY_GB:.2f} ГБ, {LESSON_GPU_FP16_SECONDS:.2f} с, "
        f"{LESSON_GPU_FP16_TOKENS} токенов, {LESSON_GPU_FP16_TPS:.1f} ток/с. "
        f"INT4 {LESSON_GPU_INT4_MEMORY_GB:.2f} ГБ, {LESSON_GPU_INT4_SECONDS:.2f} с, "
        f"{LESSON_GPU_INT4_TOKENS} токенов, {LESSON_GPU_INT4_TPS:.1f} ток/с. "
        f"По памяти INT4 меньше в {memory_ratio:.1f} раза, не в 4: эмбеддинги "
        "и масштабы блоков остаются шире четырёх бит. По времени INT4 медленнее. "
        "На коротких одиночных запросах разворот весов перед каждым умножением "
        "перевешивает экономию перевозки. Это ожидаемо, особенно на старых картах."
    )
    if int4_kernels_available():
        print(f"Ядра bitsandbytes есть. Гружу {LESSON_QUANT_MODEL} дважды, как в уроке.")
        run_cuda_comparison()
        return
    print(
        "Ядер bitsandbytes нет, Qwen2.5-3B не скачивается. Беру "
        f"{QWEN_INSTRUCT_MODEL}. Сначала ответы во float16. Затем Linear-веса "
        "округляются блоками NF4 по 64 числа, масштабы блоков ещё раз сжимаются "
        "в INT8, и восстановленные значения записываются обратно во float16. "
        "Второй прогон показывает качество после этой ошибки. Память процесса "
        "не уменьшается: упакованных 4-битных весов в генерации нет. "
        "Оценку хранения печатаю отдельно."
    )
    tokenizer, model = load_chat_model(QWEN_INSTRUCT_MODEL)
    try:
        report = weight_storage_report(model)
        print(
            f"Сейчас веса занимают {gib(report['float_bytes']):.2f} ГиБ, "
            f"из них Linear {gib(report['linear_bytes']):.2f} ГиБ. "
            f"Если упаковать только Linear в NF4 с double quant, оценка "
            f"{gib(report['nf4_bytes']):.2f} ГиБ на всю модель и "
            f"{gib(report['linear_nf4_bytes']):.2f} ГиБ на сами матрицы. "
            f"Вся модель меньше в {report['float_bytes'] / report['nf4_bytes']:.2f} раза, "
            f"матрицы — в {report['linear_bytes'] / report['linear_nf4_bytes']:.2f} раза."
        )
        # У 0.5B цепочка процентов длиннее, чем у 3B из урока. Потолок 256
        # обрезает float16 на последней скидке, и сравнение с NF4 становится
        # сравнением обрыва, а не суммы.
        float_result = benchmark_chat(
            model,
            tokenizer,
            COMPARISON_QUESTIONS,
            max_new_tokens=512,
            label="float16",
        )
        print("Округляю Linear-веса в NF4.", flush=True)
        rounded = round_linear_weights_to_nf4(model)
        print(
            f"Округлены {rounded['linears']} матриц, "
            f"{rounded['weight_numel']} параметров."
        )
        nf4_result = benchmark_chat(
            model,
            tokenizer,
            COMPARISON_QUESTIONS,
            max_new_tokens=512,
            label="NF4",
        )
    finally:
        release_chat_model(model)

    print()
    print(f"{'Режим':<22}{'Память, ГиБ':<12}{'Время, сек':<12}{'Токенов':<10}{'Ток/с':<8}")
    print_run("float16", float_result, gib(int(float_result["memory_bytes"])))
    print_run("NF4, те же float-веса", nf4_result, gib(int(nf4_result["memory_bytes"])))
    print(
        "Вторая строка памяти — это всё ещё float16-модель после округления, "
        "а не упакованный INT4. Оба прогона умножают float-веса, лишнего "
        f"разворота на каждом шаге нет. Здесь вышло {float_result['tok_per_sec']:.1f} "
        f"и {nf4_result['tok_per_sec']:.1f} ток/с: первый длинный прогон ещё "
        "прогревает MPS, поэтому по этим двум числам скорость квантизации "
        "не судят. На GPU урока веса лежат упакованными, разворот есть, и "
        "INT4 там медленнее: 19.6 ток/с против 25.5."
    )
    print()
    for index, (question, raw_float, raw_nf4) in enumerate(
        zip(COMPARISON_QUESTIONS, float_result["answers"], nf4_result["answers"])
    ):
        print(f"Вопрос: {question}")
        print(f"  float16: {raw_float}")
        print(f"  NF4:     {raw_nf4}")
        print(
            f"  проверка: float16 — {answer_note(index, str(raw_float))}; "
            f"NF4 — {answer_note(index, str(raw_nf4))}"
        )
        print()
    print(
        "Итог задачи — 62 100, и по пути обязаны появиться 92 000 и 69 000: "
        "последняя скидка — это 90% от 69 000, а не прибавка 6 900. Короткий "
        "факт и перевод обычно остаются. Предложение о сне может стать "
        "длиннее и начать добавлять лишние детали. 0.5B и так слабее 3B из "
        "урока, поэтому ошибка float16 здесь — свойство маленькой модели, "
        "а дополнительный развал цепочки после NF4 — уже цена округления."
    )


def run_cuda_comparison() -> None:
    """Авторский порядок: float16, освободить память, затем NF4 bitsandbytes."""
    import gc

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(LESSON_QUANT_MODEL)
    model = load_model(LESSON_QUANT_MODEL, use_int4=False)
    try:
        float_result = benchmark_chat(model, tokenizer, label="float16")
    finally:
        del model
        gc.collect()
        torch_empty_cache()
    model = load_model(LESSON_QUANT_MODEL, use_int4=True)
    try:
        int4_result = benchmark_chat(model, tokenizer, label="INT4")
    finally:
        del model
        gc.collect()
        torch_empty_cache()
    print(f"{'Режим':<10}{'Память, ГБ':<13}{'Время, сек':<12}{'Токенов':<10}{'Ток/с':<8}")
    print(
        f"{'float16':<10}{float_result['memory_bytes'] / (1024 ** 3):<13.2f}"
        f"{float_result['seconds']:<12.2f}{float_result['new_tokens']:<10}"
        f"{float_result['tok_per_sec']:<8.1f}"
    )
    print(
        f"{'INT4':<10}{int4_result['memory_bytes'] / (1024 ** 3):<13.2f}"
        f"{int4_result['seconds']:<12.2f}{int4_result['new_tokens']:<10}"
        f"{int4_result['tok_per_sec']:<8.1f}"
    )
    print(
        "INT4 меньше по памяти в "
        f"{float_result['memory_bytes'] / int4_result['memory_bytes']:.1f}x"
    )
    for question, raw_float, raw_int4 in zip(
        COMPARISON_QUESTIONS,
        float_result["answers"],
        int4_result["answers"],
    ):
        print(f"Вопрос: {question}")
        print(f"  float16: {raw_float}")
        print(f"  INT4:    {raw_int4}")
        print()


def torch_empty_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
