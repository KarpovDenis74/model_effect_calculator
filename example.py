"""Пример: считаем токены и оцениваем, насколько текст дорогой для модели.

Модель не читает строку как человек. Сначала токенизатор режет её на фрагменты
из своего словаря и заменяет их числовыми id. Два текста одной длины в символах
могут занять разное число токенов: кириллица, код, идентификаторы и эмодзи часто
дробятся сильнее, чем обычный английский. Считать это нужно токенизатором той
модели, которой потом уйдёт запрос. Чужой токенизатор даст другую длину и другую цену.

В примере четыре шага.

1. GPT через tiktoken. ``count_tokens`` возвращает число id, ``show_tokens`` —
   сами фрагменты. Пробел и запятая тоже становятся токенами: это куски словаря,
   а не слова языка.
2. LLaMA-подобный токенизатор TinyLlama через Hugging Face. Скачивается только
   токенизатор, не веса модели. ``▁`` отмечает границу слова, ``<0xE2>`` — байт
   UTF-8, если готового токена не нашлось. ``<s>`` — отдельный служебный токен
   начала последовательности. Его добавляет шаблон входа, в исходном тексте его нет.
3. Одна и та же пачка текстов считается обоими токенизаторами. В столбце
   ``difference`` стоит ``llama_tokens - gpt_tokens``. Плюс значит, что TinyLlama
   нарезала эту строку мельче. Это сравнение двух конкретных токенизаторов на
   этих строках, а не вывод обо всех GPT- и LLaMA-моделях.
4. ``estimate_cost`` переводит токены в деньги. Цена одного запроса часто выглядит
   как доли цента, а на 10 000 запросов в день уже виден дневной бюджет. Если
   начало промпта повторяется, эту часть можно посчитать по цене кеша.

Отдельно показан BERT: у него другие служебные токены, ``[CLS]`` и ``[SEP]``.
Префикс ``##`` к ним не относится. Это пометка WordPiece, что фрагмент продолжает
предыдущее слово.

Запуск из корня проекта:

    python example.py
"""

import pandas as pd

from utils import (
    count_hf_tokens,
    count_tokens,
    estimate_cost,
    get_hf_tokenizer,
    show_bert_pair_tokens,
    show_bert_tokens,
    show_hf_tokens,
    show_tokens,
)
from utils.tokenization.gpt import texts


def compare_tokenizers(samples: dict[str, str]) -> pd.DataFrame:
    """Собирает таблицу символов и токенов GPT и LLaMA по одним и тем же текстам."""
    rows = []
    for text_type, text in samples.items():
        gpt_tokens = count_tokens(text)
        llama_tokens = count_hf_tokens(text)
        rows.append(
            {
                "text_type": text_type,
                "characters": len(text),
                "gpt_tokens": gpt_tokens,
                "llama_tokens": llama_tokens,
                "difference": llama_tokens - gpt_tokens,
            }
        )
    return pd.DataFrame(rows)


def budget_verdict(gpt_total_cost: float, llama_total_cost: float, budget: float) -> str:
    """Говорит, какая модель укладывается в бюджет на пачку запросов."""
    gpt_fits = gpt_total_cost <= budget
    llama_fits = llama_total_cost <= budget
    if gpt_fits and llama_fits:
        return "Обе модели проходят по бюджету"
    if gpt_fits:
        return "По бюджету проходит GPT"
    if llama_fits:
        return "По бюджету проходит LLaMA"
    return "Ни одна модель не проходит по бюджету"


def main() -> None:
    print("1. GPT видит текст как список id и фрагментов словаря")
    text = "Студенты любят практику и хотят понять, как LLM видит текст."
    print(count_tokens(text))
    print(show_tokens(text))
    print(show_tokens("Привет, мир!"))

    print("\n2. TinyLlama режет тот же смысл иначе, а эмодзи может уйти в байты")
    heart = "Студенты любят практику❤️"
    heart_tokens = show_hf_tokens(heart)
    print(heart_tokens)
    print(count_hf_tokens(heart))
    print("со служебным токеном:", show_hf_tokens(heart, add_special_tokens=True))
    print("что токенизатор считает служебным:", get_hf_tokenizer().special_tokens_map)
    restored = get_hf_tokenizer().decode(
        get_hf_tokenizer()(heart, add_special_tokens=False)["input_ids"]
    )
    print("текст собрался обратно:", restored)

    print("\n3. Одни и те же строки: где какой токенизатор дробит сильнее")
    comparison = compare_tokenizers(texts)
    print(comparison.to_string(index=False))

    print("\n4. Стоимость запроса, который уже токенизировали")
    gpt_input_tokens = 78
    llama_input_tokens = 112
    expected_output_tokens = 500
    requests_count = 10_000
    budget = 20

    gpt_cost = estimate_cost(
        input_tokens=gpt_input_tokens,
        output_tokens=expected_output_tokens,
        input_price_per_million=0.75,
        output_price_per_million=4.50,
    )
    llama_cost = estimate_cost(
        input_tokens=llama_input_tokens,
        output_tokens=expected_output_tokens,
        input_price_per_million=0.60,
        output_price_per_million=0.60,
    )
    gpt_total_cost = gpt_cost * requests_count
    llama_total_cost = llama_cost * requests_count
    print("GPT total cost:", gpt_total_cost)
    print("LLaMA total cost:", llama_total_cost)
    print(budget_verdict(gpt_total_cost, llama_total_cost, budget))

    print("\n5. Длинный промпт: без кеша и с повторяющимися 8 000 токенами")
    input_tokens = 12_000
    output_tokens = 1_500
    cached_tokens = 8_000
    price_in = 0.75
    price_cached = 0.075
    price_out = 4.50
    cost_without_cache = estimate_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_price_per_million=price_in,
        output_price_per_million=price_out,
    )
    cost_with_cache = estimate_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_price_per_million=price_in,
        output_price_per_million=price_out,
        cached_tokens=cached_tokens,
        cached_price_per_million=price_cached,
    )
    print(f"один запрос без кеша: {cost_without_cache:.5f}")
    print(f"10 000 запросов в день без кеша: {cost_without_cache * 10_000:.1f}")
    print(f"один запрос с кешем: {cost_with_cache:.5f}")
    print(f"10 000 запросов в день с кешем: {cost_with_cache * 10_000:.1f}")

    print("\n6. BERT добавляет [CLS] и [SEP], а ## только продолжает слово")
    bert_text = "Студенты любят практику"
    print(show_bert_tokens(bert_text, add_special_tokens=False))
    print(show_bert_tokens(bert_text))
    print(
        show_bert_pair_tokens(
            bert_text,
            "Токенизация помогает считать стоимость",
        )
    )


if __name__ == "__main__":
    main()
