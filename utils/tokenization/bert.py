from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase

BERT_MODEL_NAME = "bert-base-multilingual-cased"


@lru_cache(maxsize=1)
def get_bert_tokenizer(
    model_name: str = BERT_MODEL_NAME,
) -> PreTrainedTokenizerBase:
    """Загружает BERT-токенизатор WordPiece.

    У BERT другие правила входа, чем у GPT. Токенизатор режет слова на
    фрагменты словаря и умеет добавлять служебные токены ``[CLS]`` и ``[SEP]``.

    Повторный вызов с тем же именем модели возвращает уже загруженный объект.

    Args:
        model_name: Имя модели на Hugging Face. По умолчанию
            ``bert-base-multilingual-cased``: она умеет и русский, и английский.

    Returns:
        Токенизатор, которым можно кодировать один текст или пару текстов.
    """
    return AutoTokenizer.from_pretrained(model_name)


def show_bert_tokens(
    text: str,
    add_special_tokens: bool = True,
    model_name: str = BERT_MODEL_NAME,
) -> list[str]:
    """Показывает WordPiece-токены BERT для одного текста.

    Префикс ``##`` не входит в исходный текст. Им WordPiece помечает фрагмент,
    который продолжает предыдущее слово: ``практику`` распадается на
    ``пра``, ``##кти``, ``##ку``.

    Args:
        text: Исходный текст.
        add_special_tokens: Добавлять ли ``[CLS]`` в начало и ``[SEP]`` в конец.
            ``[CLS]`` отмечает начало последовательности, ``[SEP]`` — её конец.
        model_name: Имя BERT-модели, чей токенизатор использовать.

    Returns:
        Токены в том порядке, в котором BERT читает текст.
    """
    tokenizer = get_bert_tokenizer(model_name)
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    return tokenizer.convert_ids_to_tokens(encoded["input_ids"])


def show_bert_pair_tokens(
    first_text: str,
    second_text: str,
    add_special_tokens: bool = True,
    model_name: str = BERT_MODEL_NAME,
) -> list[str]:
    """Показывает токены BERT для пары текстов.

    Между текстами ставится ``[SEP]``, и ещё один ``[SEP]`` закрывает пару.
    Так BERT отличает первый фрагмент от второго, например вопрос и контекст.

    Args:
        first_text: Первый текст пары.
        second_text: Второй текст пары.
        add_special_tokens: Добавлять ли ``[CLS]`` и ``[SEP]``.
        model_name: Имя BERT-модели, чей токенизатор использовать.

    Returns:
        Токены обоих текстов одной последовательностью.
    """
    tokenizer = get_bert_tokenizer(model_name)
    encoded = tokenizer(
        first_text,
        second_text,
        add_special_tokens=add_special_tokens,
    )
    return tokenizer.convert_ids_to_tokens(encoded["input_ids"])


if __name__ == "__main__":
    text = "Студенты любят практику"
    tokens_without_special = show_bert_tokens(text, add_special_tokens=False)
    tokens_with_special = show_bert_tokens(text, add_special_tokens=True)

    print("Без специальных токенов:")
    print(tokens_without_special)
    print(len(tokens_without_special))

    print("Со специальными токенами:")
    print(tokens_with_special)
    print(len(tokens_with_special))

    pair_tokens = show_bert_pair_tokens(
        "Студенты любят практику",
        "Токенизация помогает считать стоимость",
    )
    print(pair_tokens)
