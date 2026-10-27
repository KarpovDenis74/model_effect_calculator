from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase

LLAMA_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@lru_cache(maxsize=1)
def get_hf_tokenizer(
    model_name: str = LLAMA_MODEL_NAME,
) -> PreTrainedTokenizerBase:
    """Загружает токенизатор LLaMA-подобной модели с Hugging Face.

    Повторный вызов с тем же именем модели возвращает уже загруженный объект.

    Args:
        model_name: Имя модели на Hugging Face.

    Returns:
        Токенизатор, которым можно посчитать токены текста.
    """
    return AutoTokenizer.from_pretrained(model_name)


def count_hf_tokens(text: str, model_name: str = LLAMA_MODEL_NAME) -> int:
    """Считает токены текста токенизатором LLaMA-подобной модели.

    Служебные токены вроде начала последовательности не добавляются, чтобы
    число можно было сравнивать с ``count_tokens`` для GPT: там tiktoken тоже
    считает только сам текст.

    Args:
        text: Исходный текст.
        model_name: Имя модели на Hugging Face.

    Returns:
        Сколько токенов получится, если отправить текст этой модели.
    """
    tokenizer = get_hf_tokenizer(model_name)
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded["input_ids"])


def show_hf_tokens(
    text: str,
    model_name: str = LLAMA_MODEL_NAME,
    add_special_tokens: bool = False,
) -> list[str]:
    """Показывает, как LLaMA-подобный токенизатор разрезал текст.

    Символ ``▁`` в начале фрагмента отмечает пробел или границу слова.
    Записи вида ``<0xE2>`` — byte fallback: символа не было в словаре,
    и токенизатор сохранил его байтами UTF-8.

    По умолчанию служебные токены вроде ``<s>`` не добавляются. С
    ``add_special_tokens=True`` токенизатор применяет обычный шаблон входа
    модели. У TinyLlama это начало последовательности, а не все токены
    из ``special_tokens_map``.

    Args:
        text: Исходный текст.
        model_name: Имя модели на Hugging Face.
        add_special_tokens: Добавлять ли служебные токены шаблона модели.

    Returns:
        Фрагменты текста в том порядке, в котором модель их читает.
    """
    tokenizer = get_hf_tokenizer(model_name)
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    return tokenizer.convert_ids_to_tokens(encoded["input_ids"])
