import tiktoken


def get_openai_encoding(model_name: str):
    """Возвращает кодировку tiktoken для модели OpenAI.

    Если tiktoken ещё не знает имя модели, используется ``o200k_base`` —
    кодировка GPT-4o и более новых GPT.

    Args:
        model_name: Имя модели, по которому tiktoken выбирает кодировку.

    Returns:
        Объект кодировки, которым можно кодировать и декодировать текст.
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, model_name: str = "gpt-5") -> int:
    """Считает число токенов в тексте.

    Args:
        text: Исходный текст.
        model_name: Имя модели. Неизвестные имена считаются как ``o200k_base``.

    Returns:
        Сколько токенов получится, если отправить текст этой модели.
    """
    encoding = get_openai_encoding(model_name)
    token_ids = encoding.encode(text)
    return len(token_ids)


def show_tokens(text: str, model_name: str = "gpt-5") -> list[str]:
    """Показывает фрагменты, на которые токенизатор разбил текст.

    Пробелы, знаки препинания и части слов тоже становятся отдельными
    токенами: словарь токенизатора хранит не слова языка, а свои куски.

    Args:
        text: Исходный текст.
        model_name: Имя модели. Неизвестные имена считаются как ``o200k_base``.

    Returns:
        Фрагменты текста в том порядке, в котором модель их читает.
    """
    encoding = get_openai_encoding(model_name)
    token_ids = encoding.encode(text)
    return [encoding.decode([token_id]) for token_id in token_ids]


def encode_text(text: str, model_name: str = "gpt-5") -> list[int]:
    """Кодирует текст в идентификаторы токенов.

    Args:
        text: Исходный текст.
        model_name: Имя модели, по которому выбирается кодировка.

    Returns:
        Идентификаторы токенов. Их количество считает ``count_tokens``.
    """
    encoding = get_openai_encoding(model_name)
    return encoding.encode(text)


texts = {
    "простой": "Токенизация помогает заранее оценить размер контекста перед запросом.",
    "сложный": "Токенизация помогает заранее оценить размер контекста.",
    "смешанный": "Пользователь попросил модель summarize 3 JSON файла.",
    "код": "def calculate_price(price, discount): return price * (1 - discount)",
    "числа": "order_id=984120, sku_id=100239991, date=2026-05-10",
    "эмодзи": "🛠 ℹ ⚖ 📐 🔑 🏧.",
}


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import pandas as pd

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from utils.tokenization.llama import count_hf_tokens

    rows = []

    for text_type, text in texts.items():
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

    comparison_df = pd.DataFrame(rows)

    print(comparison_df)
