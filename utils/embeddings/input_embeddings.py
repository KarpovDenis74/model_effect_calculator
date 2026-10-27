"""Входные эмбеддинги: строка таблицы по token_id, ещё без контекста предложения."""

from functools import lru_cache

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

MODEL_NAME = "cointegrated/rubert-tiny2"


@lru_cache(maxsize=1)
def load_embedding_model(
    model_name: str = MODEL_NAME,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Загружает токенизатор и модель с таблицей входных эмбеддингов.

    Берётся небольшая русскоязычная модель: на русских примерах слова реже
    распадаются на отдельные буквы. Режим ``eval`` выключает обучение.
    Повторный вызов с тем же именем возвращает уже загруженные объекты.

    Args:
        model_name: Имя модели на Hugging Face.

    Returns:
        Токенизатор и сама модель.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def get_embedding_matrix(model_name: str = MODEL_NAME) -> torch.Tensor:
    """Возвращает таблицу входных эмбеддингов формы ``[vocab_size, d_model]``.

    Первая ось — число токенов в словаре, вторая — длина вектора одного токена.
    ``token_id`` — это номер строки в этой таблице. Градиент отсоединён:
    таблицу здесь читают, а не обучают.

    Args:
        model_name: Имя модели на Hugging Face.

    Returns:
        Матрица весов слоя входных эмбеддингов.
    """
    _, model = load_embedding_model(model_name)
    return model.get_input_embeddings().weight.detach()


def tokenize_without_special(
    text: str,
    model_name: str = MODEL_NAME,
) -> tuple[list[str], list[int]]:
    """Разбивает текст на токены и их id без служебных токенов модели.

    Args:
        text: Исходный текст.
        model_name: Имя модели на Hugging Face.

    Returns:
        Список фрагментов и список их id в том же порядке.
    """
    tokenizer, _ = load_embedding_model(model_name)
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return tokens, list(token_ids)


def get_single_token_id(token: str, model_name: str = MODEL_NAME) -> int:
    """Возвращает id слова, если токенизатор не разбивает его на части.

    Одна строка таблицы есть только у целого токена. Если слово распалось
    на несколько фрагментов, у него нет одного input embedding.

    Args:
        token: Слово, которое должно кодироваться одним токеном.
        model_name: Имя модели на Hugging Face.

    Returns:
        Идентификатор единственного токена.

    Raises:
        ValueError: слово кодируется несколькими токенами.
    """
    token_parts, token_ids = tokenize_without_special(token, model_name)
    if len(token_ids) != 1:
        raise ValueError(
            f"Слово {token!r} разбилось на несколько токенов: {token_parts}"
        )
    return token_ids[0]


def get_token_embedding(token: str, model_name: str = MODEL_NAME) -> torch.Tensor:
    """Достаёт input embedding одного токена — строку таблицы по его id.

    Вектор не зависит от предложения: слой эмбеддингов смотрит только на id.

    Args:
        token: Слово из одного токена.
        model_name: Имя модели на Hugging Face.

    Returns:
        Вектор длины ``d_model``.
    """
    matrix = get_embedding_matrix(model_name)
    return matrix[get_single_token_id(token, model_name)]


def same_input_embedding(
    token: str,
    first_text: str,
    second_text: str,
    model_name: str = MODEL_NAME,
) -> bool:
    """Проверяет, что в двух текстах у токена одна и та же строка таблицы.

    «Замок» как здание и «замок» как механизм получают один входной вектор.
    Различать значения модель начинает позже, в слоях трансформера.

    Args:
        token: Токен, который есть в обоих текстах.
        first_text: Первое предложение.
        second_text: Второе предложение.
        model_name: Имя модели на Hugging Face.

    Returns:
        ``True``, если векторы совпадают.
    """
    matrix = get_embedding_matrix(model_name)
    first_tokens, first_ids = tokenize_without_special(first_text, model_name)
    second_tokens, second_ids = tokenize_without_special(second_text, model_name)
    first_embedding = matrix[first_ids[first_tokens.index(token)]]
    second_embedding = matrix[second_ids[second_tokens.index(token)]]
    return bool(torch.allclose(first_embedding, second_embedding))


def normalize_embeddings(embedding_matrix: torch.Tensor) -> torch.Tensor:
    """Приводит каждый вектор таблицы к длине 1.

    После этого скалярное произведение совпадает с косинусным сходством:
    сравниваются направления, а не длины векторов. Форма матрицы не меняется.

    Args:
        embedding_matrix: Таблица входных эмбеддингов.

    Returns:
        Таблица той же формы с нормированными строками.
    """
    return F.normalize(embedding_matrix, p=2, dim=1)


def find_nearest_tokens(
    target_token: str,
    top_k: int = 10,
    model_name: str = MODEL_NAME,
) -> list[tuple[str, float]]:
    """Ищет ближайшие токены по косинусному сходству входных эмбеддингов.

    Сам токен и служебные токены вроде ``[CLS]``, ``[SEP]``, ``[PAD]`` из
    выдачи убираются. Соседи — не готовый список синонимов: близость возникает
    из форм слова, общей темы или похожих контекстов, которые модель видела
    при обучении. Контекст конкретного предложения сюда ещё не входит.

    Args:
        target_token: Слово из одного токена.
        top_k: Сколько соседей вернуть.
        model_name: Имя модели на Hugging Face.

    Returns:
        Пары ``(токен, сходство)`` от большего сходства к меньшему.
        Сходство округлено до 4 знаков.
    """
    tokenizer, _ = load_embedding_model(model_name)
    normalized = normalize_embeddings(get_embedding_matrix(model_name))

    target_id = get_single_token_id(target_token, model_name)
    similarities = (normalized @ normalized[target_id]).clone()
    similarities[target_id] = -1
    for special_token_id in tokenizer.all_special_ids:
        similarities[special_token_id] = -1

    top_scores, top_ids = torch.topk(similarities, k=top_k)
    nearest_tokens = []
    for score, token_id in zip(top_scores, top_ids):
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        nearest_tokens.append((token, round(float(score), 4)))
    return nearest_tokens
