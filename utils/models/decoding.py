"""Управление кандидатами после логитов: top-k, top-p и штрафы за повтор.

Обычный порядок вызовов совпадает с шагами генерации:

исходные логиты
    → apply_repetition_penalty или apply_frequency_penalty
    → apply_temperature
    → apply_top_k, затем apply_top_p
    → choose_greedy_token или sample_tokens

Температура меняет форму распределения и не выбрасывает токены.
top-k и top-p уже режут множество кандидатов: исключённым логитам
ставится ``-inf``. Штрафы смотрят на историю и только уменьшают оценки
уже встреченных токенов.
"""

from __future__ import annotations

import math

import torch
import pandas as pd


DEMO_TOKENS = [
    " with",
    " you",
    " First",
    " Let's",
    " To",
    " take",
    " go",
    " stay",
    " read",
    " DNS",
]


def apply_top_k(
    logits: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Оставляет только top-k токенов с максимальными логитами.

    Остальным токенам присваивается ``-inf``, чтобы после softmax их
    вероятность стала нулевой. Исходный тензор не изменяется.

    Args:
        logits: Логиты токенов, тензор формы ``[vocab_size]``.
        top_k: Число токенов, которые нужно оставить.

    Returns:
        Копия логитов, в которой исключённые токены имеют значение ``-inf``.
    """
    if logits.ndim != 1:
        raise ValueError("logits must be a one-dimensional tensor")

    if not 1 <= top_k <= logits.numel():
        raise ValueError("top_k must be between 1 and the vocabulary size")

    processed_logits = torch.full_like(logits, fill_value=float("-inf"))
    top_token_ids = torch.topk(logits, k=top_k).indices
    processed_logits[top_token_ids] = logits[top_token_ids]
    return processed_logits


def apply_top_p(
    logits: torch.Tensor,
    top_p: float,
) -> torch.Tensor:
    """Оставляет минимальный набор токенов с массой не меньше top-p.

    Токены сортируются по убыванию вероятности. Накопленная сумма растёт,
    пока не достигнет ``top_p``. Первый токен, который пересёк порог,
    остаётся в множестве, всё после него исключается через ``-inf``.
    Порядок токенов в результате совпадает с исходным.

    Args:
        logits: Логиты токенов, тензор формы ``[vocab_size]``.
        top_p: Доля вероятностной массы, которую нужно сохранить.

    Returns:
        Копия логитов, в которой исключённые токены имеют значение ``-inf``.
    """
    if logits.ndim != 1:
        raise ValueError("logits must be a one-dimensional tensor")

    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")

    sorted_logits, sorted_token_ids = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False

    processed_logits = logits.clone()
    token_ids_to_remove = sorted_token_ids[sorted_indices_to_remove]
    processed_logits[token_ids_to_remove] = float("-inf")
    return processed_logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    seen_token_ids: set[int],
    penalty: float,
) -> torch.Tensor:
    """Применяет штраф к уже встречавшимся токенам.

    Положительный логит делится на ``penalty``, отрицательный умножается.
    В обоих случаях оценка становится меньше, и повторный выбор менее вероятен.
    Токены вне ``seen_token_ids`` не меняются. Исходный тензор не изменяется.

    Args:
        logits: Логиты токенов, тензор формы ``[vocab_size]``.
        seen_token_ids: Идентификаторы уже встречавшихся токенов.
        penalty: Значение repetition penalty. ``1.0`` ничего не меняет.

    Returns:
        Копия логитов после применения штрафа.
    """
    if logits.ndim != 1:
        raise ValueError("logits must be a one-dimensional tensor")

    if penalty <= 0:
        raise ValueError("penalty must be greater than 0")

    vocab_size = logits.numel()
    invalid_token_ids = {
        token_id
        for token_id in seen_token_ids
        if not 0 <= token_id < vocab_size
    }
    if invalid_token_ids:
        raise ValueError(f"Unknown token IDs: {sorted(invalid_token_ids)}")

    processed_logits = logits.clone()
    for token_id in seen_token_ids:
        token_logit = processed_logits[token_id]
        if token_logit < 0:
            processed_logits[token_id] = token_logit * penalty
        else:
            processed_logits[token_id] = token_logit / penalty
    return processed_logits


def apply_temperature(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Делит логиты на температуру и возвращает новый тензор.

    Температуру применяют до top-k и top-p. Она не меняет порядок
    положительных логитов: самый большой остаётся самым большим.
    Значение меньше 1 делает распределение острее, больше 1 — ровнее.
    Поэтому один и тот же top-p при низкой температуре оставляет мало
    кандидатов, а при высокой — много.

    Args:
        logits: Логиты токенов, тензор формы ``[vocab_size]``.
        temperature: Положительная температура. ``1.0`` оставляет логиты
            как есть.

    Returns:
        Новые логиты. Исходный тензор не изменяется.
    """
    if logits.ndim != 1:
        raise ValueError("logits must be a one-dimensional tensor")

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than 0")

    return logits / temperature


def apply_frequency_penalty(
    logits: torch.Tensor,
    token_counts: dict[int, int],
    alpha: float,
) -> torch.Tensor:
    """Снижает логит пропорционально числу предыдущих появлений токена.

    Форма из урока: ``z' = z - alpha * count``. Первый повтор получает
    штраф ``alpha``, пятый — ``5 * alpha``. ``repetition_penalty`` так
    не делает: ему важен только факт появления, и штраф каждый раз один.

    Точная формула в чужом API может отличаться. Здесь реализован
    вариант, которым урок объясняет разницу двух штрафов.

    Args:
        logits: Логиты токенов, тензор формы ``[vocab_size]``.
        token_counts: Сколько раз каждый id уже встречался.
        alpha: Сила частотного штрафа. ``0`` ничего не меняет.

    Returns:
        Копия логитов после штрафа.
    """
    if logits.ndim != 1:
        raise ValueError("logits must be a one-dimensional tensor")

    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and greater than or equal to 0")

    vocab_size = logits.numel()
    invalid_token_ids = sorted(
        token_id
        for token_id in token_counts
        if not 0 <= token_id < vocab_size
    )
    if invalid_token_ids:
        raise ValueError(f"Unknown token IDs: {invalid_token_ids}")

    negative_counts = sorted(
        token_id for token_id, count in token_counts.items() if count < 0
    )
    if negative_counts:
        raise ValueError(
            f"token counts must be non-negative, got ids {negative_counts}"
        )

    processed_logits = logits.clone()
    if alpha == 0:
        return processed_logits

    for token_id, count in token_counts.items():
        if count == 0:
            continue
        processed_logits[token_id] = processed_logits[token_id] - alpha * count

    return processed_logits


def logits_from_probabilities(
    probabilities: list[float] | torch.Tensor,
) -> torch.Tensor:
    """Переводит готовую таблицу вероятностей в логиты.

    Учебные распределения урока заданы вероятностями, а фильтры работают
    с логитами. ``softmax(log(p))`` возвращает исходные ``p``, если они
    положительны и в сумме дают 1.

    Args:
        probabilities: Вероятности токенов в исходном порядке.

    Returns:
        Логиты той же длины.
    """
    probs = torch.as_tensor(probabilities, dtype=torch.float32)
    if probs.ndim != 1 or probs.numel() == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional tensor")

    if not torch.isfinite(probs).all() or torch.any(probs <= 0):
        raise ValueError("probabilities must be finite and positive")

    if not torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4):
        raise ValueError("probabilities must sum to 1")

    return torch.log(probs)


def describe_logits(
    logits: torch.Tensor,
    tokens: list[str],
) -> list[dict[str, object]]:
    """Описывает текущие логиты: оценка, вероятность и признак «токен оставлен».

    Вероятность считается softmax-ом по конечным логитам. После top-k или
    top-p это уже нормализованное распределение среди оставшихся кандидатов:
    у исключённых токенов вероятность 0, у оставшихся сумма равна 1.

    Args:
        logits: Логиты токенов, тензор формы ``[vocab_size]``.
        tokens: Имена токенов в том же порядке, что и логиты.

    Returns:
        Список словарей с ключами ``token``, ``logit``, ``probability``,
        ``kept``.
    """
    if logits.ndim != 1:
        raise ValueError("logits must be a one-dimensional tensor")

    if len(tokens) != logits.numel():
        raise ValueError("tokens and logits must have the same length")

    kept = torch.isfinite(logits)
    if not bool(kept.any()):
        raise ValueError("at least one logit must be finite")

    probabilities = torch.softmax(logits, dim=-1)
    rows: list[dict[str, object]] = []
    for index, token in enumerate(tokens):
        is_kept = bool(kept[index].item())
        rows.append(
            {
                "token": token,
                "logit": float(logits[index].item()) if is_kept else float("-inf"),
                "probability": float(probabilities[index].item()),
                "kept": is_kept,
            }
        )
    return rows


def kept_probability_mass(
    original_logits: torch.Tensor,
    processed_logits: torch.Tensor,
) -> float:
    """Суммирует исходные вероятности токенов, прошедших фильтр.

    Это масса до повторной нормализации. У top-p она не меньше порога:
    в набор входит и тот токен, на котором накопленная вероятность
    впервые его пересекла.

    Args:
        original_logits: Логиты до фильтрации.
        processed_logits: Логиты после фильтра. Исключённые равны ``-inf``.

    Returns:
        Доля исходного распределения, которая осталась в кандидатах.
    """
    if original_logits.shape != processed_logits.shape:
        raise ValueError("logits must have the same shape")

    original_probs = torch.softmax(original_logits, dim=-1)
    kept = torch.isfinite(processed_logits)
    return float(original_probs[kept].sum().item())


def kept_token_ids(logits: torch.Tensor) -> list[int]:
    """Возвращает id токенов с конечным логитом, от большего логита к меньшему."""
    if logits.ndim != 1:
        raise ValueError("logits must be a one-dimensional tensor")

    finite_ids = torch.nonzero(torch.isfinite(logits), as_tuple=False).flatten()
    if finite_ids.numel() == 0:
        return []

    order = torch.argsort(logits[finite_ids], descending=True)
    return [int(token_id) for token_id in finite_ids[order]]


def kept_names(logits: torch.Tensor, tokens: list[str]) -> list[str]:
    """Возвращает имена оставшихся токенов от большего логита к меньшему."""
    if len(tokens) != logits.numel():
        raise ValueError("tokens and logits must have the same length")

    return [tokens[token_id] for token_id in kept_token_ids(logits)]


def probability_rows(
    tokens: list[str],
    original_logits: torch.Tensor,
    processed_logits: torch.Tensor,
) -> list[dict[str, object]]:
    """Собирает вероятность до фильтра и после нормализации оставшихся.

    Args:
        tokens: Имена токенов в том же порядке, что и логиты.
        original_logits: Логиты до фильтрации.
        processed_logits: Логиты после фильтра. Исключённые равны ``-inf``.

    Returns:
        Строки с ключами ``token``, ``probability_before``,
        ``probability_after`` и ``kept``.
    """
    before = describe_logits(original_logits, tokens)
    after = describe_logits(processed_logits, tokens)
    rows: list[dict[str, object]] = []
    for left, right in zip(before, after):
        rows.append(
            {
                "token": left["token"],
                "probability_before": left["probability"],
                "probability_after": right["probability"],
                "kept": right["kept"],
            }
        )
    return rows


def choose_greedy_token(logits: torch.Tensor) -> int:
    """Возвращает идентификатор токена с максимальным логитом."""
    return int(torch.argmax(logits).item())


def sample_tokens(
    logits: torch.Tensor,
    seeds: list[int],
) -> list[int]:
    """Выбирает по одному токену для каждого seed.

    Args:
        logits: Логиты после обработки.
        seeds: Список seed для воспроизводимых запусков.

    Returns:
        Список выбранных идентификаторов токенов.
    """
    probabilities = torch.softmax(logits, dim=-1)
    sampled_token_ids = []

    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        token_id = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        ).item()
        sampled_token_ids.append(token_id)

    return sampled_token_ids


def decode_token_ids(
    token_ids: list[int],
    tokens: list[str] | None = None,
) -> list[str]:
    """Преобразует идентификаторы токенов в строки учебного словаря."""
    vocabulary = DEMO_TOKENS if tokens is None else tokens
    return [vocabulary[token_id] for token_id in token_ids]


def generate_greedy_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    repetition_penalty: float = 1.0,
    max_new_tokens: int = 35,
) -> str:
    """Генерирует продолжение через greedy decoding со штрафом за повторы.

    ``do_sample=False`` каждый раз выбирает токен с максимальной оценкой.
    ``repetition_penalty`` меняет логиты уже встречавшихся токенов до выбора.
    Значение ``1.0`` ничего не штрафует. Функция возвращает только новые
    токены после промпта.

    Args:
        model: Causal-модель в режиме ``eval``.
        tokenizer: Токенизатор этой модели.
        prompt: Исходный текст.
        repetition_penalty: Штраф за уже встречавшиеся токены.
        max_new_tokens: Максимальное число новых токенов.

    Returns:
        Сгенерированное продолжение ``prompt``.
    """
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be greater than 0")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_token_ids = output_ids[0, input_length:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()


def diagnose_decoding_controls(
    tokens: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Прогоняет три учебных кейса: цикл, узкий top-k и длинный хвост.

    Returns:
        Таблицы кейсов 1–3 и итоговую диагностику с выбранными значениями:
        ``repetition_penalty=1.05``, ``top_k=3``, ``top_p=0.95``.
    """
    vocabulary = DEMO_TOKENS if tokens is None else tokens

    case_1_logits = torch.tensor(
        [5.0, 4.7, 4.8, 4.6, 4.4, 2.0, 1.8, 1.5, 1.2, -1.0]
    )
    case_1_seen_token_ids = {0, 1}
    case_1_rows = []
    for penalty in [1.0, 1.05, 1.1, 1.2]:
        processed = apply_repetition_penalty(
            case_1_logits, case_1_seen_token_ids, penalty
        )
        chosen = choose_greedy_token(processed)
        case_1_rows.append(
            {
                "penalty": penalty,
                "chosen_token": vocabulary[chosen],
                "token_was_seen": chosen in case_1_seen_token_ids,
            }
        )
    case_1_results = pd.DataFrame(case_1_rows)
    case_1_solution = float(
        case_1_results.loc[~case_1_results["token_was_seen"], "penalty"].min()
    )

    case_2_logits = torch.tensor(
        [1.0, 0.8, 0.5, 0.4, 0.3, 5.0, 4.8, 4.5, 4.2, -1.0]
    )
    case_2_seeds = [11, 22, 33, 44, 55]
    case_2_rows = []
    for top_k in [1, 2, 3, 5]:
        processed = apply_top_k(case_2_logits, top_k=top_k)
        sampled_ids = sample_tokens(processed, seeds=case_2_seeds)
        case_2_rows.append(
            {
                "top_k": top_k,
                "sampled_tokens": decode_token_ids(sampled_ids, vocabulary),
                "n_unique_tokens": len(set(sampled_ids)),
            }
        )
    case_2_results = pd.DataFrame(case_2_rows)
    case_2_solution = int(
        case_2_results.loc[
            case_2_results["n_unique_tokens"] >= 3, "top_k"
        ].min()
    )

    case_3_logits = torch.log(
        torch.tensor(
            [0.28, 0.21, 0.16, 0.11, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01]
        )
    )
    case_3_rows = []
    for top_p in [1.0, 0.95, 0.9, 0.8]:
        processed = apply_top_p(case_3_logits, top_p=top_p)
        available_mask = torch.isfinite(processed)
        case_3_rows.append(
            {
                "top_p": top_p,
                "available_count": int(available_mask.sum().item()),
                "kept_probability_mass": kept_probability_mass(
                    case_3_logits, processed
                ),
                "problem_token_available": bool(available_mask[9].item()),
            }
        )
    case_3_results = pd.DataFrame(case_3_rows)
    case_3_solution = float(
        case_3_results.loc[
            ~case_3_results["problem_token_available"], "top_p"
        ].max()
    )

    diagnosis = pd.DataFrame(
        [
            {
                "case": "Зацикливание",
                "symptom": "Модель снова выбирает встречавшийся токен",
                "parameter": "repetition_penalty",
                "selected_value": case_1_solution,
            },
            {
                "case": "Нет разнообразия",
                "symptom": "Разные seed выбирают один токен",
                "parameter": "top_k",
                "selected_value": case_2_solution,
            },
            {
                "case": "Длинный хвост",
                "symptom": "Слабый токен участвует в sampling",
                "parameter": "top_p",
                "selected_value": case_3_solution,
            },
        ]
    )
    return case_1_results, case_2_results, case_3_results, diagnosis
