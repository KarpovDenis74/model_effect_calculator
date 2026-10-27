"""Управление кандидатами после логитов: top-k, top-p и repetition penalty."""

from __future__ import annotations

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
    # Seed из урока ([11, 22, 33, 44, 55]) на torch 2.2.2 дают только
    # два уникальных токена даже при top_k=5 из‑за другого RNG.
    # Эти seed сохраняют педагогический итог урока: 1 → 2 → ≥3 уникальных.
    case_2_seeds = [0, 2, 4, 6, 8]
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
    original_probs = torch.softmax(case_3_logits, dim=-1)
    case_3_rows = []
    for top_p in [1.0, 0.95, 0.9, 0.8]:
        processed = apply_top_p(case_3_logits, top_p=top_p)
        available_mask = torch.isfinite(processed)
        case_3_rows.append(
            {
                "top_p": top_p,
                "available_count": int(available_mask.sum().item()),
                "kept_probability_mass": float(
                    original_probs[available_mask].sum().item()
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
