"""Один шаг генерации и ручной цикл: logits → softmax → выбор токена."""

from __future__ import annotations

import torch
import pandas as pd
from transformers import set_seed


def next_token_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Возвращает сырые оценки следующего токена для текущего контекста.

    Модель отдаёт ``logits`` формы ``[batch, sequence_length, vocab_size]``.
    Для генерации нужна только последняя позиция: оценки после всего уже
    известного текста. Результат имеет форму ``[batch, vocab_size]``.

    Args:
        model: Causal-модель в режиме ``eval``.
        input_ids: Номера токенов текущего контекста.

    Returns:
        Логиты следующего токена. Это ещё не вероятности.
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    return outputs.logits[:, -1, :]


def next_token_probs(logits: torch.Tensor) -> torch.Tensor:
    """Превращает логиты в распределение вероятностей по словарю.

    Softmax сохраняет порядок значений: самый большой логит станет самой
    большой вероятностью. Поэтому greedy-выбор можно брать и по логитам,
    и по вероятностям — результат один. Softmax нужен, когда смотрят
    уверенность модели или сэмплируют из распределения.

    Args:
        logits: Сырые оценки модели, обычно формы ``[batch, vocab_size]``.

    Returns:
        Вероятности от 0 до 1. Сумма по словарю равна 1.
    """
    return torch.softmax(logits, dim=-1)


def top_token_candidates(
    tokenizer,
    probs: torch.Tensor,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Возвращает ``top_k`` самых вероятных следующих токенов.

    Args:
        tokenizer: Токенизатор модели.
        probs: Распределение вероятностей формы ``[batch, vocab_size]``.
        top_k: Сколько кандидатов оставить.

    Returns:
        Пары ``(токен, вероятность)`` от большей вероятности к меньшей.
    """
    top_probs, top_token_ids = torch.topk(probs, k=top_k, dim=-1)
    candidates = []
    for prob, token_id in zip(top_probs[0], top_token_ids[0]):
        token = tokenizer.decode(token_id.item())
        candidates.append((token, float(prob.item())))
    return candidates


def show_next_token_candidates(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Печатает и возвращает top-k кандидатов после одного промпта.

    Цепочка шага: текст → ``input_ids`` → модель → логиты последней
    позиции → softmax → топ кандидатов. На этом шаге модель сравнивает
    только один следующий токен, а не готовые фразы целиком.

    Args:
        model: Causal-модель.
        tokenizer: Токенизатор этой модели.
        prompt: Текущий контекст.
        top_k: Сколько кандидатов показать.

    Returns:
        Список пар ``(токен, вероятность)``.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    logits = next_token_logits(model, inputs["input_ids"])
    probs = next_token_probs(logits)
    candidates = top_token_candidates(tokenizer, probs, top_k=top_k)

    print(f"Промпт: {prompt!r}")
    print("Топ кандидатов:")
    for rank, (token, prob) in enumerate(candidates, start=1):
        print(f"{rank}. {token!r:<12} probability={prob:.4f}")
    return candidates


def choose_next_token_id(
    next_token_logits_tensor: torch.Tensor,
    strategy: str = "greedy",
) -> torch.Tensor:
    """Выбирает id следующего токена greedy-способом или sampling.

    Greedy берёт максимум по логитам. Sampling сэмплирует из полного
    softmax-распределения: более вероятные токены выпадают чаще, но не
    всегда. Результат имеет форму ``[batch, 1]``, чтобы его можно было
    сразу дописать к ``input_ids``.

    Args:
        next_token_logits_tensor: Логиты последней позиции.
        strategy: ``greedy`` или ``sample``.

    Returns:
        Тензор с id выбранного токена.

    Raises:
        ValueError: неизвестная стратегия.
    """
    if strategy not in {"greedy", "sample"}:
        raise ValueError("strategy must be either 'greedy' or 'sample'")

    if strategy == "greedy":
        return torch.argmax(next_token_logits_tensor, dim=-1, keepdim=True)

    probs = next_token_probs(next_token_logits_tensor)
    return torch.multinomial(probs, num_samples=1)


def generate_with_strategy(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 30,
    strategy: str = "greedy",
    seed: int | None = None,
) -> str:
    """Генерирует текст через greedy decoding или sampling.

    На каждом шаге текущий контекст проходит через модель, из логитов
    выбирается один токен, и он дописывается к входу. Чем больше
    ``max_new_tokens``, тем больше последовательных шагов и тем выше
    задержка ответа. При ``seed`` sampling становится воспроизводимым.

    Args:
        model: Causal-модель.
        tokenizer: Токенизатор этой модели.
        prompt: Начальный текст.
        max_new_tokens: Сколько новых токенов дописать.
        strategy: ``greedy`` или ``sample``.
        seed: Зерно генератора случайных чисел для sampling.

    Returns:
        Промпт вместе со сгенерированным продолжением.
    """
    if strategy not in {"greedy", "sample"}:
        raise ValueError("strategy must be either 'greedy' or 'sample'")

    if seed is not None:
        torch.manual_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    model.eval()

    for _ in range(max_new_tokens):
        logits = next_token_logits(model, input_ids)
        next_token_id = choose_next_token_id(logits, strategy=strategy)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        if (
            tokenizer.eos_token_id is not None
            and next_token_id[0].item() == tokenizer.eos_token_id
        ):
            break

    return tokenizer.decode(input_ids[0])


def generate_greedy_with_trace(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 10,
    top_k: int = 5,
) -> str:
    """Генерирует текст greedy-способом и печатает top-k на каждом шаге.

    Печать показывает, как выбранный токен меняет следующий контекст.
    Softmax здесь нужен только для вывода вероятностей: сам greedy-выбор
    берётся по ``argmax`` логитов.

    Args:
        model: Causal-модель.
        tokenizer: Токенизатор этой модели.
        prompt: Начальный текст.
        max_new_tokens: Сколько новых токенов дописать.
        top_k: Сколько кандидатов печатать на шаге.

    Returns:
        Промпт вместе со сгенерированным продолжением.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    model.eval()

    for step in range(1, max_new_tokens + 1):
        logits = next_token_logits(model, input_ids)
        probs = next_token_probs(logits)
        candidates = top_token_candidates(tokenizer, probs, top_k=top_k)
        next_token_id = choose_next_token_id(logits, strategy="greedy")

        print(f"\nШаг {step}")
        print(f"Текущий текст: {tokenizer.decode(input_ids[0])!r}")
        print("Топ кандидатов:")
        for rank, (token, prob) in enumerate(candidates, start=1):
            print(f"{rank}. {token!r:<12} вероятность={prob:.4f}")

        chosen_token = tokenizer.decode(next_token_id[0].item())
        print(f"Выбранный токен: {chosen_token!r}")

        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        if (
            tokenizer.eos_token_id is not None
            and next_token_id[0].item() == tokenizer.eos_token_id
        ):
            break

    return tokenizer.decode(input_ids[0])


def build_greedy_trace(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 35,
    top_k: int = 5,
) -> tuple[pd.DataFrame, str]:
    """Собирает пошаговый отчёт для ручной greedy-генерации.

    Таблица нужна для диагностики повторов: видно, какой текст был до
    выбора, какой токен взяли и какие ещё кандидаты были рядом. В greedy
    выбранный токен всегда совпадает с ``top_1_token``.

    Args:
        model: Causal-модель.
        tokenizer: Токенизатор этой модели.
        prompt: Начальный текст.
        max_new_tokens: Сколько новых токенов дописать.
        top_k: Сколько кандидатов сохранить в строке таблицы.

    Returns:
        Таблица шагов и итоговый текст.
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    rows = []

    for step in range(1, max_new_tokens + 1):
        current_text = tokenizer.decode(input_ids[0])
        logits = next_token_logits(model, input_ids)
        probs = next_token_probs(logits)
        top_probs, top_token_ids = torch.topk(probs, k=top_k, dim=-1)
        next_token_id = torch.argmax(probs, dim=-1, keepdim=True)

        row = {
            "step": step,
            "current_text": current_text,
            "chosen_token": tokenizer.decode(next_token_id[0].item()),
            "chosen_probability": probs[0, next_token_id[0].item()].item(),
        }

        for rank, (prob, token_id) in enumerate(
            zip(top_probs[0], top_token_ids[0]),
            start=1,
        ):
            row[f"top_{rank}_token"] = tokenizer.decode(token_id.item())
            row[f"top_{rank}_probability"] = prob.item()

        rows.append(row)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        if (
            tokenizer.eos_token_id is not None
            and next_token_id[0].item() == tokenizer.eos_token_id
        ):
            break

    greedy_text = tokenizer.decode(input_ids[0])
    return pd.DataFrame(rows), greedy_text


def generate_sample_with_full_distribution(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 35,
    seed: int = 42,
) -> str:
    """Генерирует текст через ``model.generate`` из полного распределения.

    ``do_sample=True`` включает sampling. ``top_k=0`` отключает отбор
    пятидесяти лучших токенов, который transformers включает по умолчанию.
    Так библиотечный метод сэмплирует из того же полного распределения,
    что и ручной ``torch.multinomial``.

    Args:
        model: Causal-модель.
        tokenizer: Токенизатор этой модели.
        prompt: Начальный текст.
        max_new_tokens: Сколько новых токенов дописать.
        seed: Зерно генератора случайных чисел.

    Returns:
        Промпт вместе со сгенерированным продолжением.
    """
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt")
    sample_output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=0,
    )
    return tokenizer.decode(sample_output[0])


def diagnose_generation_checks(
    greedy_trace: pd.DataFrame,
    greedy_text: str,
    sample_text: str,
    repeated_phrase: str = "я не буду",
) -> dict[str, bool]:
    """Собирает короткие проверки по диагностическому отчёту.

    Args:
        greedy_trace: Таблица из ``build_greedy_trace``.
        greedy_text: Итоговый greedy-текст.
        sample_text: Текст после sampling.
        repeated_phrase: Фраза, которую ищем как признак повтора в greedy-тексте.

    Returns:
        Словарь с тремя проверками:
        greedy всегда берёт top-1, sampling отличается от greedy,
        в greedy-тексте есть заданная повторяющаяся фраза.
    """
    return {
        "greedy_always_uses_top_1": (
            greedy_trace["chosen_token"] == greedy_trace["top_1_token"]
        ).all(),
        "sampling_differs_from_greedy": sample_text != greedy_text,
        "greedy_contains_repeated_phrase": repeated_phrase in greedy_text,
    }


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    temperature: float,
    max_new_tokens: int = 120,
    seed: int = 42,
) -> str:
    """Генерирует продолжение текста с заданной температурой.

    Температура делит логиты перед softmax и меняет остроту распределения.
    Порядок токенов по вероятности не меняется, меняются шансы менее
    вероятных кандидатов. При ``temperature = 0`` случайности нет: включается
    greedy decoding через ``do_sample=False``, параметр ``temperature`` в
    ``generate`` не передаётся. При ``temperature > 0`` включается sampling.

    Функция возвращает только новые токены после промпта, без исходного текста.

    Args:
        model: Causal-модель в режиме ``eval``.
        tokenizer: Токенизатор этой модели.
        prompt: Исходный текст для модели.
        temperature: Значение температуры генерации.
        max_new_tokens: Максимальное число новых токенов.
        seed: Seed для воспроизводимости эксперимента.

    Returns:
        Сгенерированное продолжение ``prompt``.
    """
    if temperature < 0:
        raise ValueError("temperature не может быть отрицательной")

    set_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]

    if temperature == 0:
        generation_kwargs = {"do_sample": False}
    else:
        # min_new_tokens=1 не даёт сразу остановиться на EOS:
        # при некоторых seed sampling может выбрать конец последовательности
        # первым же токеном и вернуть пустую строку.
        generation_kwargs = {
            "do_sample": True,
            "temperature": temperature,
            "min_new_tokens": 1,
        }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )

    new_token_ids = output_ids[0][input_length:]
    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return generated_text.strip()
