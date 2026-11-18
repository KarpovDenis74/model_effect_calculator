"""Токен-бюджет и три способа укоротить растущую историю диалога.

В окно модели должны вместе войти системный промпт, история и запас
под ответ. Промпт обычно короткий и постоянный, запас задаётся заранее,
а история растёт с каждым ходом.

Truncation выбрасывает самые старые сообщения, в том числе инструкции,
если они лежат в общем списке. Sliding window оставляет системный промпт
неприкосновенным и набирает историю с конца. Summarization сжимает старую
часть отдельным запросом: смысл сохраняется, но появляются задержка,
стоимость и ещё один вызов модели.
"""

from __future__ import annotations

import gc
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from utils.models.long_context import inference_device

LESSON_CONTEXT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
SUPPORT_CHAT_PROMPT = (
    "Ты — вежливый ассистент службы поддержки. Отвечай кратко и по делу."
)
CONTEXT_LIMIT = 4096
RESPONSE_RESERVE = 512

STRATEGIES_BY_COST = ("truncation", "sliding window", "summarization")

SUMMARY_PROMPT_PREFIX = (
    "Ниже фрагмент диалога службы поддержки. "
    "Перескажи его суть в 2-3 предложениях. "
    "Не повторяй каждую реплику отдельно. "
    "Обязательно сохрани номера заказов, можно диапазоном, "
    "и срок доставки:\n\n"
)

Message = dict[str, str]


def count_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    """Считает токены токенизатором той модели, которая получит запрос.

    Один и тот же текст у разных моделей распадается на разное число
    токенов, поэтому бюджет нельзя считать чужим токенизатором.
    """
    return len(tokenizer(text)["input_ids"])


def history_token_count(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[Message],
) -> int:
    """Суммирует токены текстов сообщений, без служебных токенов шаблона."""
    return sum(count_tokens(tokenizer, message["content"]) for message in messages)


def make_support_history(exchanges: int = 25) -> list[Message]:
    """Собирает диалог техподдержки: у каждого заказа свой номер.

    Пара реплик ``i`` говорит о заказе ``1000 + i``. Первые десять
    обменов — это заказы 1001–1010, их и сжимает пример с резюме.
    """
    if exchanges < 1:
        raise ValueError("exchanges должен быть не меньше 1")

    history = []
    for number in range(1, exchanges + 1):
        order_id = 1000 + number
        history.append(
            {
                "role": "user",
                "content": (
                    f"Здравствуйте, подскажите по заказу №{order_id}: "
                    "где он сейчас и когда ждать доставку?"
                ),
            }
        )
        history.append(
            {
                "role": "assistant",
                "content": (
                    f"Заказ №{order_id} в пути, ожидаемая доставка — "
                    "через 3–5 рабочих дней. Спасибо за обращение!"
                ),
            }
        )
    return history


def history_budget(
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: str,
    context_limit: int,
    response_reserve: int,
) -> int:
    """Сколько токенов истории остаётся после промпта и запаса под ответ."""
    if context_limit < 1 or response_reserve < 0:
        raise ValueError("лимит контекста и запас под ответ заданы неверно")
    budget = context_limit - count_tokens(tokenizer, system_prompt) - response_reserve
    if budget < 0:
        raise ValueError(
            "Системный промпт и запас под ответ уже больше контекстного лимита"
        )
    return budget


def truncate_messages(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[Message],
    token_budget: int,
) -> list[Message]:
    """Удаляет сообщения с начала списка, пока сумма токенов не влезет в бюджет.

    Системный промпт здесь ничем не защищён. Если он стоит первым,
    при нехватке места он исчезнет вместе со старой историей.
    """
    if token_budget < 0:
        raise ValueError("token_budget не может быть отрицательным")

    kept = list(messages)
    while kept and history_token_count(tokenizer, kept) > token_budget:
        kept.pop(0)
    return kept


def build_context(
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: str,
    history: list[Message],
    context_limit: int = CONTEXT_LIMIT,
    response_reserve: int = RESPONSE_RESERVE,
) -> list[Message]:
    """Собирает скользящее окно: системный промпт и хвост истории.

    Бюджет истории равен лимиту окна минус токены промпта и минус запас
    под ответ. Сообщения набираются с конца и останавливаются на первом,
    которое уже не помещается. Исходный порядок сохранён: промпт первый,
    затем отобранные реплики от старых к новым.

    Args:
        tokenizer: Токенизатор модели, которая получит этот контекст.
        system_prompt: Инструкции. Они не удаляются.
        history: Реплики диалога без системного сообщения.
        context_limit: Верхняя граница всего запроса, включая ответ.
        response_reserve: Сколько токенов заранее оставить под генерацию.

    Returns:
        Список сообщений для чат-шаблона.
    """
    budget = history_budget(tokenizer, system_prompt, context_limit, response_reserve)
    kept: list[Message] = []
    total = 0
    for message in reversed(history):
        message_tokens = count_tokens(tokenizer, message["content"])
        if total + message_tokens > budget:
            break
        kept.insert(0, message)
        total += message_tokens
    return [{"role": "system", "content": system_prompt}] + kept


def render_dialog(messages: list[Message]) -> str:
    """Склеивает реплики в текст ``роль: содержание``."""
    return "\n".join(
        f"{message['role']}: {message['content']}" for message in messages
    )


def _load_causal_model(model_name: str, dtype: torch.dtype):
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)


def load_chat_model(model_name: str):
    """Загружает чат-модель на MPS, CUDA или CPU.

    В уроке это ``Qwen/Qwen2.5-3B-Instruct`` на GPU. Здесь устройство
    выбирается само, а имя модели передаёт вызывающий код.

    Returns:
        Токенизатор и модель в режиме eval.
    """
    device = inference_device()
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_causal_model(model_name, dtype)
    model.to(device)
    model.eval()
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def release_chat_model(model: torch.nn.Module) -> None:
    """Удаляет веса и очищает кеш ускорителя."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def summarize_history(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    old_messages: list[Message],
    max_new_tokens: int = 150,
) -> str:
    """Сжимает старую часть диалога в короткое резюме.

    Промпт прямо называет, что нельзя потерять: номера заказов и
    договорённости. Общая просьба «перескажи короче» часто выбрасывает
    именно эти детали. Резюме — новый запрос к модели, его не запускают
    после каждой реплики, а только когда история подходит к порогу.

    Args:
        model: Уже загруженная чат-модель.
        tokenizer: Её токенизатор.
        old_messages: Ранние реплики, которые нужно заменить резюме.
        max_new_tokens: Верхняя граница длины резюме.

    Returns:
        Текст резюме без входного промпта.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens должен быть не меньше 1")
    if not old_messages:
        raise ValueError("Нечего сжимать: список сообщений пуст")

    prompt = SUMMARY_PROMPT_PREFIX + render_dialog(old_messages)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    prompt_length = int(inputs["input_ids"].shape[1])

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    summary = tokenizer.decode(
        output[0][prompt_length:],
        skip_special_tokens=True,
    )
    return summary.strip()


def orders_in_text(text: str, first: int, last: int) -> list[int]:
    """Какие номера заказов из диапазона буквально есть в тексте."""
    return [number for number in range(first, last + 1) if str(number) in text]


def mentions_delivery_window(text: str) -> bool:
    """Есть ли в тексте срок «3–5» в любом из обычных написаний тире."""
    return re.search(r"3\s*[–—-]\s*5", text) is not None
