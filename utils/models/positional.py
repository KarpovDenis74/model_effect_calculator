"""Позиционное кодирование: абсолютные метки, RoPE, ALiBi и YaRN.

Внимание сравнивает токены попарно и само не знает, кто стоял раньше.
Позиционное кодирование добавляет место токена. От схемы зависит,
насколько длинную последовательность модель вообще может принять.

RoPE поворачивает вектор на угол, кратный позиции, и при сравнении
видит относительное расстояние. За обучающей длиной углы становятся
незнакомыми. ALiBi не хранит позиционные векторы: к оценке внимания
добавляется штраф, пропорциональный расстоянию, и его можно посчитать
для любой длины. YaRN — не новый тип кодирования, а способ сжать
позиции уже обученной RoPE-модели в знакомый диапазон углов.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from utils.models.long_context import inference_device

TINYLLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
QWEN_INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TINYLLAMA_CONTEXT_WINDOW = 2048

CONTRACT_FILLER = "Компания ведёт учёт договоров и следит за сроками их исполнения. "

BEACON_EXPECTED = {
    "АЛЬФА": "4821",
    "БЕТА": "7395",
    "ГАММА": "1608",
}
BEACON_QUESTIONS = {
    "АЛЬФА": "Какое кодовое число у АЛЬФА? Ответь только числом.",
    "БЕТА": "Какое кодовое число у БЕТА? Ответь только числом.",
    "ГАММА": "Какое кодовое число у ГАММА? Ответь только числом.",
}

METHOD_IDEAS = (
    (
        "Абсолютное кодирование",
        (
            "Каждой позиции сопоставлена своя метка-вектор, "
            "а число меток ограничено обучающей длиной.",
            "За пределом обучающей длины дополнительным токенам "
            "не хватает позиционных меток.",
        ),
    ),
    (
        "RoPE",
        (
            "Подход используется в Llama, Qwen и Mistral; "
            "за пределом обучающей длины качество может снижаться.",
            "Позиция задаётся поворотом вектора токена, а при сравнении "
            "важна разница углов — относительное расстояние.",
        ),
    ),
    (
        "ALiBi",
        (
            "Позиционных векторов нет, а связь с далёкими токенами "
            "штрафуется пропорционально расстоянию.",
            "Может работать на последовательностях длиннее обучающих "
            "без отдельного дообучения на расширенное окно.",
        ),
    ),
    (
        "YaRN",
        (
            "Расширяет окно готовой RoPE-модели с помощью масштабирования "
            "позиций и дополнительной настройки.",
            "Позволяет получить длинноконтекстную версию модели "
            "с меньшими затратами, чем полное обучение с нуля.",
        ),
    ),
)

YARN_RECOMMENDATION = (
    "Взять версию существующей модели, окно которой расширено с помощью YaRN."
)


@dataclass(frozen=True)
class BeaconAnswer:
    """Ответ модели на один маячок."""

    name: str
    expected: str
    answer: str
    found: bool


def rope_rotation_angle(position: int, base_angle: float) -> float:
    """Угол поворота вектора токена в RoPE.

    Позиция 1 поворачивается на θ, позиция 2 — на 2θ. Чем дальше токен,
    тем сильнее поворот.

    Args:
        position: Место токена в последовательности.
        base_angle: Угол одного шага, θ.

    Returns:
        Угол поворота для этой позиции.
    """
    if position < 0:
        raise ValueError("position не может быть отрицательной")
    return position * base_angle


def rope_relative_angle(later: int, earlier: int, base_angle: float) -> float:
    """Разница углов RoPE между двумя позициями.

    Позиции 5 и 4 отличаются на θ, позиции 101 и 100 — тоже на θ.
    Абсолютный поворот при сравнении превращается в расстояние.
    """
    return rope_rotation_angle(later, base_angle) - rope_rotation_angle(
        earlier, base_angle
    )


def alibi_penalty(distance: int, slope: float) -> float:
    """Штраф внимания ALiBi за расстояние до предыдущего токена.

    Штраф равен ``-distance * slope``. Он зависит только от расстояния,
    поэтому считается и для 100, и для 10 000 токенов. Чем больше
    расстояние, тем слабее связь.

    Args:
        distance: Сколько позиций назад лежит токен.
        slope: Наклон головы внимания.

    Returns:
        Отрицательное смещение оценки внимания.
    """
    if distance < 0:
        raise ValueError("distance не может быть отрицательным")
    return -distance * slope


def yarn_scaled_position(
    position: int,
    trained_window: int,
    extended_window: int,
) -> float:
    """Переводит длинную позицию в угол, знакомый RoPE-модели.

    Позиция 64 000 при окнах 8 000 и 64 000 получает масштаб позиции 8 000:
    модель видит угол из диапазона обучения, а не новый большой угол.
    Это упрощённая схема. В полном YaRN разные пары вектора сжимаются
    по-разному, а затем модель ещё донастраивают на длинных примерах.

    Args:
        position: Позиция в расширенной последовательности.
        trained_window: Длина, на которой модель училась.
        extended_window: Длина, которую нужно принять.

    Returns:
        Позиция после равномерного масштабирования.
    """
    if position < 0:
        raise ValueError("position не может быть отрицательной")
    if trained_window <= 0 or extended_window <= 0:
        raise ValueError("окна должны быть положительными")
    return position * trained_window / extended_window


def absolute_position_defined(position: int, trained_length: int) -> bool:
    """Есть ли у абсолютного кодирования метка для этой позиции.

    Метки создаются только для длины обучения. Позиция за этой границей
    не получает вектор, и модель не может обработать лишний токен.
    Нумерация здесь с единицы, как «первый, второй, третий токен».
    """
    if trained_length < 1:
        raise ValueError("trained_length должен быть не меньше 1")
    return 1 <= position <= trained_length


def recommend_context_extension(
    document_tokens: int,
    model_window: int,
    uses_rope: bool,
    can_fully_retrain: bool,
) -> str:
    """Выбирает, как принять документ длиннее текущего окна.

    Если документ и так входит в окно, кодирование менять не нужно.
    Полное обучение с нуля возможно только при отдельном бюджете.
    Для уже проверенной RoPE-модели без такого бюджета подходит YaRN.
    Иначе имеет смысл модель с ALiBi: её штраф не привязан к обучающей длине.

    Args:
        document_tokens: Длина документа.
        model_window: Заявленное окно текущей модели.
        uses_rope: Текущая модель кодирует позиции через RoPE.
        can_fully_retrain: Можно ли обучить новую модель с нуля.

    Returns:
        Короткая рекомендация.
    """
    if document_tokens < 1 or model_window < 1:
        raise ValueError("длины должны быть положительными")
    if document_tokens <= model_window:
        return "Текущего окна достаточно, расширять его не нужно."
    if can_fully_retrain:
        return "Бюджет позволяет обучить модель с нуля на новой длине."
    if uses_rope:
        return YARN_RECOMMENDATION
    return (
        "Рассмотреть модель с ALiBi: она лучше переносится на длину "
        "больше обучающей без отдельного расширения окна."
    )


def count_text_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    """Считает токены так же, как длина документа в уроке."""
    return len(tokenizer(text)["input_ids"])


def build_filler_block(
    tokenizer: PreTrainedTokenizerBase,
    target_tokens: int,
    filler: str = CONTRACT_FILLER,
) -> str:
    """Набирает нейтральный текст не короче ``target_tokens``."""
    if target_tokens < 1:
        raise ValueError("target_tokens должен быть не меньше 1")
    if not filler:
        raise ValueError("filler не должен быть пустым")

    text = ""
    steps = 0
    while count_text_tokens(tokenizer, text) < target_tokens:
        text += filler
        steps += 1
        if steps > target_tokens:
            raise RuntimeError("Наполнитель не увеличивает число токенов")
    return text


def make_beacon_document(
    tokenizer: PreTrainedTokenizerBase,
    block_tokens: int,
    filler: str = CONTRACT_FILLER,
) -> str:
    """Собирает документ с тремя кодовыми числами и наполнителем между ними.

    Одинаковые маячки стоят в начале, середине и конце. Меняется только
    длина блоков, поэтому короткий и длинный документы отличаются окном,
    а не заданием.
    """
    block = build_filler_block(tokenizer, block_tokens, filler)
    return (
        f"Кодовое число АЛЬФА равно {BEACON_EXPECTED['АЛЬФА']}. "
        + block
        + f"Кодовое число БЕТА равно {BEACON_EXPECTED['БЕТА']}. "
        + block
        + f"Кодовое число ГАММА равно {BEACON_EXPECTED['ГАММА']}. "
    )


def _load_causal_model(model_name: str, dtype: torch.dtype):
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)


def _release_model(model: torch.nn.Module) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def ask_about_beacons(
    model_name: str,
    document: str,
    questions: dict[str, str] | None = None,
    expected: dict[str, str] | None = None,
    max_new_tokens: int = 20,
) -> dict:
    """Спрашивает модель о каждом маячке и выгружает её.

    Модели грузятся по одной: перед возвратом веса удаляются, чтобы
    следующая модель не делила с ними память. Попадание считается
    найденным, если ожидаемое число есть в тексте ответа.

    На CUDA и MPS веса берутся в float16, на CPU — в float32.
    Устройство выбирает ``inference_device``.

    Args:
        model_name: Имя модели на Hugging Face.
        document: Текст, в котором спрятаны маячки.
        questions: Имя маячка и вопрос. По умолчанию три кодовых числа.
        expected: Имя маячка и строка, которую нужно найти в ответе.
        max_new_tokens: Сколько новых токенов генерировать на вопрос.

    Returns:
        Окно модели, длины промптов, ответы и текст ошибки, если
        генерация оборвалась из-за длины или устройства.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens должен быть не меньше 1")

    questions = questions or BEACON_QUESTIONS
    expected = expected or BEACON_EXPECTED
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

    context_window = int(getattr(model.config, "max_position_embeddings", 0))
    answers: list[BeaconAnswer] = []
    prompt_tokens: list[int] = []
    error = None

    try:
        for name, question in questions.items():
            messages = [{"role": "user", "content": document + "\n\n" + question}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            prompt_length = int(inputs["input_ids"].shape[1])
            prompt_tokens.append(prompt_length)

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            answer = tokenizer.decode(
                output[0][prompt_length:],
                skip_special_tokens=True,
            ).strip()
            answers.append(
                BeaconAnswer(
                    name=name,
                    expected=expected[name],
                    answer=answer,
                    found=expected[name] in answer,
                )
            )
    except (RuntimeError, IndexError) as exc:
        error = str(exc)
    finally:
        _release_model(model)

    return {
        "model_name": model_name,
        "device": str(device),
        "context_window": context_window,
        "prompt_tokens": prompt_tokens,
        "answers": answers,
        "error": error,
    }


def beacon_marks(answers: list[BeaconAnswer]) -> dict[str, str]:
    """Переводит попадания в подписи «нашла» и «нет»."""
    return {answer.name: "нашла" if answer.found else "нет" for answer in answers}
