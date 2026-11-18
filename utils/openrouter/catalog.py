"""Каталог OpenRouter: бесплатные модели и их параметры."""

from __future__ import annotations

import os

import httpx

from utils.openrouter.client import OPENROUTER_BASE_URL

REQUIRED_GENERATION_PARAMETERS = frozenset({"temperature", "max_tokens"})
JSON_MODE_PARAMETERS = frozenset({"temperature", "max_tokens", "response_format"})
SEED_PARAMETERS = frozenset({"temperature", "seed"})
PREFERRED_SEED_MODEL = "openai/gpt-oss-20b:free"


def _is_free_text_model(model: dict, required_parameters: frozenset[str]) -> bool:
    architecture = model.get("architecture") or {}
    input_modalities = architecture.get("input_modalities") or []
    output_modalities = architecture.get("output_modalities") or []
    supported_parameters = set(model.get("supported_parameters") or [])
    return (
        str(model.get("id", "")).endswith(":free")
        and "text" in input_modalities
        and "text" in output_modalities
        and required_parameters.issubset(supported_parameters)
    )


def _authorized_get(api_key: str, url: str) -> httpx.Response:
    response = httpx.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )
    response.raise_for_status()
    return response


def list_models(api_key: str) -> list[dict]:
    """Возвращает карточки моделей из ``/models``.

    Args:
        api_key: Ключ OpenRouter.

    Returns:
        Список описаний моделей.
    """
    response = _authorized_get(api_key, f"{OPENROUTER_BASE_URL}/models")
    return response.json()["data"]


def get_model_info(api_key: str, model_id: str) -> dict:
    """Возвращает карточку одной модели.

    Сначала запрашивается адрес из урока, ``/model/{id}``. Если сервис
    отвечает 404, карточка ищется в общем каталоге ``/models``.

    Args:
        api_key: Ключ OpenRouter.
        model_id: Полный идентификатор, например ``openai/gpt-oss-20b:free``.

    Returns:
        Словарь с полями модели, включая ``supported_parameters``.

    Raises:
        LookupError: такой модели в каталоге нет.
    """
    direct_url = f"{OPENROUTER_BASE_URL}/model/{model_id}"
    try:
        response = _authorized_get(api_key, direct_url)
    except httpx.HTTPStatusError as error:
        if error.response.status_code != 404:
            raise
        response = None

    if response is not None:
        payload = response.json()
        if isinstance(payload, dict) and "data" in payload:
            return payload["data"]
        return payload

    for model in list_models(api_key):
        if model.get("id") == model_id:
            return model

    raise LookupError(f"Модель {model_id} не найдена в каталоге OpenRouter")


def check_supported_parameters(
    model_info: dict,
    required_parameters: list[str],
) -> dict[str, bool]:
    """Печатает, какие из нужных параметров есть у модели.

    Поддержка зависит от модели и провайдера. Общий список параметров
    OpenRouter ещё не значит, что конкретная модель их принимает.

    Args:
        model_info: Карточка модели с полем ``supported_parameters``.
        required_parameters: Имена параметров, которые код собирается передать.

    Returns:
        Словарь «параметр → поддерживается ли».
    """
    supported_parameters = set(model_info.get("supported_parameters", []))
    result = {}
    for parameter in required_parameters:
        is_supported = parameter in supported_parameters
        result[parameter] = is_supported
        status = "поддерживается" if is_supported else "не поддерживается"
        print(f"{parameter}: {status}")
    return result


def get_free_text_models(api_key: str) -> list[str]:
    """Возвращает бесплатные текстовые модели с temperature и max_tokens.

    Бесплатный вариант узнаётся по суффиксу ``:free``. Модель должна
    принимать и возвращать текст: иначе сравнение промптов на ней
    некорректно.

    Args:
        api_key: Ключ OpenRouter.

    Returns:
        Отсортированные идентификаторы моделей.
    """
    free_models = [
        model["id"]
        for model in list_models(api_key)
        if _is_free_text_model(model, REQUIRED_GENERATION_PARAMETERS)
    ]
    return sorted(free_models)


def get_free_json_models(api_key: str) -> list[str]:
    """Возвращает бесплатные модели, которые принимают JSON mode.

    Каталог запрашивается с сортировкой ``most-popular`` и фильтром
    ``response_format``. Порядок ответа сохраняется: первая модель —
    самая популярная из подходящих.

    Args:
        api_key: Ключ OpenRouter.

    Returns:
        Идентификаторы моделей.
    """
    response = httpx.get(
        f"{OPENROUTER_BASE_URL}/models",
        params={
            "supported_parameters": "response_format",
            "sort": "most-popular",
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )
    response.raise_for_status()

    return [
        model["id"]
        for model in response.json()["data"]
        if _is_free_text_model(model, JSON_MODE_PARAMETERS)
    ]


def select_json_model(model_ids: list[str]) -> str:
    """Выбирает модель для JSON-эксперимента.

    Если задана ``OPENROUTER_MODEL`` и она есть в списке, берётся она.
    Иначе берётся первая модель из ``model_ids``.

    Args:
        model_ids: Доступные бесплатные модели с JSON mode.

    Returns:
        Идентификатор модели.

    Raises:
        RuntimeError: список пуст или предпочтительная модель недоступна.
    """
    preferred_model = os.getenv("OPENROUTER_MODEL")
    if preferred_model:
        if preferred_model not in model_ids:
            raise RuntimeError(
                "Модель из OPENROUTER_MODEL не найдена среди "
                "доступных бесплатных моделей с поддержкой JSON mode"
            )
        return preferred_model

    if not model_ids:
        raise RuntimeError(
            "Не удалось найти бесплатную модель с поддержкой JSON mode"
        )
    return model_ids[0]


def select_seed_model(api_key: str) -> str:
    """Выбирает бесплатную текстовую модель с параметром ``seed``.

    Сначала ищется модель из урока, ``openai/gpt-oss-20b:free``.
    Если её нет в каталоге, берётся другая подходящая модель.

    Args:
        api_key: Ключ OpenRouter.

    Returns:
        Идентификатор модели.
    """
    candidates = [
        model["id"]
        for model in list_models(api_key)
        if _is_free_text_model(model, SEED_PARAMETERS)
    ]
    if PREFERRED_SEED_MODEL in candidates:
        return PREFERRED_SEED_MODEL
    if not candidates:
        raise RuntimeError("Нет бесплатной текстовой модели с параметром seed")
    return sorted(candidates)[0]


def select_two_models(model_ids: list[str]) -> list[str]:
    """Берёт по одной модели у двух разных разработчиков.

    Идентификатор имеет вид ``developer/model-name``. Если взять две
    модели одного разработчика, сравнение хуже показывает разницу
    между API. Маршрутизатор ``openrouter/free`` сюда не подходит:
    он может выбрать разную модель на каждом запросе.

    Args:
        model_ids: Идентификаторы, обычно результат ``get_free_text_models``.

    Returns:
        Два идентификатора.

    Raises:
        RuntimeError: разных разработчиков меньше двух.
    """
    models_by_developer = {}
    for model_id in model_ids:
        developer = model_id.split("/", maxsplit=1)[0]
        if developer == "openrouter":
            continue
        models_by_developer.setdefault(developer, model_id)

    selected_models = list(models_by_developer.values())[:2]
    if len(selected_models) < 2:
        raise RuntimeError(
            "Не удалось найти две бесплатные модели разных разработчиков"
        )
    return selected_models
