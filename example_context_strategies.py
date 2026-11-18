"""Пример: токен-бюджет, truncation, скользящее окно и резюме истории.

В запрос входят три части: системный промпт, история и запас под ответ.
История растёт и первой упирается в окно. Её сокращают тремя способами,
от дешёвого к дорогому: выбросить старые сообщения, оставить инструкции
и свежий хвост, сжать ранний диалог отдельным запросом.

Для чата техподдержки, где недавние реплики важнее старых, а каждый
вызов API стоит денег, подходит скользящее окно. Резюме нужно, когда
номер заказа из начала разговора нельзя потерять.

Урок считает токены и пишет резюме моделью Qwen2.5-3B-Instruct на GPU.
Здесь стратегии те же, а резюме пишет уже скачанная Qwen2.5-0.5B:
имя модели передаётся в функцию, окно от размера весов не зависит.

Запуск из корня проекта:

    python example_context_strategies.py
"""

import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    sys.exit(
        "Для этого примера нужен torch.\n"
        "Поставьте зависимости из requirements.txt в тот же интерпретатор, "
        "которым запускаете скрипт."
    )

from utils.models import (
    CONTEXT_LIMIT,
    LESSON_CONTEXT_MODEL,
    QWEN_INSTRUCT_MODEL,
    RESPONSE_RESERVE,
    STRATEGIES_BY_COST,
    SUPPORT_CHAT_PROMPT,
    build_context,
    history_budget,
    history_token_count,
    load_chat_model,
    make_support_history,
    mentions_delivery_window,
    orders_in_text,
    release_chat_model,
    summarize_history,
    truncate_messages,
)


def section(title: str) -> None:
    print("=" * 80, flush=True)
    print(title, flush=True)
    print("=" * 80, flush=True)


def print_context(
    title: str,
    context: list[dict[str, str]],
    tokenizer,
    possible_messages: int,
) -> None:
    used = history_token_count(tokenizer, context)
    print(f"\n{title}")
    print(
        f"сообщений {len(context)} из {possible_messages}, "
        f"токенов в текстах {used}"
    )
    print(f"первое: {context[0]['role']} — {context[0]['content'][:48]}")
    print(f"последнее: {context[-1]['content'][:48]}")


def main() -> None:
    section("1. Три стратегии по возрастанию цены")
    print("От самой простой и дешёвой к самой дорогой:")
    for index, name in enumerate(STRATEGIES_BY_COST, start=1):
        print(f"{index}. {name}")
    print(
        "Truncation только выбрасывает старые сообщения. "
        "Sliding window дополнительно защищает system prompt. "
        "Summarization зовёт модель ещё раз, поэтому тратит время, "
        "токены и код, но сохраняет смысл раннего диалога."
    )

    section("2. Чат техподдержки на полчаса")
    print(
        "Всю переписку слать нельзя: она дорожает и выйдет за окно. "
        "Резюме перед каждым ходом тоже дорого, если старые реплики "
        "быстро теряют актуальность. Truncation без защиты может "
        "удалить сами инструкции."
    )
    print(
        "Подходящий выбор — скользящее окно: system prompt остаётся, "
        "в запрос входят последние сообщения, которые помещаются в бюджет."
    )

    section("3. Токен-бюджет")
    print(
        f"В уроке резюме пишет {LESSON_CONTEXT_MODEL}. "
        f"Здесь токены и сжатие считает {QWEN_INSTRUCT_MODEL}."
    )
    tokenizer, model = load_chat_model(QWEN_INSTRUCT_MODEL)
    try:
        prompt_tokens = history_token_count(
            tokenizer,
            [{"role": "system", "content": SUPPORT_CHAT_PROMPT}],
        )
        print(f"System prompt занимает {prompt_tokens} токенов")
        print(
            f"Лимит {CONTEXT_LIMIT}, запас под ответ {RESPONSE_RESERVE}. "
            "На историю остаётся "
            f"{history_budget(tokenizer, SUPPORT_CHAT_PROMPT, CONTEXT_LIMIT, RESPONSE_RESERVE)} "
            "токенов."
        )

        history = make_support_history(25)
        print(
            f"История: {len(history)} сообщений, "
            f"{history_token_count(tokenizer, history)} токенов"
        )

        section("4. Скользящее окно при разных лимитах")
        print(
            "При 4096 подготовленная история обычно входит целиком. "
            "При 900 отрезается начало. При 600 остаётся только свежий хвост."
        )
        for limit in (CONTEXT_LIMIT, 900, 600):
            context = build_context(
                tokenizer,
                SUPPORT_CHAT_PROMPT,
                history,
                context_limit=limit,
                response_reserve=RESPONSE_RESERVE,
            )
            print_context(
                f"CONTEXT_LIMIT = {limit}",
                context,
                tokenizer,
                len(history) + 1,
            )
            print(f"system prompt на месте: {context[0]['role'] == 'system'}")
            print(
                "последнее сообщение истории сохранено: "
                f"{context[-1]['content'] == history[-1]['content']}"
            )

        section("5. Truncation может выбросить инструкции")
        combined = [{"role": "system", "content": SUPPORT_CHAT_PROMPT}] + history
        tight_budget = history_budget(
            tokenizer,
            SUPPORT_CHAT_PROMPT,
            900,
            RESPONSE_RESERVE,
        )
        truncated = truncate_messages(tokenizer, combined, tight_budget)
        print(
            "Тот же бюджет, что у истории при лимите 900, "
            "но системный промпт лежит в общем списке и не защищён."
        )
        print(f"после truncation сообщений: {len(truncated)}")
        print(f"system prompt сохранился: {truncated[0]['role'] == 'system'}")
        print(f"первое оставшееся: {truncated[0]['role']} — {truncated[0]['content'][:48]}")

        section("6. Резюме старой половины диалога")
        old_part = history[:20]
        summary = summarize_history(model, tokenizer, old_part)
        old_tokens = history_token_count(tokenizer, old_part)
        summary_tokens = history_token_count(
            tokenizer,
            [{"role": "assistant", "content": summary}],
        )
        print(f"Старая история: {old_tokens} токенов -> резюме: {summary_tokens} токенов")
        print(f"Сжатие примерно в {old_tokens // max(1, summary_tokens)} раз")
        print(f"\nРезюме:\n{summary}")

        kept_orders = orders_in_text(summary, 1001, 1010)
        print(f"номера 1001–1010 в резюме: {kept_orders or 'нет'}")
        print(f"срок 3–5 дней упомянут: {mentions_delivery_window(summary)}")
        print(
            "Скользящее окно эти двадцать реплик просто выкинуло бы. "
            "Резюме оставляет факты, но это отдельный вызов модели, "
            "поэтому его запускают у порога бюджета, а не после каждой реплики."
        )
    finally:
        release_chat_model(model)


if __name__ == "__main__":
    main()
