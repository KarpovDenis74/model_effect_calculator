from utils import estimate_cost

# Вы выбираете модель для обработки пользовательских обращений. Один и тот же запрос уже токенизировали двумя токенизаторами
gpt_input_tokens = 78
llama_input_tokens = 112 
# Ожидаемая длина ответа:
expected_output_tokens = 500
# Цены моделей за 1 миллион токенов:
gpt_input_price = 0.75
gpt_output_price = 4.50

llama_input_price = 0.60
llama_output_price = 0.60
# Бюджет на 10 000 запросов:
budget = 20
requests_count = 10_000

"""
    Напишите код, который:
    Посчитает стоимость одного запроса для GPT-модели.
    Посчитает стоимость одного запроса для LLaMA-подобной модели.
    Посчитает стоимость 10 000 запросов для каждой модели.
    Выведет, какая модель проходит по бюджету.
    """

gpt_cost = estimate_cost(
    input_tokens=gpt_input_tokens,
    output_tokens=expected_output_tokens,
    input_price_per_million=gpt_input_price,
    output_price_per_million=gpt_output_price,
)

llama_cost = estimate_cost(
    input_tokens=llama_input_tokens,
    output_tokens=expected_output_tokens,
    input_price_per_million=llama_input_price,
    output_price_per_million=llama_output_price,
)

gpt_total_cost = gpt_cost * requests_count
llama_total_cost = llama_cost * requests_count

print("GPT total cost:", gpt_total_cost)
print("LLaMA total cost:", llama_total_cost)

if gpt_total_cost <= budget and llama_total_cost <= budget:
    print("Обе модели проходят по бюджету")
elif gpt_total_cost <= budget:
    print("По бюджету проходит GPT")
elif llama_total_cost <= budget:
    print("По бюджету проходит LLaMA")
else:
    print("Ни одна модель не проходит по бюджету")
