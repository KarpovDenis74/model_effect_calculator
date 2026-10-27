from utils.cost import estimate_cost
from utils.tokenization import (
    count_hf_tokens,
    count_tokens,
    encode_text,
    get_bert_tokenizer,
    get_hf_tokenizer,
    get_openai_encoding,
    show_bert_pair_tokens,
    show_bert_tokens,
    show_hf_tokens,
    show_tokens,
)

# Эмбеддинги и causal-модели подгружаются при первом обращении: пакет torch
# нужен только им, а не подсчёту токенов и стоимости.
_LAZY_EXPORTS = {
    "find_nearest_tokens": "utils.embeddings",
    "get_embedding_matrix": "utils.embeddings",
    "get_single_token_id": "utils.embeddings",
    "get_token_embedding": "utils.embeddings",
    "load_embedding_model": "utils.embeddings",
    "normalize_embeddings": "utils.embeddings",
    "same_input_embedding": "utils.embeddings",
    "tokenize_without_special": "utils.embeddings",
    "DEMO_TOKENS": "utils.models",
    "architecture_summary": "utils.models",
    "apply_repetition_penalty": "utils.models",
    "apply_top_k": "utils.models",
    "apply_top_p": "utils.models",
    "build_greedy_trace": "utils.models",
    "choose_greedy_token": "utils.models",
    "choose_next_token_id": "utils.models",
    "count_parameters": "utils.models",
    "decode_token_ids": "utils.models",
    "diagnose_decoding_controls": "utils.models",
    "diagnose_generation_checks": "utils.models",
    "estimate_memory_mb": "utils.models",
    "estimate_model_weights_memory_mb": "utils.models",
    "generate_continuation": "utils.models",
    "generate_greedy_text": "utils.models",
    "generate_greedy_with_trace": "utils.models",
    "generate_sample_with_full_distribution": "utils.models",
    "generate_text": "utils.models",
    "generate_with_strategy": "utils.models",
    "load_causal_model": "utils.models",
    "measure_generation_speed": "utils.models",
    "next_token_logits": "utils.models",
    "next_token_probs": "utils.models",
    "sample_tokens": "utils.models",
    "show_next_token_candidates": "utils.models",
    "top_token_candidates": "utils.models",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, name)


__all__ = [
    "DEMO_TOKENS",
    "architecture_summary",
    "apply_repetition_penalty",
    "apply_top_k",
    "apply_top_p",
    "build_greedy_trace",
    "choose_greedy_token",
    "choose_next_token_id",
    "count_hf_tokens",
    "count_parameters",
    "count_tokens",
    "decode_token_ids",
    "diagnose_decoding_controls",
    "diagnose_generation_checks",
    "encode_text",
    "estimate_cost",
    "estimate_memory_mb",
    "estimate_model_weights_memory_mb",
    "find_nearest_tokens",
    "generate_continuation",
    "generate_greedy_text",
    "generate_greedy_with_trace",
    "generate_sample_with_full_distribution",
    "generate_text",
    "generate_with_strategy",
    "get_bert_tokenizer",
    "get_embedding_matrix",
    "get_hf_tokenizer",
    "get_openai_encoding",
    "get_single_token_id",
    "get_token_embedding",
    "load_causal_model",
    "load_embedding_model",
    "measure_generation_speed",
    "next_token_logits",
    "next_token_probs",
    "normalize_embeddings",
    "same_input_embedding",
    "sample_tokens",
    "show_bert_pair_tokens",
    "show_bert_tokens",
    "show_hf_tokens",
    "show_next_token_candidates",
    "show_tokens",
    "tokenize_without_special",
    "top_token_candidates",
]
