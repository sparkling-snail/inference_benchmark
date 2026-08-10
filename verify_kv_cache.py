"""
Phase 2 verification.

Confirms the from-scratch single-request KV cache produces
token-for-token identical output to HuggingFace's own cached
generate(), then reports how long cached generation took as a
preview of the Phase 1 -> Phase 2 latency win (a proper side-by-side
benchmark is Phase 5's job, once Phase 3/4 exist too).

Usage:
    python verify_kv_cache.py
"""

import time

import torch

from src.kv_cache_single import run_single_with_cache
from src.model_wrapper import ModelWrapper
from src.request import Request

PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time in a small village,",
    "The best way to learn a new programming language is",
]

MAX_NEW_TOKENS = 30


def hf_reference_generate(model: ModelWrapper, prompt_token_ids: list[int]) -> list[int]:
    """Greedy-decodes the same prompt via HF's built-in cached generate()."""
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=model.tokenizer.pad_token_id,
        )
    return output[0][len(prompt_token_ids):].tolist()


def main():
    print("Loading model...")
    model = ModelWrapper()
    print(f"Device: {model.device}\n")

    all_passed = True
    cached_time_total = 0.0

    for prompt in PROMPTS:
        req = Request(prompt=prompt, max_new_tokens=MAX_NEW_TOKENS)

        start = time.perf_counter()
        req = run_single_with_cache(model, req)
        cached_time_total += time.perf_counter() - start

        reference = hf_reference_generate(model, req.prompt_token_ids)
        match = req.generated_token_ids == reference
        all_passed &= match

        print(f"[{'PASS' if match else 'FAIL'}] {prompt!r}")
        if not match:
            print(f"  ours:      {req.generated_token_ids}")
            print(f"  reference: {reference}")

    print()
    if all_passed:
        print(f"All {len(PROMPTS)} prompts match HF's cached generate() token-for-token.")
        print("Phase 2 verified -- safe to move on to Phase 3 (batched KV cache).")
    else:
        print("MISMATCH -- KV cache implementation has a bug. Do not proceed to Phase 3.")

    print(
        f"\nSingle-request cached generation: {cached_time_total:.3f}s total "
        f"for {len(PROMPTS)} sequential requests ({MAX_NEW_TOKENS} tokens each)."
    )


if __name__ == "__main__":
    main()
