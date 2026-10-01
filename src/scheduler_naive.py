"""
Phase 1 -- naive static batching baseline.

Rules for this phase (intentionally dumb, this is the "before" number):
  - A batch is fixed once formed: every request in it runs for the same
    number of steps (max_new_tokens across the batch), even if it
    finished generating earlier. No admit/evict mid-batch.
  - No KV cache reuse: every generation step re-runs the FULL sequence
    (prompt + tokens generated so far) through the model from scratch.
    This is the main inefficiency Phase 2/3 will remove.
  - Requests of different prompt lengths are right-padded to the batch's
    longest current sequence.

This gives you a correctness-verified, measurable floor. Phase 3/4
should beat this on throughput at moderate-to-high concurrency, and the
benchmark script is what proves it.
"""

import torch

from .model_wrapper import ModelWrapper
from .request import Request


def run_naive_batch(model: ModelWrapper, requests: list[Request]) -> list[Request]:
    """
    Runs a fixed batch of requests to completion using static batching
    with full-sequence recomputation at every step (no KV cache).
    """
    for req in requests:
        req.prompt_token_ids = model.encode(req.prompt) # asks the tokenizer to turn each prompt into numbers.

    # sequences we mutate in place as generation proceeds
    sequences = [list(req.prompt_token_ids) for req in requests] # sequences is just: "the current full input for this request, kept up to date"
    max_steps = max(req.max_new_tokens for req in requests)

    for step in range(max_steps):
        # figure out which requests still need tokens this step
        active_idx = [
            i for i, req in enumerate(requests)
            if len(req.generated_token_ids) < req.max_new_tokens
        ]
        if not active_idx:
            break

        # pad the ACTIVE sequences to the longest active sequence
        active_seqs = [sequences[i] for i in active_idx]
        max_len = max(len(s) for s in active_seqs)
        pad_id = model.tokenizer.pad_token_id

        input_ids = torch.full((len(active_idx), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(active_idx), max_len), dtype=torch.long) # attention_mask marks which columns are real tokens (1) vs padding (0), so the model knows to ignore the filler.
        for row, seq in enumerate(active_seqs):
            # left-pad so the "last token" position lines up at index -1
            offset = max_len - len(seq)
            input_ids[row, offset:] = torch.tensor(seq, dtype=torch.long)
            attention_mask[row, offset:] = 1

        # full forward pass over prompt + all generated tokens so far --
        # this is the recompute-from-scratch cost this phase pays
        logits = model.forward_batch(input_ids, attention_mask)

        for row, i in enumerate(active_idx):
            next_token = model.greedy_next_token(logits[row])
            requests[i].record_token(next_token)
            sequences[i].append(next_token)

    for req in requests:
        req.mark_finished()

    return requests
