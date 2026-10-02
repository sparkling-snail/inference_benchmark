"""
Phase 2 -- single-request KV cache.

Generates one sequence token-by-token, reusing the transformer's
key/value cache between steps instead of recomputing the full
sequence like Phase 1 does. After the initial prompt pass, only the
newest token is fed into the model each step; past_key_values carries
the rest of the sequence's attention state forward.

Scoped to one request at a time on purpose -- Phase 3 is where the
cache has to survive requests being admitted/evicted mid-batch, which
is a much easier bug to introduce than to notice. Getting a single
sequence bit-for-bit correct first (see verify_kv_cache.py) isolates
that risk.
"""

import torch

from .model_wrapper import ModelWrapper
from .request import Request


def run_single_with_cache(model: ModelWrapper, req: Request) -> Request:
    """Generates req.max_new_tokens (or until EOS) using a growing KV cache."""
    req.prompt_token_ids = model.encode(req.prompt)
    next_input = torch.tensor([req.prompt_token_ids], dtype=torch.long) #turns the Python list into a PyTorch tensor, which is what the model needs.
    past_key_values = None

    while not req.is_finished(model.eos_token_id):
        logits, past_key_values = model.forward_step(next_input, past_key_values)
        next_token = model.greedy_next_token(logits) #gets the token with the highest logit score
        req.record_token(next_token) #records the token in the request object
        # every step after the first feeds only the one new token --
        # the cache already holds attention state for everything before it
        next_input = torch.tensor([[next_token]], dtype=torch.long)

    req.mark_finished()
    return req
