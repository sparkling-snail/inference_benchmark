"""
Batched KV cache.

Extends the single-request cache (kv_cache_single.py) to a batch of requests that each
get their own per-sequence cache slot in the same shared cache object.
Split into two capabilities:

  - Checkpoint A (this file's core): start_batch / step /
    run_batch_to_completion -- a FIXED set of requests, processed
    together, each with correct per-row caching. Mirrors
    scheduler_naive.py's "batch locked once formed" semantics, but pays
    the cache-reuse cost instead of full recompute every step.
  - Checkpoint B: evict / admit -- lets requests join or leave a running
    batch without restarting everyone else. This is the part that's
    actually "continuous batching."

Requests of different lengths still get left-padded to a common width
(same reason as the static batcher: the underlying cache tensor is one rectangular
block per layer, every row forced to share the same physical length).
See model_wrapper.build_position_ids for why left-padding stays correct
under caching -- each row's position ids are derived from its own
attention mask row, so padding never shifts a row's own numbering.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache

from .model_wrapper import ModelWrapper
from .request import Request


@dataclass
class BatchedKVCache:
    """
    Row i of requests / pending_tokens / attention_mask / past_key_values
    always refers to the same sequence. evict() compacts rows and shifts
    every row above a dropped one down by one -- any row index held
    across a call to evict() is invalidated. Look a row up by identity
    (batch.requests.index(req)) if you need one after an evict.
    """

    requests: list[Request]
    pending_tokens: list[int]  # each row's most-recent generated token, not yet fed
    attention_mask: torch.Tensor  # (batch, seq_len), 1 = real token
    past_key_values: DynamicCache | None

    @property
    def batch_size(self) -> int:
        return len(self.requests)

    @property
    def seq_len(self) -> int:
        return self.attention_mask.shape[1]


def _left_pad_batch(model: ModelWrapper, requests: list[Request]) -> tuple[torch.Tensor, torch.Tensor]:
    for req in requests:
        req.prompt_token_ids = model.encode(req.prompt)

    max_len = max(len(req.prompt_token_ids) for req in requests)
    pad_id = model.tokenizer.pad_token_id

    input_ids = torch.full((len(requests), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(requests), max_len), dtype=torch.long)
    for row, req in enumerate(requests):
        offset = max_len - len(req.prompt_token_ids)
        input_ids[row, offset:] = torch.tensor(req.prompt_token_ids, dtype=torch.long)
        attention_mask[row, offset:] = 1
    return input_ids, attention_mask

# start a fix batch first, prefill the cache
def start_batch(model: ModelWrapper, requests: list[Request]) -> BatchedKVCache:
    """Prefills every request's prompt together and returns the initial batch state."""
    input_ids, attention_mask = _left_pad_batch(model, requests)
    logits, past_key_values = model.forward_batch_step(input_ids, attention_mask, past_key_values=None)

    pending_tokens = []
    for row, req in enumerate(requests):
        next_token = model.greedy_next_token(logits[row])
        req.record_token(next_token) #picked but not yet in cache
        pending_tokens.append(next_token)

    return BatchedKVCache(
        requests=list(requests),
        pending_tokens=pending_tokens,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
    )

# the first decode step
def step(model: ModelWrapper, batch: BatchedKVCache, eos_token_id: int | None) -> None:
    """Advances every row in the batch by one token."""
    if batch.batch_size == 0:
        return

    input_ids = torch.tensor([[t] for t in batch.pending_tokens], dtype=torch.long)
    batch.attention_mask = torch.cat(
        [batch.attention_mask, torch.ones((batch.batch_size, 1), dtype=torch.long)], dim=1
    )

    logits, batch.past_key_values = model.forward_batch_step(
        input_ids, batch.attention_mask, past_key_values=batch.past_key_values
    ) #batch.past_key_values: the updated cache, saved straight back into the folder, replacing the old cache.

    for row, req in enumerate(batch.requests):
        next_token = model.greedy_next_token(logits[row]) #gets the token with the highest logit score
        # EOS-aware guard: a plain length check would record extra tokens
        # for a row that emits EOS before its own max_new_tokens, making
        # it diverge from the single-sequence reference used to verify.
        if not req.is_finished(eos_token_id):
            req.record_token(next_token) #records the token in the request object
        batch.pending_tokens[row] = next_token #update the pending token for the row
        batch.pending_tokens[row] = next_token


def evict(batch: BatchedKVCache, finished_rows: list[int]) -> list[Request]:
    """
    Drops the given row indices from every part of the batch state --
    the cache (all layers, in place), the attention mask, and the
    Python-side bookkeeping lists -- and marks those requests finished.
    Row indices for any request NOT in finished_rows shift down to close
    the gap; look requests up by identity after calling this, not by
    the row index they had before.
    """
    keep = [i for i in range(batch.batch_size) if i not in finished_rows]
    keep_idx = torch.tensor(keep, dtype=torch.long)

    evicted = [batch.requests[i] for i in finished_rows]
    for req in evicted:
        req.mark_finished()

    batch.past_key_values.batch_select_indices(keep_idx)
    batch.attention_mask = batch.attention_mask[keep_idx]
    batch.requests = [batch.requests[i] for i in keep]
    batch.pending_tokens = [batch.pending_tokens[i] for i in keep]
    return evicted


def _left_pad_kv(legacy_cache, pad_len: int):
    """Left-pads every layer's key/value tensors by pad_len zero columns on the seq-len dim."""
    return tuple(
        (
            torch.cat(
                [torch.zeros(k.shape[0], k.shape[1], pad_len, k.shape[3], dtype=k.dtype, device=k.device), k],
                dim=2,
            ),
            torch.cat(
                [torch.zeros(v.shape[0], v.shape[1], pad_len, v.shape[3], dtype=v.dtype, device=v.device), v],
                dim=2,
            ),
        )
        for k, v in legacy_cache
    )


def admit(model: ModelWrapper, batch: BatchedKVCache, req: Request) -> None:
    """
    Brings a new request into a running batch mid-generation, without
    restarting or disturbing any existing row.

    The new request is prefilled on its own (batch size 1) to get its
    own cache. That cache's sequence length is usually different from
    the running batch's, but every row in a DynamicCache is physically
    forced to share one sequence-length dimension per layer -- so
    whichever side is shorter gets left-padded (zero columns + a 0 in
    the attention mask) up to the longer side's length before the two
    caches are concatenated along the batch dimension. Left-padding is
    safe here because build_position_ids derives each row's position
    ids from its own attention mask row (see model_wrapper.py) -- extra
    leading padding never perturbs a row's own position numbering.
    """
    req.prompt_token_ids = model.encode(req.prompt)
    prompt_len = len(req.prompt_token_ids)

    new_input_ids = torch.tensor([req.prompt_token_ids], dtype=torch.long)
    new_attention_mask = torch.ones((1, prompt_len), dtype=torch.long)
    logits, new_cache = model.forward_batch_step(new_input_ids, new_attention_mask, past_key_values=None)
    new_token = model.greedy_next_token(logits[0])
    req.record_token(new_token)

    target_len = max(batch.seq_len, prompt_len)
    # DynamicCache.to_legacy_cache()/from_legacy_cache() were removed in
    # newer transformers releases (the per-layer tensors now live at
    # cache.layers[i].keys / .values instead) -- this reads/rebuilds the
    # same (key, value) tuple-per-layer shape those used to produce, so
    # the merge logic below is otherwise unchanged.
    existing_legacy = tuple((layer.keys, layer.values) for layer in batch.past_key_values.layers)
    new_legacy = tuple((layer.keys, layer.values) for layer in new_cache.layers)

    if prompt_len < target_len:
        pad = target_len - prompt_len
        new_legacy = _left_pad_kv(new_legacy, pad)
        new_attention_mask = torch.cat(
            [torch.zeros((1, pad), dtype=torch.long), new_attention_mask], dim=1
        )
    elif batch.seq_len < target_len:
        pad = target_len - batch.seq_len
        existing_legacy = _left_pad_kv(existing_legacy, pad)
        batch.attention_mask = torch.cat(
            [torch.zeros((batch.batch_size, pad), dtype=torch.long), batch.attention_mask], dim=1
        )

    merged_legacy = tuple(
        (torch.cat([ek, nk], dim=0), torch.cat([ev, nv], dim=0))
        for (ek, ev), (nk, nv) in zip(existing_legacy, new_legacy)
    )
    batch.past_key_values = DynamicCache(ddp_cache_data=merged_legacy)
    batch.attention_mask = torch.cat([batch.attention_mask, new_attention_mask], dim=0)
    batch.requests.append(req)
    batch.pending_tokens.append(new_token)


def run_batch_to_completion(model: ModelWrapper, requests: list[Request]) -> list[Request]:
    """
    Runs a fixed batch of requests to completion using per-sequence KV
    caching instead of the static batcher's full-sequence recompute. No evict:
    a request that finishes early keeps being fed (harmlessly) until the
    whole batch is done -- that idle cost is exactly what Checkpoint B's
    evict() removes.
    """
    batch = start_batch(model, requests)
    max_steps = max(req.max_new_tokens for req in requests)

    for _ in range(max_steps - 1):  # prefill already produced one token per row
        step(model, batch, model.eos_token_id)

    for req in batch.requests:
        req.mark_finished()

    return batch.requests
