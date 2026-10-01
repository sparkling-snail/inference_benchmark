"""
Thin wrapper around a HuggingFace causal LM.

Deliberately NOT reimplementing the transformer itself -- the point of
this project is the request queue, scheduler, and KV cache *management*,
not re-deriving attention. Swap MODEL_NAME for a small Llama/Qwen later
without touching scheduler code.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2"  # ~124M params, good for fast local iteration


def build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Per-row position ids derived from the attention mask, so a left-padded
    row's real tokens get position 0, 1, 2... starting at its own first
    real token -- not at column 0 of the padded tensor. 
    
    Without this,
    GPT-2's learned position embeddings get misaligned for any row shorter
    than the batch's longest, since the model's default position ids are
    a single arange() shared across every row regardless of padding.

    the whole point of position_ids is to tell GPT-2 "this token is the 1st word, and so on."
    """
    position_ids = attention_mask.long().cumsum(-1) - 1 #position_ids tells the model "here's where each word actually sits in the sentence,
    position_ids.masked_fill_(attention_mask == 0, 1)  # mask out padding tokens, since we got attention_mask as source of truth
    return position_ids


class ModelWrapper:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            # GPT-2 has no pad token by default -- needed for batched padding, so set it to end of sequence token first
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name) #load model
        self.model.to(self.device) # move model to device
        self.model.eval() # switch to evaluation mode

        self.eos_token_id = self.tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.no_grad() # disable gradient calculation for inference
    def forward_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        One forward pass over a padded batch.
        Returns logits for the last position of each sequence.
        Shapes: input_ids/attention_mask -> (batch, seq_len)
        """
        input_ids = input_ids.to(self.device) # the padded token numbers
        attention_mask = attention_mask.to(self.device) # [0,1,1] / [1,1,1] — which columns are real
        position_ids = build_position_ids(attention_mask) # [1,0,1] / [0,1,2] — correct word-order labels, just fixed up
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids) #forward pass over the batch
        last_token_logits = outputs.logits[:, -1, :]  # (batch, vocab) #You don't care about predictions at every position — you only care about "what comes next, after everything we've fed in so far." That's always the last column of each row.
        return last_token_logits

    @torch.no_grad()
    def forward_step(self, input_ids: torch.Tensor, past_key_values=None):
        """
        One forward pass step, reusing a KV cache across calls.

        First call: pass the full prompt (1, prompt_len) with
        past_key_values=None. Every call after that: pass just the
        single newest token (1, 1) plus the past_key_values returned
        by the previous call -- the model only recomputes attention
        for that one new position instead of the whole sequence.

        Returns (last_token_logits, new_past_key_values).
        """
        input_ids = input_ids.to(self.device)
        outputs = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        last_token_logits = outputs.logits[0, -1, :]  # (vocab,) -- batch size 1
        return last_token_logits, outputs.past_key_values

    @torch.no_grad()
    def forward_batch_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values=None,
    ):
        """
        Batched analog of forward_step -- multiple sequences, each with
        its own growing KV cache slot in the same batched cache object.

        attention_mask always covers the FULL sequence so far, including
        input_ids: shape (batch, prompt_len) on the first (prefill) call
        with past_key_values=None, then (batch, total_len_so_far) on
        every call after that, where input_ids is just the newest token
        per row, shape (batch, 1).

        Returns (last_token_logits, new_past_key_values).
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        full_position_ids = build_position_ids(attention_mask)
        positxion_ids = full_position_ids[:, -input_ids.shape[1]:]
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        last_token_logits = outputs.logits[:, -1, :]  # (batch, vocab)
        return last_token_logits, outputs.past_key_values

    def greedy_next_token(self, logits_row: torch.Tensor) -> int:
        return int(torch.argmax(logits_row).item())
x