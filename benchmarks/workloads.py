from __future__ import annotations

from itertools import cycle, islice

from .schema import WorkloadCase

BASE_PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time in a small village,",
    "The best way to learn a new programming language is",
    "In the year 2050, cities will",
    "My favorite recipe for a quick dinner is",
    "Climate change is affecting the way we",
]


def build_prompts(case: WorkloadCase) -> list[str]:
    prompts = list(islice(cycle(BASE_PROMPTS), case.num_requests))
    return [_expand_prompt(prompt, case.prompt_chars) for prompt in prompts]


def _expand_prompt(prompt: str, target_chars: int) -> str:
    if len(prompt) >= target_chars:
        return prompt[:target_chars]

    pieces = [prompt]
    while len(" ".join(pieces)) < target_chars:
        pieces.append(prompt)
    return " ".join(pieces)[:target_chars]
