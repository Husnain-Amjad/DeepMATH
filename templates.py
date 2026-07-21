"""
Per-model prompt rendering, done at train/eval time from each model's OWN
tokenizer rather than baked into the dataset at build-sft time. This is what
makes one sft_data.jsonl usable across all 7 models: build-sft stores raw
{problem, think, solution} fields, and this module renders them into
whatever format the model currently being trained/evaluated actually expects
(ChatML, Alpaca, plain completion, user-only chat template, etc.) - covering
exactly the model diversity discussed (Qwen ChatML, WizardMath Alpaca,
DeepSeekMath base with no template, DeepSeekMath-RL with no system role).

Both render_sft_example() and render_prompt_only() call the SAME
build_prompt_prefix() internally, so the eval/GRPO prompt is guaranteed to be
an exact prefix of the SFT training text - no separate hardcoded template to
drift out of sync.
"""

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful mathematical problem solver. First think through the "
    "problem's structure, then solve it."
)


def build_prompt_prefix(tokenizer, problem: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """
    Returns the prompt prefix up to (not including) the model's own response,
    using the tokenizer's native chat template if it has one, with graceful
    fallback for tokenizers that reject a system role (e.g. deepseek-math-7b-rl),
    and a plain-text fallback for base models with no chat template at all
    (e.g. deepseek-math-7b-base, Qwen2.5-Math-1.5B base).
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": problem}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass  # some tokenizers (deepseek-math-7b-rl) reject a system role entirely
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{system_prompt}\n\n{problem}"}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass  # fall through to plain-text below if even user-only rendering fails

    # No usable chat template - base/completion model. Plain-text prompt.
    return f"{system_prompt}\n\n{problem}\n\n"


def render_sft_example(tokenizer, problem: str, think: str, solution: str,
                        system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Full training text: prompt prefix + the <think>/<solution> completion + EOS."""
    prefix = build_prompt_prefix(tokenizer, problem, system_prompt)
    completion = f"<think>\n{think}\n</think>\n<solution>\n{solution}\n</solution>"
    eos = tokenizer.eos_token or ""
    return prefix + completion + eos


def render_prompt_only(tokenizer, problem: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Prompt-only prefix for generation (GRPO rollouts, eval) - exact prefix of render_sft_example."""
    return build_prompt_prefix(tokenizer, problem, system_prompt)
