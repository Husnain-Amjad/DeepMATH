"""
Stage 5: pure rule-based GRPO on Qwen2.5-Math-7B + LoRA.

No PRM resident during RL - only policy + reference model, per the memory
budget discussion (this is what actually fits on a single 80GB A100 at
useful group sizes / sequence lengths).

Usage:
  python grpo_train.py --model ckpts/qwen7b_stage4 --data outputs/sft_data.jsonl \
      --output_dir ckpts/qwen7b_grpo --num_generations 8 --max_new_tokens 1024
"""

import argparse
import json
import re

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reward_fn import score_completion, extract_boxed
from determinism import set_all_seeds
from storage_utils import ensure_dir, add_destination_args, dispatch_destination

# ---------------------------------------------------------------------------
# ROCm notes:
#   - The policy/reference models load fine on ROCm PyTorch as-is; we force
#     attn_implementation="sdpa" (portable) instead of flash_attention_2
#     (CUDA-only pip wheel).
#   - vLLM has a ROCm build but it's a separate install path (ROCm docker
#     image or the vllm-rocm wheels), not a drop-in `pip install vllm`. We
#     probe for a working vLLM at runtime and fall back to TRL's built-in
#     HF-generate rollout backend if it's unavailable, so this script runs
#     on a ROCm box even without vLLM installed (slower rollouts, but correct).
# ---------------------------------------------------------------------------

def vllm_available() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except Exception:
        return False


def is_rocm() -> bool:
    return torch.cuda.is_available() and bool(getattr(torch.version, "hip", None))


def load_grpo_prompts(path):
    """
    Builds the RL prompt set from the same jsonl used for SFT, but strips the
    <think>/<solution> assistant turn - GRPO needs prompts only, the model
    generates its own completion which gets scored by reward_fn.
    """
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # reconstruct the user-turn-only prompt (everything up to and
            # including "<|im_start|>assistant\n")
            text = obj["text"]
            cut = text.find("<|im_start|>assistant")
            prompt = text[:cut] + "<|im_start|>assistant\n"
            rows.append({
                "prompt": prompt,
                "gold_boxed": obj["gold_boxed"],
                "subject": obj.get("subject", "unknown"),
                "level": obj.get("level", "unknown"),
            })
    return Dataset.from_list(rows)


def make_reward_function():
    """
    TRL's GRPOTrainer calls the reward function with the batch of completions
    and passes through any extra dataset columns as kwargs (here: gold_boxed).
    Returns a list of float rewards, one per completion.
    """
    def reward_fn(completions, gold_boxed, **kwargs):
        rewards = []
        for completion, gold in zip(completions, gold_boxed):
            # completion may be a list of chat turns (dict) or plain string
            text = completion if isinstance(completion, str) else completion[-1]["content"]
            rewards.append(score_completion(text, gold))
        return rewards

    return reward_fn


def format_length_penalty(completions, **kwargs):
    """
    Optional secondary reward: mild penalty for pathologically short completions
    that skip reasoning (e.g. just emitting \\boxed{...} with no derivation) or
    for completions that never close a boxed answer at all. Kept small relative
    to correctness reward so it shapes behavior without dominating it.
    """
    rewards = []
    for completion in completions:
        text = completion if isinstance(completion, str) else completion[-1]["content"]
        has_box = extract_boxed(text) is not None
        n_tokens_approx = len(text.split())
        penalty = 0.0
        if not has_box:
            penalty -= 0.2
        if n_tokens_approx < 15:
            penalty -= 0.1
        rewards.append(penalty)
    return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_generations", type=int, default=8,
                     help="GRPO group size (samples per prompt). Reduce this first "
                          "if you run out of memory - it does not change reward "
                          "correctness, only advantage-estimate variance.")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_prompt_length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--per_device_batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0,
                     help="KL penalty coefficient. 0.0 matches DAPO-style GRPO "
                          "(no KL term); set >0 for vanilla GRPO stability.")
    ap.add_argument("--use_vllm", action="store_true", default=True)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42,
                     help="see sft_train.py --seed docstring - same scope/caveats apply, "
                          "plus GRPO rollout sampling itself is stochastic per-step "
                          "regardless of this seed (that's the point of on-policy sampling).")
    ap.add_argument("--strict_deterministic", action="store_true", default=False)
    add_destination_args(ap, default_repo_type="model")
    args = ap.parse_args()

    set_all_seeds(args.seed, strict_deterministic=args.strict_deterministic)
    torch.backends.cuda.matmul.allow_tf32 = True  # no-op on ROCm, free speedup on CUDA

    if args.use_vllm and not vllm_available():
        print("[warn] --use_vllm requested but vLLM isn't importable in this environment. "
              "On ROCm, vLLM needs its ROCm-specific build (docker image or vllm-rocm "
              "wheels), not plain `pip install vllm`. Falling back to TRL's HF-generate "
              "rollout backend - correct, but noticeably slower rollouts.")
        args.use_vllm = False

    ensure_dir(args.output_dir)
    train_ds = load_grpo_prompts(args.data)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build the model explicitly (rather than passing a bare model-name string
    # to GRPOTrainer) so we control attn_implementation for CUDA/ROCm portability.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        seed=args.seed,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        num_train_epochs=args.num_train_epochs,
        beta=args.beta,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=args.use_vllm,          # rollouts via vLLM - critical for throughput
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[make_reward_function(), format_length_penalty],
        args=grpo_config,
        train_dataset=train_ds,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[grpo_train] saved to {args.output_dir}")
    dispatch_destination(args.output_dir, args)


if __name__ == "__main__":
    main()
