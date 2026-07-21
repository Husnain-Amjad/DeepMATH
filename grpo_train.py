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
from evaluator import parse_model_output, arithmetic_consistency_score
from determinism import set_all_seeds
from storage_utils import ensure_dir, add_destination_args, dispatch_destination
from templates import render_prompt_only

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


def load_grpo_prompts(path, tokenizer):
    """
    Builds the RL prompt set directly from the raw {problem, subject, level,
    gold_boxed} jsonl using this run's own tokenizer/template (templates.py) -
    correct per-model, and no longer needs to string-search a baked chat
    template that doesn't exist in the data anymore (build_sft_dataset now
    stores raw fields, not a pre-rendered "text" column - see data_pipeline.py).
    """
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = render_prompt_only(tokenizer, obj["problem"])
            rows.append({
                "prompt": prompt,
                "gold_boxed": obj["gold_boxed"],
                "subject": obj.get("subject", "unknown"),
                "level": obj.get("level", "unknown"),
            })
    return Dataset.from_list(rows)


def _completion_text(completion):
    """completion may be a list of chat turns (dict) or a plain string, depending on TRL version."""
    return completion if isinstance(completion, str) else completion[-1]["content"]


def correctness_reward_fn(completions, gold_boxed, **kwargs):
    """1.0 correct boxed answer, 0.1 wrong-but-well-formatted, 0.0 no boxed answer at all."""
    return [score_completion(_completion_text(c), g) for c, g in zip(completions, gold_boxed)]


def format_reward_fn(completions, **kwargs):
    """
    Format fidelity: 0.5 for having a <think> section, 0.5 for having a resolvable
    boxed answer - reuses evaluator.py's own parser so 'has the model learned the
    trained format' is scored identically here and at evaluation time.
    """
    rewards = []
    for completion in completions:
        parsed = parse_model_output(_completion_text(completion))
        r = (0.5 if parsed["has_think"] else 0.0) + (0.5 if parsed["has_boxed"] else 0.0)
        rewards.append(r)
    return rewards


def persistence_reward_fn(completions, **kwargs):
    """
    Persistence: rewards committing to ONE final answer rather than flip-flopping
    between several different \\boxed{...} values within a single completion - a
    proxy for not abandoning/reversing a derivation mid-trace. 1.0 for exactly one
    distinct boxed value, 0.0 for none, an increasing penalty for each additional
    distinct value beyond the first.
    """
    rewards = []
    for completion in completions:
        text = _completion_text(completion)
        boxed_vals = {v.strip() for v in re.findall(r"\\boxed\{(.*?)\}", text)}
        if len(boxed_vals) == 0:
            rewards.append(0.0)
        elif len(boxed_vals) == 1:
            rewards.append(1.0)
        else:
            rewards.append(max(-1.0, -0.3 * (len(boxed_vals) - 1)))
    return rewards


def chain_stability_reward_fn(completions, **kwargs):
    """
    Chain stability: reuses evaluator.py's rule-based arithmetic-consistency proxy
    (symbolic verification of extractable 'A = B' step assertions) as a per-step
    validity signal - the fraction of checkable assertions verified true. Returns
    a neutral 0.5 when a completion has no checkable assertions at all, rather than
    0.0, so purely-verbal-but-correct reasoning isn't penalized for not containing
    an equation to check.
    """
    rewards = []
    for completion in completions:
        parsed = parse_model_output(_completion_text(completion))
        n_checkable, n_correct = 0, 0
        for _, step_text in parsed["steps"]:
            nc, ncorr = arithmetic_consistency_score(step_text)
            n_checkable += nc
            n_correct += ncorr
        rewards.append((n_correct / n_checkable) if n_checkable else 0.5)
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
    ap.add_argument("--w_correctness", type=float, default=1.0,
                     help="weight for the correctness reward component")
    ap.add_argument("--w_format", type=float, default=0.2,
                     help="weight for the format-fidelity reward component (<think>/boxed)")
    ap.add_argument("--w_persistence", type=float, default=0.15,
                     help="weight for the persistence reward component (commits to one "
                          "final answer rather than flip-flopping between several)")
    ap.add_argument("--w_chain_stability", type=float, default=0.25,
                     help="weight for the chain-stability reward component (rule-based "
                          "step-level arithmetic consistency, reused from evaluator.py)")
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

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_grpo_prompts(args.data, tokenizer)

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

    reward_funcs = [correctness_reward_fn, format_reward_fn, persistence_reward_fn, chain_stability_reward_fn]
    reward_weights = [args.w_correctness, args.w_format, args.w_persistence, args.w_chain_stability]
    if hasattr(grpo_config, "reward_weights"):
        grpo_config.reward_weights = reward_weights
        print(f"[grpo_train] reward weights (correctness/format/persistence/chain_stability): {reward_weights}")
    else:
        print(f"[grpo_train] WARNING: this TRL version's GRPOConfig has no reward_weights "
              f"field - all {len(reward_funcs)} reward components will be summed with EQUAL "
              f"weight instead of the requested {reward_weights}. Upgrade trl for weighted "
              f"multi-component rewards, or fold weighting into a single custom reward_fn.")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
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
