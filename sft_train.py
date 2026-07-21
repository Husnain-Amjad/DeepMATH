"""
Stage 1 / Stage 4 SFT training.

Usage:
  # Stage 1 cold-start, 1.5B, full fine-tune
  python sft_train.py --model Qwen/Qwen2.5-Math-1.5B --data outputs/sft_data.jsonl \
      --mode full --output_dir ckpts/qwen1.5b_stage1

  # Stage 1 cold-start, 7B, LoRA
  python sft_train.py --model Qwen/Qwen2.5-Math-7B --data outputs/sft_data.jsonl \
      --mode lora --output_dir ckpts/qwen7b_stage1

  # Stage 4 replay-mixed round 2 (original + augmented, resumed from stage1 ckpt)
  python sft_train.py --model ckpts/qwen7b_stage1 --data outputs/sft_data.jsonl \
      --extra_data outputs/semantic_aug.jsonl outputs/numeric_aug.jsonl outputs/multi_solution.jsonl \
      --replay_ratio 0.7 --mode lora --output_dir ckpts/qwen7b_stage4
"""

import argparse
import json
import os
import random
from collections import defaultdict

import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
from trl import SFTTrainer, SFTConfig

from determinism import set_all_seeds
from storage_utils import ensure_dir, add_destination_args, dispatch_destination, push_to_hf
from templates import render_sft_example

# ---------------------------------------------------------------------------
# ROCm / CUDA portability notes:
#   - ROCm PyTorch builds expose the same torch.cuda.* API namespace as CUDA
#     builds, so device_map="auto", .to("cuda"), bf16, etc. all work unchanged.
#   - flash_attention_2 (the pip package) is CUDA-only. We use "sdpa" instead,
#     which dispatches to PyTorch's native scaled-dot-product-attention kernels
#     - on ROCm (MI200/MI300) this uses the AOTriton backend and gets most of
#     flash-attention's speed without needing a ROCm-specific flash-attn build.
#   - bitsandbytes has poor/partial ROCm support. Quantized LoRA (QLoRA) is
#     therefore OFF by default; plain bf16 LoRA is used instead, which is
#     fully portable and, for a 7B model on an 80GB card, isn't memory-starved
#     enough to need 4-bit loading anyway. Pass --use_bnb only on a CUDA box
#     with bitsandbytes installed if you specifically want 4-bit.
# ---------------------------------------------------------------------------

def is_rocm() -> bool:
    return torch.cuda.is_available() and bool(getattr(torch.version, "hip", None))


def best_attn_implementation() -> str:
    """sdpa is portable across CUDA and ROCm; flash_attention_2 is CUDA-only."""
    return "sdpa"


def find_latest_checkpoint(output_dir):
    """Returns the checkpoint-N directory with the highest N under output_dir, or None."""
    if not os.path.isdir(output_dir):
        return None
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not ckpts:
        return None

    def step_num(d):
        try:
            return int(d.split("-")[-1])
        except ValueError:
            return -1

    ckpts.sort(key=step_num)
    return os.path.join(output_dir, ckpts[-1])


class PushCheckpointCallback(TrainerCallback):
    """
    Pushes each RAW checkpoint (adapter-only for LoRA, full weights for full-FT -
    exactly what the Trainer just wrote to disk) to a distinct subfolder of the HF
    repo as soon as it's saved. Deliberately does NOT touch trainer.model or merge
    anything here: merge_and_unload() mutates the live model in place and would
    corrupt the ongoing training state if called mid-run, and loading a second
    full copy of the base model to merge separately would risk OOM on a single
    GPU that's already holding the training run. Merging happens later, decoupled,
    via merge_lora.py or run_multi_checkpoint_eval.py.
    """
    def __init__(self, hf_repo_id, hf_repo_type, hf_private, hf_token):
        self.hf_repo_id = hf_repo_id
        self.hf_repo_type = hf_repo_type
        self.hf_private = hf_private
        self.hf_token = hf_token

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            print(f"[push-checkpoint] expected {ckpt_dir} after save but it's missing - skipping push")
            return control
        label = f"checkpoint-{state.global_step}"
        try:
            push_to_hf(ckpt_dir, self.hf_repo_id, repo_type=self.hf_repo_type,
                       private=self.hf_private, path_in_repo=label, token=self.hf_token,
                       commit_message=f"{label} (epoch={state.epoch:.2f})")
        except Exception as e:
            print(f"[push-checkpoint] WARNING: push failed for {label} "
                  f"({type(e).__name__}: {e}) - training continues; push it manually "
                  f"later with push_artifact.py --path {ckpt_dir} --push_to hf "
                  f"--hf_repo_id {self.hf_repo_id} --hf_repo_type {self.hf_repo_type} "
                  f"--hf_path_in_repo {label}")
        return control


DEFAULT_THINK_PLACEHOLDER = "Relevant skills: (not labeled - augmented example)"


def _normalize_row(obj):
    """Guarantees the fields templates.py and the replay strategies need are present,
    regardless of whether this row came from build_sft_dataset (has problem/think/
    solution/skills) or an augmentation file (semantic/numeric perturbation - has
    problem/solution only, no think section or skills, since perturbation doesn't
    regenerate the reasoning trace)."""
    return {
        "problem": obj["problem"],
        "think": obj.get("think") or DEFAULT_THINK_PLACEHOLDER,
        "solution": obj["solution"],
        "subject": obj.get("subject", "unknown"),
        "level": str(obj.get("level", "unknown")),
        "gold_boxed": obj.get("gold_boxed", ""),
        "skills": obj.get("skills", ""),
    }


def load_jsonl_as_dataset(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "problem" in obj and "solution" in obj:
                    rows.append(_normalize_row(obj))
    return Dataset.from_list(rows)


def render_dataset_for_model(dataset, tokenizer):
    """Renders the final 'text' column using this run's OWN tokenizer/template
    (templates.render_sft_example) - this is what makes one raw sft_data.jsonl
    reusable across all 7 models instead of baking one hardcoded chat template
    into the data at build-sft time."""
    def _render(row):
        return {"text": render_sft_example(tokenizer, row["problem"], row["think"], row["solution"])}
    return dataset.map(_render, remove_columns=[c for c in dataset.column_names if c != "text"])


def _stratified_sample(dataset, n_target, key_fn, rng):
    """Round-robin samples across clusters defined by key_fn, so no single cluster
    dominates just because it's naturally more frequent in the source data - this
    is the 'balanced' and 'skill' replay strategies; 'random' skips this entirely."""
    buckets = defaultdict(list)
    for i, row in enumerate(dataset):
        buckets[key_fn(row)].append(i)
    for idxs in buckets.values():
        rng.shuffle(idxs)

    keys = list(buckets.keys())
    rng.shuffle(keys)
    selected = []
    cursors = {k: 0 for k in keys}
    while len(selected) < n_target and keys:
        progressed = False
        for k in list(keys):
            if cursors[k] < len(buckets[k]):
                selected.append(buckets[k][cursors[k]])
                cursors[k] += 1
                progressed = True
                if len(selected) >= n_target:
                    break
        if not progressed:
            break
    return dataset.select(selected[:n_target])


def build_replay_mixed_dataset(original_path, extra_paths, replay_ratio, strategy="random", seed=0):
    """
    strategy:
      "none"     - ignore extra_paths entirely; pure original data (explicit no-replay baseline).
      "random"   - uniform random sampling from original+augmented at replay_ratio (default,
                   matches the original behavior of this function).
      "balanced" - stratified sampling by (subject, level) cluster, so weak clusters being
                   augmented don't dominate the mix just by being oversampled.
      "skill"    - stratified sampling by primary skill label (first entry in the pipe-
                   separated 'skills' field), for skill-balanced replay.
    replay_ratio: fraction of the FINAL training set that comes from the original
    (unaugmented) data - applies to all strategies except "none".
    """
    rng = random.Random(seed)
    original = load_jsonl_as_dataset([original_path])

    if strategy == "none" or not extra_paths:
        if strategy != "none" and not extra_paths:
            print("[replay-mix] no --extra_data given - using original data only")
        return original

    augmented = load_jsonl_as_dataset(extra_paths)
    if len(augmented) == 0:
        return original

    n_total_target = len(original)
    n_original = int(n_total_target * replay_ratio)
    n_augmented = n_total_target - n_original

    if strategy == "random":
        orig_idx = list(range(len(original)))
        rng.shuffle(orig_idx)
        aug_idx = list(range(len(augmented)))
        rng.shuffle(aug_idx)
        orig_sample = original.select(orig_idx[:min(n_original, len(original))])
        aug_sample = augmented.select([aug_idx[i % len(aug_idx)] for i in range(n_augmented)])

    elif strategy == "balanced":
        cluster_key = lambda row: (row["subject"], row["level"])
        orig_sample = _stratified_sample(original, min(n_original, len(original)), cluster_key, rng)
        aug_sample = _stratified_sample(augmented, n_augmented, cluster_key, rng)

    elif strategy == "skill":
        def skill_key(row):
            skills = (row.get("skills") or "").split("|")
            return skills[0].strip() if skills and skills[0].strip() else "unknown"
        orig_sample = _stratified_sample(original, min(n_original, len(original)), skill_key, rng)
        aug_sample = _stratified_sample(augmented, n_augmented, skill_key, rng)

    else:
        raise ValueError(f"unknown replay strategy: {strategy}")

    mixed = concatenate_datasets([orig_sample, aug_sample]).shuffle(seed=seed)
    print(f"[replay-mix] strategy={strategy} original={len(orig_sample)} augmented={len(aug_sample)} "
          f"ratio_actual={len(orig_sample)/len(mixed):.2f}")
    return mixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="path to jsonl produced by build_sft_dataset")
    ap.add_argument("--extra_data", nargs="*", default=[],
                     help="augmented jsonl files (semantic/numeric/multi-solution) for round 2")
    ap.add_argument("--replay_ratio", type=float, default=1.0,
                     help="fraction of final set from original data; 1.0 = no augmentation mixed in")
    ap.add_argument("--replay_strategy", choices=["none", "random", "balanced", "skill"], default="random",
                     help="none=ignore augmented data entirely (no-replay baseline); "
                          "random=uniform mixing at --replay_ratio; balanced=stratify by "
                          "(subject, level) cluster; skill=stratify by primary skill label. "
                          "This is the replay-strategy comparison axis (random/balanced/skill/none).")
    ap.add_argument("--mode", choices=["full", "lora"], required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--per_device_batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--use_bnb", action="store_true", default=False,
                     help="4-bit QLoRA via bitsandbytes. CUDA-only - leave off on ROCm.")
    ap.add_argument("--torch_compile", action="store_true", default=False,
                     help="torch.compile(model) for extra throughput. Supported on both "
                          "CUDA and recent ROCm PyTorch builds; off by default since it "
                          "adds compile-time overhead on short runs.")
    ap.add_argument("--packing", action="store_true", default=True,
                     help="Pack multiple examples per sequence (TRL handles EOS-bounded "
                          "loss masking) to cut padding waste - meaningfully faster on "
                          "variable-length math solutions.")
    ap.add_argument("--seed", type=int, default=42,
                     help="controls weight init RNG, LoRA dropout masks, and data "
                          "shuffle order. Fixing this makes repeated runs on the SAME "
                          "machine/library stack close to reproducible - it does NOT "
                          "guarantee identical results across different GPUs/driver/"
                          "CUDA-vs-ROCm versions (floating-point reduction order differs "
                          "at the kernel level regardless of seed).")
    ap.add_argument("--strict_deterministic", action="store_true", default=False,
                     help="opt-in: forces deterministic algorithms where available. "
                          "Slower - off by default.")
    ap.add_argument("--skip_merge", action="store_true", default=False,
                     help="skip the automatic LoRA merge-and-save step (mode=lora only). "
                          "Use if you specifically want to keep only the adapter, e.g. "
                          "for vLLM's --enable-lora serving mode instead of a merged model.")
    ap.add_argument("--save_every_epochs", type=float, default=None,
                     help="checkpoint every N epochs (e.g. 0.5) instead of the default "
                          "once-per-epoch. Computed from the ACTUAL post-packing dataloader "
                          "length, not an analytical estimate, since --packing changes how "
                          "many optimizer steps an epoch actually takes.")
    ap.add_argument("--save_total_limit", type=int, default=None,
                     help="cap on local checkpoints kept on disk (HF Trainer deletes "
                          "older ones). Recommended alongside --push_every_checkpoint on "
                          "disk-constrained/ephemeral environments, since older ones are "
                          "already safe on HF once pushed.")
    ap.add_argument("--push_every_checkpoint", action="store_true", default=False,
                     help="push each raw checkpoint to --hf_repo_id as it's saved, under "
                          "a distinct subfolder per checkpoint - protects progress if the "
                          "session dies mid-run. Requires --push_to hf and --hf_repo_id.")
    ap.add_argument("--resume_from_checkpoint", type=str, default=None,
                     help="'auto' to resume from the latest checkpoint-N under --output_dir "
                          "if one exists, or an explicit checkpoint directory path. Omit to "
                          "start fresh. If your local disk was wiped (ephemeral session) "
                          "but you pushed checkpoints to HF, download the desired one first "
                          "(see TUTORIAL.md) then pass its local path here.")
    add_destination_args(ap, default_repo_type="model")
    args = ap.parse_args()

    if args.push_every_checkpoint and (args.push_to != "hf" or not args.hf_repo_id):
        raise SystemExit("[sft_train] --push_every_checkpoint requires --push_to hf "
                          "and --hf_repo_id (checked before training starts, not after).")

    set_all_seeds(args.seed, strict_deterministic=args.strict_deterministic)
    torch.backends.cuda.matmul.allow_tf32 = True  # no-op on ROCm, free speedup on CUDA

    if args.use_bnb and is_rocm():
        print("[warn] --use_bnb requested on a ROCm device; bitsandbytes ROCm support is "
              "partial/unstable. Falling back to plain bf16 LoRA (--use_bnb ignored).")
        args.use_bnb = False

    # Save the full training spec immediately - before any expensive work starts -
    # so experiment_ledger.py always has a record of exactly what was run, even if
    # the session dies partway through training. Idempotent: safe to call again below.
    ensure_dir(args.output_dir)
    training_config = vars(args).copy()
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2, default=str)
    print(f"[sft_train] saved training_config.json -> {args.output_dir}/training_config.json")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.extra_data:
        train_ds_raw = build_replay_mixed_dataset(args.data, args.extra_data, args.replay_ratio,
                                                    strategy=args.replay_strategy, seed=args.seed)
    else:
        train_ds_raw = load_jsonl_as_dataset([args.data])
    print(f"[sft_train] rendering {len(train_ds_raw)} examples with {args.model}'s own "
          f"chat template (or plain-completion format if it has none) via templates.py")
    train_ds = render_dataset_for_model(train_ds_raw, tokenizer)

    quantization_config = None
    if args.use_bnb:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="bfloat16" if args.bf16 else "auto",
        device_map="auto" if args.mode == "lora" else None,
        attn_implementation=best_attn_implementation(),
        quantization_config=quantization_config,
    )
    if args.torch_compile:
        model = torch.compile(model)

    peft_config = None
    if args.mode == "lora":
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
        )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=args.bf16,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=10,
        save_strategy="epoch" if args.save_every_epochs is None else "steps",
        save_total_limit=args.save_total_limit,
        max_seq_length=args.max_seq_len,
        packing=args.packing,  # TRL packs with EOS-bounded loss masking, so
                                # <think>/<solution> boundaries stay intact even packed
        gradient_checkpointing=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if args.save_every_epochs is not None:
        # Measure the ACTUAL post-packing dataloader length rather than estimating
        # analytically - --packing changes how many examples become one training
        # sequence, so len(train_ds) alone doesn't tell you steps-per-epoch.
        n_batches_per_epoch = len(trainer.get_train_dataloader())
        steps_per_epoch = max(1, n_batches_per_epoch // args.grad_accum)
        save_steps = max(1, round(args.save_every_epochs * steps_per_epoch))
        trainer.args.save_steps = save_steps
        print(f"[sft_train] measured {n_batches_per_epoch} batches/epoch -> "
              f"{steps_per_epoch} optimizer steps/epoch -> checkpointing every "
              f"{save_steps} steps (~{args.save_every_epochs} epoch)")

    if args.push_every_checkpoint:
        trainer.add_callback(PushCheckpointCallback(
            hf_repo_id=args.hf_repo_id, hf_repo_type=args.hf_repo_type,
            hf_private=args.hf_private, hf_token=args.hf_token,
        ))
        print(f"[sft_train] will push each checkpoint to hf://{args.hf_repo_id} "
              f"under its own checkpoint-N subfolder as training progresses")

    resume_path = None
    if args.resume_from_checkpoint == "auto":
        resume_path = find_latest_checkpoint(args.output_dir)
        if resume_path:
            print(f"[sft_train] --resume_from_checkpoint auto -> resuming from {resume_path}")
        else:
            print(f"[sft_train] --resume_from_checkpoint auto but no checkpoint-N found "
                  f"under {args.output_dir} - starting fresh")
    elif args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint
        print(f"[sft_train] resuming from explicit checkpoint: {resume_path}")

    ensure_dir(args.output_dir)
    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[sft_train] saved to {args.output_dir}")

    # This is the artifact eval/vLLM/GRPO should actually load: for full-FT it's
    # just output_dir; for LoRA it's the merged dir (adapter-only dirs can't be
    # loaded directly by vLLM - see merge_lora.py for the same fix applied to
    # older/intermediate checkpoints that predate this auto-merge).
    final_model_dir = args.output_dir

    if args.mode == "lora" and not args.skip_merge:
        merged_dir = args.output_dir.rstrip("/") + "_merged"
        print(f"[sft_train] mode=lora -> attempting merge-and-save to {merged_dir} "
              f"(this is what you should point --model at for eval/vLLM/GRPO).")
        try:
            ensure_dir(merged_dir)
            merged_model = trainer.model.merge_and_unload()
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f"[sft_train] merged model saved -> {merged_dir}")
            final_model_dir = merged_dir
        except Exception as e:
            print(f"[sft_train] WARNING: LoRA merge failed ({type(e).__name__}: {e}). "
                  f"The adapter-only checkpoint at {args.output_dir} is still valid and "
                  f"saved, but vLLM/transformers can't load it directly - run "
                  f"`python merge_lora.py --base_model {args.model} "
                  f"--adapter_path {args.output_dir} --out {merged_dir}` manually once "
                  f"you've resolved the issue. Common cause: base model was loaded "
                  f"quantized (--use_bnb) - LoRA can't merge into quantized weights; "
                  f"rerun without --use_bnb if you need a merged model.")
    elif args.mode == "lora" and args.skip_merge:
        print(f"[sft_train] --skip_merge set: leaving {args.output_dir} as an "
              f"adapter-only checkpoint. Use merge_lora.py later if you need a "
              f"standalone model, or serve it via vLLM's --enable-lora mode directly.")

    dispatch_destination(final_model_dir, args)


if __name__ == "__main__":
    main()
