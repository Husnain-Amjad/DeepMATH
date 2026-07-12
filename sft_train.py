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

import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig

from determinism import set_all_seeds
from storage_utils import ensure_dir, add_destination_args, dispatch_destination

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


def load_jsonl_as_dataset(paths, text_field="text"):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if text_field in obj:
                    rows.append({"text": obj[text_field]})
    return Dataset.from_list(rows)


def build_replay_mixed_dataset(original_path, extra_paths, replay_ratio, seed=0):
    """
    replay_ratio: fraction of the FINAL training set that comes from the original
    (unaugmented) data. e.g. 0.7 -> 70% original, 30% augmented, preventing the
    weak-domain oversampling from crowding out previously-solid domains.
    """
    rng = random.Random(seed)
    original = load_jsonl_as_dataset([original_path])
    augmented = load_jsonl_as_dataset(extra_paths) if extra_paths else None

    if augmented is None or len(augmented) == 0:
        return original

    n_total_target = len(original)  # keep dataset size stable across rounds
    n_original = int(n_total_target * replay_ratio)
    n_augmented = n_total_target - n_original

    orig_idx = list(range(len(original)))
    rng.shuffle(orig_idx)
    aug_idx = list(range(len(augmented)))
    rng.shuffle(aug_idx)

    orig_sample = original.select(orig_idx[:min(n_original, len(original))])
    aug_sample = augmented.select(
        [aug_idx[i % len(aug_idx)] for i in range(min(n_augmented, n_augmented))]
    )
    mixed = concatenate_datasets([orig_sample, aug_sample]).shuffle(seed=seed)
    print(f"[replay-mix] original={len(orig_sample)} augmented={len(aug_sample)} "
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
    add_destination_args(ap, default_repo_type="model")
    args = ap.parse_args()

    set_all_seeds(args.seed, strict_deterministic=args.strict_deterministic)
    torch.backends.cuda.matmul.allow_tf32 = True  # no-op on ROCm, free speedup on CUDA

    if args.use_bnb and is_rocm():
        print("[warn] --use_bnb requested on a ROCm device; bitsandbytes ROCm support is "
              "partial/unstable. Falling back to plain bf16 LoRA (--use_bnb ignored).")
        args.use_bnb = False

    if args.extra_data:
        train_ds = build_replay_mixed_dataset(args.data, args.extra_data, args.replay_ratio)
    else:
        train_ds = load_jsonl_as_dataset([args.data])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        save_strategy="epoch",
        max_length=args.max_seq_len,
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
    ensure_dir(args.output_dir)
    trainer.train()
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
