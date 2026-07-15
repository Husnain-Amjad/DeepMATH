"""
Evaluates every checkpoint produced by ONE continuous sft_train.py run (e.g.
--save_every_epochs 0.5 over --epochs 2 gives checkpoint-N at 0.5, 1.0, 1.5, 2.0
epochs) and produces a single accuracy-vs-epoch comparison via evaluator.py -
this is what replaces launching training separately for each eval point.

For LoRA runs, each checkpoint is merged into a throwaway temp directory before
evaluation (never touching the original adapter checkpoint), since vLLM/HF
generation needs a standalone model, not an adapter-only directory.

Usage:
  python run_multi_checkpoint_eval.py --checkpoints_dir ckpts/qwen7b_run \
      --base_model Qwen/Qwen2.5-Math-7B --mode lora \
      --split test --out-dir outputs/checkpoint_eval

  # if some checkpoints were only ever pushed to HF and the local disk was wiped:
  python run_multi_checkpoint_eval.py --checkpoints_dir ckpts/qwen7b_run \
      --hf_repo_id me/my-model --base_model Qwen/Qwen2.5-Math-7B --mode lora \
      --split test --out-dir outputs/checkpoint_eval
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from storage_utils import ensure_dir, require_input_path


def find_all_checkpoints(checkpoints_dir):
    """Returns [(step, path), ...] sorted by step, for every checkpoint-N under checkpoints_dir."""
    if not os.path.isdir(checkpoints_dir):
        return []
    out = []
    for d in os.listdir(checkpoints_dir):
        m = re.match(r"^checkpoint-(\d+)$", d)
        if m:
            out.append((int(m.group(1)), os.path.join(checkpoints_dir, d)))
    return sorted(out)


def download_checkpoint_from_hf(hf_repo_id, checkpoint_label, dest_dir):
    """Pulls one checkpoint-N subfolder back down from HF (for when local disk was wiped)."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=hf_repo_id, allow_patterns=[f"{checkpoint_label}/*"])
    src = os.path.join(path, checkpoint_label)
    shutil.copytree(src, dest_dir, dirs_exist_ok=True)
    return dest_dir


def merge_checkpoint(base_model, adapter_path, out_dir):
    """Calls merge_lora.py as a subprocess into a throwaway directory - never touches
    the original adapter checkpoint, so this is safe to run on every checkpoint in a loop."""
    result = subprocess.run(
        [sys.executable, "merge_lora.py", "--base_model", base_model,
         "--adapter_path", adapter_path, "--out", out_dir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"merge_lora.py failed for {adapter_path}:\n{result.stderr}")
    return out_dir


def run_eval_and_score(model_path, split, run_name, out_dir, seed, limit=None):
    """Calls run_eval.py then evaluator.py --score as subprocesses, returns the summary path."""
    predictions_path = os.path.join(out_dir, f"predictions_{run_name}.jsonl")
    eval_cmd = [sys.executable, "run_eval.py", "--model", model_path, "--split", split,
                "--out", predictions_path, "--seed", str(seed)]
    if limit:
        eval_cmd += ["--limit", str(limit)]
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"run_eval.py failed for {run_name}:\n{result.stderr}")

    detailed_path = os.path.join(out_dir, f"eval_{run_name}_detailed.jsonl")
    summary_path = os.path.join(out_dir, f"eval_{run_name}_summary.json")
    score_cmd = [sys.executable, "evaluator.py", "--score", "--predictions", predictions_path,
                 "--split", split, "--out-detailed", detailed_path, "--out-summary", summary_path]
    result = subprocess.run(score_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"evaluator.py --score failed for {run_name}:\n{result.stderr}")
    return summary_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints_dir", required=True,
                     help="the --output_dir a single sft_train.py run wrote checkpoint-N/ into")
    ap.add_argument("--base_model", required=True, help="needed to merge LoRA checkpoints")
    ap.add_argument("--mode", choices=["full", "lora"], required=True)
    ap.add_argument("--hf_repo_id", default=None,
                     help="if a checkpoint isn't found locally, try downloading it from "
                          "this repo (see PushCheckpointCallback in sft_train.py)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="outputs/checkpoint_eval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="cap problems per checkpoint for a quick pass")
    ap.add_argument("--steps_per_epoch", type=float, default=None,
                     help="if known, converts step numbers to epoch fractions in the "
                          "final report/chart labels (purely cosmetic - the underlying "
                          "comparison works either way, labeled by step if omitted)")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    checkpoints = find_all_checkpoints(args.checkpoints_dir)

    if not checkpoints and args.hf_repo_id:
        print(f"[run_multi_checkpoint_eval] no local checkpoints under {args.checkpoints_dir} - "
              f"nothing to enumerate locally. Pass explicit checkpoint labels via a future run, "
              f"or ensure at least one checkpoint-N directory exists locally to discover the set.")
    if not checkpoints:
        raise SystemExit(f"[run_multi_checkpoint_eval] no checkpoint-N directories found under "
                          f"{args.checkpoints_dir}")

    print(f"[run_multi_checkpoint_eval] found {len(checkpoints)} checkpoints: "
          f"{[s for s, _ in checkpoints]}")

    run_summaries = {}
    for step, ckpt_path in checkpoints:
        label = f"epoch_{step / args.steps_per_epoch:.2f}" if args.steps_per_epoch else f"step_{step}"
        print(f"\n[run_multi_checkpoint_eval] === {label} (checkpoint-{step}) ===")

        eval_model_path = ckpt_path
        if args.mode == "lora":
            merged_dir = os.path.join(tempfile.mkdtemp(), f"merged_{label}")
            print(f"[run_multi_checkpoint_eval] merging into throwaway dir {merged_dir}")
            merge_checkpoint(args.base_model, ckpt_path, merged_dir)
            eval_model_path = merged_dir

        summary_path = run_eval_and_score(eval_model_path, args.split, label, args.out_dir,
                                           args.seed, limit=args.limit)
        with open(summary_path) as f:
            run_summaries[label] = json.load(f)

        if args.mode == "lora":
            shutil.rmtree(os.path.dirname(eval_model_path), ignore_errors=True)

    combined_path = os.path.join(args.out_dir, "all_checkpoint_summaries.json")
    with open(combined_path, "w") as f:
        json.dump(run_summaries, f, indent=2)
    print(f"\n[run_multi_checkpoint_eval] wrote combined summaries -> {combined_path}")

    import evaluator as ev
    baseline = next(iter(run_summaries))  # earliest checkpoint as baseline by default
    ev.compare_runs(run_summaries, os.path.join(args.out_dir, "comparison"), baseline=baseline)
    print(f"[run_multi_checkpoint_eval] wrote ablation/comparison + charts -> "
          f"{os.path.join(args.out_dir, 'comparison')}")


if __name__ == "__main__":
    main()
