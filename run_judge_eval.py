"""
Optional stronger check for metric family #2 (intermediate reasoning correctness):
uses a live model as a judge to grade each reasoning step against the reference
solution, rather than relying only on evaluator.py's rule-based arithmetic-consistency
proxy (which only catches steps containing a clean, sympy-parseable "A = B" assertion).

Reuses run_augmentation.py's model backend (vLLM if available, HF-generate fallback
otherwise) so there's no separate serving stack to configure.

Usage:
  python run_judge_eval.py --model ckpts/qwen7b_stage1_merged \
      --detailed outputs/eval_run1_detailed.jsonl \
      --predictions outputs/predictions_run1.jsonl \
      --split train --out outputs/eval_run1_judge_scores.jsonl
"""

import argparse
import json

import evaluator as ev
import data_pipeline as dp
from run_augmentation import build_backend
from storage_utils import ensure_output_path, require_input_path, add_destination_args, dispatch_destination

JUDGE_PROMPT_WRAPPER = (
    "<|im_start|>system\nYou are a strict, careful grader of mathematical reasoning "
    "steps.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the judge model - can be the same "
                     "checkpoint you're evaluating, or a stronger separate model")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--split", default="train", help="must match the split evaluator.py "
                     "--score used, so problem_id lookups against skill labels line up")
    ap.add_argument("--skill-repo", default=dp.DEFAULT_SKILL_REPO)
    ap.add_argument("--skill-labels-file", default=None)
    ap.add_argument("--out", default="outputs/judge_scores.jsonl")
    ap.add_argument("--use_vllm", action="store_true", default=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    add_destination_args(ap, default_repo_type="dataset")
    args = ap.parse_args()

    predictions_path = str(require_input_path(args.predictions))
    labels = dp.load_skill_labels(args.split, repo_id=args.skill_repo, local_path=args.skill_labels_file)

    examples_with_steps = []
    with open(predictions_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ref = labels.get(row["problem_id"])
            if ref is None:
                continue  # no reference solution available - can't judge against nothing
            parsed = ev.parse_model_output(row["prediction"])
            step_texts = [t for _, t in parsed["steps"] if t]
            if not step_texts:
                continue
            examples_with_steps.append({
                "problem_id": row["problem_id"],
                "problem": ref["problem"],
                "reference_solution": ref.get("original_solution", ""),
                "steps": step_texts,
            })

    print(f"[run_judge_eval] {len(examples_with_steps)} examples with reference + steps to judge "
          f"({sum(len(e['steps']) for e in examples_with_steps)} total steps)")
    if not examples_with_steps:
        raise SystemExit("[run_judge_eval] nothing to judge - check --split matches your "
                          "predictions and that the skill labels cover these problems.")

    backend = build_backend(args.model, args.use_vllm, seed=args.seed)

    def batch_judge_fn(prompts):
        formatted = [JUDGE_PROMPT_WRAPPER.format(instruction=p) for p in prompts]
        results = backend.generate(formatted, n=1, temperature=0.0, max_tokens=16)
        return [r[0] for r in results]

    tallies = ev.judge_steps_batch(examples_with_steps, batch_judge_fn, batch_size=args.batch_size)

    ensure_output_path(args.out)
    with open(args.out, "w") as f:
        for ex, (n_total, n_correct) in zip(examples_with_steps, tallies):
            f.write(json.dumps({
                "problem_id": ex["problem_id"],
                "n_steps_judged": n_total,
                "n_judged_correct": n_correct,
                "judge_step_consistency": (n_correct / n_total) if n_total else None,
            }) + "\n")

    print(f"[run_judge_eval] wrote judge scores -> {args.out}")
    dispatch_destination(args.out, args)


if __name__ == "__main__":
    main()
