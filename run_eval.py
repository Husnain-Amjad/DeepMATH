"""
Stage 2 (part 1): generate predictions from a trained checkpoint on the MATH
test split, producing predictions.jsonl in the shape data_pipeline.diagnose()
expects: {problem_id, subject, level, prediction, gold_boxed}.

Usage:
  python run_eval.py --model ckpts/qwen7b_stage1 --split test \
      --out outputs/predictions.jsonl
"""

import argparse
import json

import data_pipeline as dp
from run_augmentation import build_backend, MATH_PROMPT_TEMPLATE
from reward_fn import extract_boxed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="outputs/predictions.jsonl")
    ap.add_argument("--use_vllm", action="store_true", default=True)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None,
                     help="cap number of problems for a quick smoke test")
    args = ap.parse_args()

    math_rows = dp.load_hendrycks_math(args.split)
    if args.limit:
        math_rows = math_rows[:args.limit]
    print(f"[run_eval] evaluating on {len(math_rows)} problems ({args.split} split)")

    backend = build_backend(args.model, args.use_vllm)
    prompts = [MATH_PROMPT_TEMPLATE.format(problem=r["problem"]) for r in math_rows]

    # batch through vLLM in one call if available; HF backend loops internally
    completions = backend.generate(prompts, n=1, temperature=0.0, max_tokens=args.max_tokens)

    with open(args.out, "w") as f:
        for row, comp in zip(math_rows, completions):
            gold = extract_boxed(row["solution"])
            if gold is None:
                continue
            f.write(json.dumps({
                "problem_id": row["problem_id"],
                "subject": row["subject"],
                "level": row.get("level", "unknown"),
                "prediction": comp[0],
                "gold_boxed": gold,
            }) + "\n")

    print(f"[run_eval] wrote predictions -> {args.out}")
    print("Next: python data_pipeline.py --diagnose --predictions "
          f"{args.out} --weak-report outputs/weak_clusters.json")


if __name__ == "__main__":
    main()
