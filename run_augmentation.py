"""
Stage 3 driver: connects a live model to data_pipeline.py's augmentation
functions, which are deliberately model-agnostic (they take generator_fn /
solver_fn / sampler_fn callables rather than importing a serving stack
directly). This script supplies the actual model call, with a vLLM backend
(fast, preferred) and an HF-transformers fallback (portable, incl. ROCm).

Usage (run each stage separately, in order, after Stage 2's diagnose step):

  python run_augmentation.py --stage semantic \
      --model ckpts/qwen7b_stage1 --weak_report outputs/weak_clusters.json \
      --out outputs/semantic_aug.jsonl

  python run_augmentation.py --stage numeric \
      --model ckpts/qwen7b_stage1 --weak_report outputs/weak_clusters.json \
      --out outputs/numeric_aug.jsonl

  python run_augmentation.py --stage multi_solution \
      --model ckpts/qwen7b_stage1 --weak_report outputs/weak_clusters.json \
      --out outputs/multi_solution.jsonl

If --weak_report doesn't exist yet, run the diagnose step first (see USAGE.md).
"""

import argparse
import json

import data_pipeline as dp

MATH_PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a careful mathematical problem solver. "
    "First think through the problem's structure, then solve it.<|im_end|>\n"
    "<|im_start|>user\n{problem}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
GENERIC_PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def vllm_available() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except Exception:
        return False


class VLLMBackend:
    def __init__(self, model_path, max_model_len=4096):
        from vllm import LLM
        self.llm = LLM(model=model_path, max_model_len=max_model_len, dtype="bfloat16")

    def generate(self, prompts, n=1, temperature=0.7, max_tokens=1024):
        from vllm import SamplingParams
        params = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens)
        results = self.llm.generate(prompts, params)
        return [[o.text for o in r.outputs] for r in results]


class HFBackend:
    """Portable fallback (works on ROCm too). Slower than vLLM but no extra install."""
    def __init__(self, model_path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map="auto",
        )
        self.torch = torch

    def generate(self, prompts, n=1, temperature=0.7, max_tokens=1024):
        out = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(
                    **inputs,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    max_new_tokens=max_tokens,
                    num_return_sequences=n,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new_tokens = gen[:, inputs["input_ids"].shape[1]:]
            texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            out.append(texts)
        return out


def build_backend(model_path, use_vllm=True):
    if use_vllm and vllm_available():
        print(f"[backend] using vLLM for {model_path}")
        return VLLMBackend(model_path)
    if use_vllm:
        print("[backend] vLLM requested but not importable (common on ROCm without the "
              "ROCm-specific build) - falling back to HF-generate. Slower, still correct.")
    print(f"[backend] using HF-generate for {model_path}")
    return HFBackend(model_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["semantic", "numeric", "multi_solution"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--weak_report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--use_vllm", action="store_true", default=True)
    ap.add_argument("--n_per_problem", type=int, default=2)
    ap.add_argument("--votes", type=int, default=5)
    ap.add_argument("--agreement_threshold", type=float, default=0.8)
    ap.add_argument("--n_samples", type=int, default=16)
    ap.add_argument("--keep_top_k", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    with open(args.weak_report) as f:
        weak_report = json.load(f)
    if not weak_report:
        raise SystemExit(
            "[run_augmentation] weak_report is empty - nothing to augment. "
            "Re-check Stage 2's diagnose thresholds (accuracy < 0.40, n >= 10) "
            "or confirm predictions.jsonl actually covers the clusters you expect."
        )
    print(f"[run_augmentation] {len(weak_report)} weak clusters loaded from {args.weak_report}")

    math_rows = dp.load_hendrycks_math(args.split)
    backend = build_backend(args.model, args.use_vllm)

    if args.stage == "semantic":
        def generator_fn(prompt: str) -> str:
            prompt_fmt = GENERIC_PROMPT_TEMPLATE.format(instruction=prompt)
            return backend.generate([prompt_fmt], n=1, temperature=0.8, max_tokens=512)[0][0]

        dp.generate_semantic_perturbations(
            weak_report, math_rows, generator_fn, args.out, n_per_problem=args.n_per_problem,
        )

    elif args.stage == "numeric":
        def solver_fn(problem: str) -> str:
            prompt_fmt = MATH_PROMPT_TEMPLATE.format(problem=problem)
            return backend.generate([prompt_fmt], n=1, temperature=0.7, max_tokens=1024)[0][0]

        dp.generate_numeric_perturbations(
            weak_report, math_rows, solver_fn, args.out,
            n_per_problem=args.n_per_problem, votes=args.votes,
            agreement_threshold=args.agreement_threshold,
        )

    elif args.stage == "multi_solution":
        def sampler_fn(problem: str, n: int, temperature: float):
            prompt_fmt = MATH_PROMPT_TEMPLATE.format(problem=problem)
            return backend.generate([prompt_fmt], n=n, temperature=temperature, max_tokens=1024)[0]

        dp.generate_multi_solutions(
            weak_report, math_rows, sampler_fn, args.out,
            n_samples=args.n_samples, keep_top_k=args.keep_top_k, temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
