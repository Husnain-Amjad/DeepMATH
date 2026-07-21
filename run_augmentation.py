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
import os

import data_pipeline as dp
from storage_utils import add_destination_args, dispatch_destination
from templates import render_prompt_only, DEFAULT_SYSTEM_PROMPT


def vllm_available() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except Exception:
        return False


def resolve_model_path(model_path: str) -> str:
    """
    Guards against the confusing HFValidationError you get when a *local*
    path is mistyped/mis-resolved relative to cwd: transformers/vLLM first
    check os.path.isdir(), and if that's False (e.g. wrong relative path),
    they silently assume you meant a hub repo id and try to validate it as
    one - which then fails with an unrelated-looking error for any path
    with more than one "/" in it (e.g. "DeepMATH/sft_output/checkpoint-113").

    This raises a clear, specific error immediately instead of letting that
    happen, and auto-resolves valid local paths to absolute paths (some vLLM
    versions behave inconsistently with relative paths depending on cwd).
    """
    looks_like_local_path = "/" in model_path or model_path.startswith(".")
    is_local_dir = os.path.isdir(model_path)

    if is_local_dir:
        abspath = os.path.abspath(model_path)
        has_full_config = os.path.exists(os.path.join(model_path, "config.json"))
        has_adapter_only = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        if has_adapter_only and not has_full_config:
            raise SystemExit(
                f"[resolve_model_path] '{model_path}' contains adapter_config.json but "
                f"no config.json - this looks like a LoRA adapter checkpoint (saved by "
                f"--mode lora), not a standalone loadable model. vLLM/transformers can't "
                f"load an adapter directory directly.\n"
                f"Fix: run `python merge_lora.py --base_model <original base model, e.g. "
                f"Qwen/Qwen2.5-Math-7B> --adapter_path {model_path} --out {model_path}_merged` "
                f"first, then point --model at '{model_path}_merged' instead."
            )
        if not has_full_config and not has_adapter_only:
            raise SystemExit(
                f"[resolve_model_path] '{abspath}' is a directory but has neither "
                f"config.json nor adapter_config.json - it doesn't look like a model "
                f"checkpoint at all. Check the path."
            )
        return abspath

    if looks_like_local_path and model_path.count("/") >= 2:
        # 2+ slashes and not a real local dir - this is exactly the pattern that
        # produces the cryptic HFValidationError, so fail clearly here instead.
        raise SystemExit(
            f"[resolve_model_path] '{model_path}' doesn't exist as a local directory "
            f"relative to your current working directory ({os.getcwd()}), and it can't "
            f"be a Hugging Face Hub repo id either (hub ids allow only one '/', as in "
            f"'namespace/repo_name'). If this is meant to be a local checkpoint, check the "
            f"path - likely you need to drop a leading segment (e.g. 'sft_output/checkpoint-113' "
            f"instead of 'DeepMATH/sft_output/checkpoint-113') or pass an absolute path."
        )

    # single-segment or namespace/repo pattern -> assume intentional HF Hub id
    return model_path


class VLLMBackend:
    def __init__(self, model_path, max_model_len=4096, seed=42):
        from vllm import LLM
        from transformers import AutoTokenizer
        self.llm = LLM(model=model_path, max_model_len=max_model_len,
                        dtype="bfloat16", seed=seed)
        # Loaded separately (not via vllm's internal API) so this stays stable
        # across vLLM versions - this is what lets us render this model's own
        # chat template for prompts instead of a hardcoded one.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

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
        self.tokenizer.padding_side = "left"  # keeps generated continuations aligned
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            gen = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                max_new_tokens=max_tokens,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        new_tokens = gen[:, input_len:]
        texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        # gen rows are ordered prompt0-sample(0..n-1), prompt1-sample(0..n-1), ...
        return [texts[i * n:(i + 1) * n] for i in range(len(prompts))]


def build_backend(model_path, use_vllm=True, seed=42):
    model_path = resolve_model_path(model_path)
    if use_vllm and vllm_available():
        print(f"[backend] using vLLM for {model_path}")
        return VLLMBackend(model_path, seed=seed)
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
    ap.add_argument("--batch_size", type=int, default=64,
                     help="number of problems sent to the model per generate() call. "
                          "This is the actual throughput lever - raise it as far as your "
                          "VRAM allows before touching anything else.")
    ap.add_argument("--seed", type=int, default=42)
    add_destination_args(ap, default_repo_type="dataset")
    args = ap.parse_args()

    from determinism import set_all_seeds
    set_all_seeds(args.seed)

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
    backend = build_backend(args.model, args.use_vllm, seed=args.seed)

    if args.stage == "semantic":
        def batch_generate_fn(prompts):
            formatted = [render_prompt_only(backend.tokenizer, p, system_prompt="You are a helpful assistant.")
                         for p in prompts]
            results = backend.generate(formatted, n=1, temperature=0.8, max_tokens=512)
            return [r[0] for r in results]

        dp.generate_semantic_perturbations(
            weak_report, math_rows, batch_generate_fn, args.out,
            n_per_problem=args.n_per_problem, batch_size=args.batch_size,
        )
        dispatch_destination(args.out, args)

    elif args.stage == "numeric":
        def batch_solver_fn(problems, votes):
            formatted = [render_prompt_only(backend.tokenizer, p) for p in problems]
            return backend.generate(formatted, n=votes, temperature=0.7, max_tokens=1024)

        dp.generate_numeric_perturbations(
            weak_report, math_rows, batch_solver_fn, args.out,
            n_per_problem=args.n_per_problem, votes=args.votes,
            agreement_threshold=args.agreement_threshold, batch_size=args.batch_size,
        )
        dispatch_destination(args.out, args)

    elif args.stage == "multi_solution":
        def batch_sampler_fn(problems, n, temperature):
            formatted = [render_prompt_only(backend.tokenizer, p) for p in problems]
            return backend.generate(formatted, n=n, temperature=temperature, max_tokens=1024)

        dp.generate_multi_solutions(
            weak_report, math_rows, batch_sampler_fn, args.out,
            n_samples=args.n_samples, keep_top_k=args.keep_top_k, temperature=args.temperature,
            batch_size=args.batch_size,
        )
        dispatch_destination(args.out, args)


if __name__ == "__main__":
    main()
