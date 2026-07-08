# Qwen2.5-Math Skill-Metacognition + GRPO Pipeline

Target hardware: **1x A100 80GB**. Models: `Qwen/Qwen2.5-Math-1.5B` (full fine-tune sandbox)
and `Qwen/Qwen2.5-Math-7B` (LoRA + GRPO sandbox). Data: `EleutherAI/hendrycks_math`.

## Stage sequence

```
Stage 0  data_pipeline.py --build-sft         -> builds <think>/<solution> SFT set from
                                                  your skill-label file + hendrycks_math
Stage 1  sft_train.py                         -> cold-start SFT (1.5B full-FT / 7B LoRA)
Stage 2  data_pipeline.py --diagnose          -> per (subject, level) accuracy report
                                                  -> identifies weak clusters (e.g. geometry L5)
Stage 3  data_pipeline.py --augment-semantic  -> semantic-preserving perturbations of weak
         data_pipeline.py --augment-numeric      clusters (paraphrase / distractor clause /
                                                  numeric-literal substitution w/ verification)
         data_pipeline.py --multi-solution     -> rejection-sampled diverse correct solutions
Stage 4  sft_train.py --round2                -> replay-mixed SFT: original + augmented data
Stage 5  grpo_train.py                        -> pure rule-based GRPO (boxed-answer reward),
                                                  7B + LoRA, no PRM in the loop
```

## Why this shape (recap of the design decisions)

- **Stage 0 "think" section is a semantic-extraction step, not a bare skill tag.** It names
  entities/quantities/relations and the operation type. The skill label is kept as a short,
  separately-checkable field so Stage 2 diagnostics can correlate skill-classification
  accuracy with solve accuracy, not just measure end-to-end correctness.
- **Perturbation is diagnostic-first, generative-second.** We only mass-produce perturbed
  training examples for clusters where the diagnostic pass shows brittleness — mirrors the
  GSM-Symbolic finding that models fail on structurally-identical variants, so we target
  training data at exactly that failure mode instead of perturbing uniformly.
- **Semantic perturbation is prioritized** (paraphrase, entity renaming, plausible-but-irrelevant
  distractor clauses) since it doesn't require recomputing ground truth — meaning is preserved,
  so the boxed answer is untouched. **Numeric perturbation is secondary** and gated by
  verification (sympy recomputation or self-consistency vote) because changing numbers can
  silently change the answer.
- **RL stage is pure rule-based GRPO.** No PRM resident during RL — this keeps you to two
  models in memory (policy + reference) on a single 80GB card. ReasonFlux-PRM-style scoring
  is used only offline, in Stage 3's multi-solution filtering, never as a live reward.

## Environment

See `requirements.txt` for the split CUDA vs ROCm install commands. tl;dr:
- Common packages (transformers/trl/peft/datasets/sympy) install identically on both.
- `torch` is installed from a backend-specific index URL (cu121 vs rocm6.2 wheels).
- `vllm` on ROCm is not a plain `pip install` - needs AMD's ROCm build. Both `sft_train.py`'s
  rollout-free SFT and `grpo_train.py`'s RL rollouts work without vLLM (HF-generate fallback),
  just slower - the scripts detect this automatically and warn rather than fail.
- `bitsandbytes` (4-bit QLoRA, `--use_bnb`) is CUDA-only; it's auto-disabled on ROCm.

## ROCm / speed notes

- Both training scripts use `attn_implementation="sdpa"` (PyTorch's native
  scaled-dot-product-attention) instead of the `flash_attention_2` pip package, since the
  latter is CUDA-only. On ROCm (MI200/MI300-class cards), SDPA dispatches to the AOTriton
  backend and gets most of flash-attention's speed without a ROCm-specific flash-attn build.
- `sft_train.py --packing` (on by default) packs multiple examples per training sequence
  with EOS-bounded loss masking via TRL - cuts padding waste substantially on the
  variable-length MATH solutions, which is the main free speed win available on either backend.
- `--torch_compile` is available on both scripts for extra throughput on longer runs; off by
  default since compile overhead isn't worth it for short smoke-test runs.
- `torch.backends.cuda.matmul.allow_tf32 = True` is set unconditionally - it's a no-op on
  ROCm and a free speedup on CUDA (Ampere+).
- ROCm PyTorch aliases the `torch.cuda.*` API, so `device_map="auto"`, `.to("cuda")`, bf16,
  etc. all work unchanged - the only real portability gaps are the three items above
  (flash-attn, vLLM, bitsandbytes), which are all handled with explicit fallbacks in the scripts.

## Skill-label dataset schema

`data_pipeline.py` expects your skill-labeled dataset (jsonl or csv) with these columns:

| column | contents |
|---|---|
| `subject`, `level` | e.g. `"algebra"`, `"Level 2"` (matches hendrycks_math's own level strings) |
| `problem` | problem text |
| `original_solution` | the textbook solution, contains `\boxed{...}` |
| `model_answer` | extracted final answer, used as a fallback gold if `original_solution` has no boxed answer |
| `reasoning_trace` | step-by-step trace with inline `[SKILL: <name>]` tags before each step, ending `ANSWER: \boxed{...}` |
| `skills_used_in_steps` | pipe-separated skill list, e.g. `"A \| B"` |
| `minimum_skills` | pipe-separated minimal skill set, possibly prefixed `"MINIMUM_SKILLS: ..."` |
| `n_skills`, `_extraction_raw` | metadata / noisy fallback extraction, used only if `minimum_skills` is empty |

`build_sft_dataset` builds the `<think>` section directly from `reasoning_trace` (kept close
to verbatim, with the trailing `ANSWER: \boxed{...}` line stripped so it isn't duplicated
against `<solution>`) prefixed by a short `Relevant skills: ...` header parsed from
`minimum_skills`. `<solution>` is `original_solution` verbatim. This is a self-contained
build - it does not require joining against hendrycks_math by hash. If
`include_unlabeled_fallback=True` (default), hendrycks_math problems not present in your
skill-label file are still added with a minimal placeholder think section, so training data
isn't capped by labeling coverage.

## Rough VRAM budget (80GB card)

| Stage | Model | Method | Approx peak VRAM |
|---|---|---|---|
| SFT | 1.5B | full FT, bf16, AdamW | ~25-35GB (batch/seq-len dependent) |
| SFT | 7B | LoRA (r=16-64), bf16 base frozen | ~35-50GB |
| GRPO | 7B | LoRA policy + frozen reference, bf16, vLLM rollouts | ~60-75GB (tune group size / max_new_tokens to fit) |

If GRPO doesn't fit at your desired group size, first reduce `num_generations` (group size)
before reducing model precision — group size affects advantage-estimate quality, not
correctness of the reward.
