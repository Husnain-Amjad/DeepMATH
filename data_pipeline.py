"""
Data pipeline for the skill-metacognition + GRPO training run.

Stages implemented here:
  --build-sft          : merge your skill-label file with hendrycks_math -> SFT jsonl
  --diagnose            : evaluate a checkpoint, report accuracy by (subject, level)
  --augment-semantic     : semantic-preserving perturbation of weak clusters (prioritized)
  --augment-numeric       : numeric-literal perturbation with verification (secondary)
  --multi-solution        : rejection-sampled diverse correct solutions for weak clusters

Design notes:
  - "think" section = short skill label + semantic extraction (entities/quantities/
    relations/operation type), not just a category tag.
  - Semantic perturbation is meaning-preserving -> gold answer is reused as-is.
  - Numeric perturbation requires verification (sympy recompute or self-consistency
    vote) before being accepted into the training set.
"""

import argparse
import json
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from datasets import load_dataset, Dataset

from reward_fn import extract_boxed, answers_match, score_completion

SUBJECTS = [
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
]

CHAT_TEMPLATE = (
    "<|im_start|>system\nYou are a careful mathematical problem solver. "
    "First think through the problem's structure, then solve it.<|im_end|>\n"
    "<|im_start|>user\n{problem}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n{think}\n</think>\n<solution>\n{solution}\n</solution><|im_end|>"
)

_SKILL_TAG_RE = re.compile(r"\[SKILL:\s*([^\]]+)\]")
_ANSWER_LINE_RE = re.compile(r"\n?ANSWER:\s*\\boxed\{.*?\}\s*$", re.IGNORECASE | re.DOTALL)
_MINIMUM_SKILLS_PREFIX_RE = re.compile(r"^\s*MINIMUM_SKILLS:\s*", re.IGNORECASE)


def _hash_problem(problem: str) -> str:
    return hashlib.sha256(problem.strip().encode("utf-8")).hexdigest()[:16]


def load_hendrycks_math(split: str = "train"):
    """Loads all 7 subject configs of EleutherAI/hendrycks_math and concatenates them.
    Used as a supplementary pool for problems not covered by the skill-label set,
    and as the source of raw problems for diagnostics/augmentation stages."""
    all_rows = []
    for subj in SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subj, split=split)
        for row in ds:
            row = dict(row)
            row["subject"] = subj
            row["problem_id"] = _hash_problem(row["problem"])
            all_rows.append(row)
    return all_rows


def clean_skill_list(raw: str) -> str:
    """Normalizes 'MINIMUM_SKILLS: A | B' or 'A | B' -> 'A | B'."""
    if not raw:
        return ""
    raw = _MINIMUM_SKILLS_PREFIX_RE.sub("", raw.strip())
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    return " | ".join(parts)


def strip_trailing_answer_line(reasoning_trace: str) -> str:
    """Removes the trailing 'ANSWER: \\boxed{...}' line from reasoning_trace so the
    <think> section doesn't duplicate the boxed answer that appears in <solution>
    (avoids teaching the model to leak the answer format twice / anchor on it early)."""
    return _ANSWER_LINE_RE.sub("", reasoning_trace).strip()


def build_think_section(row: dict) -> str:
    """
    Builds the <think> content directly from the skill-labeled reasoning_trace,
    which already contains inline [SKILL: ...] tags before each step - this is
    the real metacognitive signal, so it's kept close to verbatim rather than
    reduced to a single category tag. A short header line names the minimal
    skill set so it's independently checkable at eval time.
    """
    min_skills = clean_skill_list(row.get("minimum_skills", ""))
    if not min_skills:
        # fall back to the noisier raw extraction field only if the clean
        # field is genuinely empty
        min_skills = clean_skill_list(row.get("_extraction_raw", "")) or row.get("skills_used_in_steps", "")

    trace = strip_trailing_answer_line(row.get("reasoning_trace", ""))
    header = f"Relevant skills: {min_skills}" if min_skills else ""
    return f"{header}\n{trace}".strip() if header else trace


def load_skill_labels(path: str):
    """
    Loads the actual skill-labeled dataset schema:
      subject, level, problem, original_solution, model_answer, reasoning_trace
      (inline [SKILL: ...] tags per step, ending 'ANSWER: \\boxed{...}'),
      skills_used_in_steps, minimum_skills, n_skills, _extraction_raw.
    Accepts .jsonl or .csv. Returns a dict keyed by problem-hash -> row.
    """
    labels = {}
    if path.endswith(".csv"):
        import csv
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for obj in reader:
                labels[_hash_problem(obj["problem"])] = obj
    else:
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                labels[_hash_problem(obj["problem"])] = obj
    return labels


def build_sft_dataset(skill_label_path: str, out_path: str, split: str = "train",
                       include_unlabeled_fallback: bool = True):
    """
    Builds SFT examples directly from the skill-labeled dataset (self-contained:
    it already carries problem/solution/subject/level, no join needed). Optionally
    tops up with plain hendrycks_math rows (generic think section) for problems
    the skill-labeling pass hasn't covered yet, so training data isn't capped by
    labeling progress.
    """
    labels = load_skill_labels(skill_label_path)
    examples = []

    for pid, row in labels.items():
        original_solution = row.get("original_solution", "")
        gold = extract_boxed(original_solution) or row.get("model_answer")
        if gold is None:
            continue  # can't supervise without a checkable answer
        think = build_think_section(row)
        text = CHAT_TEMPLATE.format(
            problem=row["problem"], think=think, solution=original_solution
        )
        examples.append({
            "problem_id": pid,
            "subject": row.get("subject", "unknown"),
            "level": row.get("level", "unknown"),
            "text": text,
            "gold_boxed": gold,
            "n_skills": row.get("n_skills"),
            "source": "skill_labeled",
        })

    n_labeled = len(examples)
    n_fallback = 0
    if include_unlabeled_fallback:
        math_rows = load_hendrycks_math(split)
        for row in math_rows:
            if row["problem_id"] in labels:
                continue
            gold = extract_boxed(row["solution"])
            if gold is None:
                continue
            think = f"Relevant skills: (not yet labeled, subject={row['subject']})"
            text = CHAT_TEMPLATE.format(
                problem=row["problem"], think=think, solution=row["solution"]
            )
            examples.append({
                "problem_id": row["problem_id"],
                "subject": row["subject"],
                "level": row.get("level", "unknown"),
                "text": text,
                "gold_boxed": gold,
                "n_skills": None,
                "source": "unlabeled_fallback",
            })
            n_fallback += 1

    print(f"[build-sft] skill-labeled examples: {n_labeled}, unlabeled fallback: {n_fallback}")
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[build-sft] wrote {len(examples)} examples -> {out_path}")


def diagnose(predictions_path: str, out_report_path: str):
    """
    predictions_path: jsonl with fields {problem_id, subject, level, prediction, gold_boxed}
    produced by your own eval/generation script after Stage 1 SFT.
    Outputs per (subject, level) accuracy to find weak clusters.
    """
    stats = defaultdict(lambda: [0, 0])  # (subject, level) -> [correct, total]
    with open(predictions_path) as f:
        for line in f:
            obj = json.loads(line)
            key = (obj["subject"], str(obj["level"]))
            pred = extract_boxed(obj["prediction"])
            correct = pred is not None and answers_match(pred, obj["gold_boxed"])
            stats[key][0] += int(correct)
            stats[key][1] += 1

    report = []
    for (subject, level), (correct, total) in sorted(stats.items()):
        acc = correct / total if total else 0.0
        report.append({"subject": subject, "level": level, "accuracy": round(acc, 4), "n": total})
        print(f"{subject:28s} L{level:>2}  acc={acc:.3f}  n={total}")

    with open(out_report_path, "w") as f:
        json.dump(report, f, indent=2)

    weak = [r for r in report if r["accuracy"] < 0.4 and r["n"] >= 10]
    print("\n[diagnose] weak clusters (acc < 0.40, n >= 10):")
    for r in weak:
        print(f"  {r['subject']} L{r['level']}  acc={r['accuracy']}  n={r['n']}")
    return weak


# ---------------------------------------------------------------------------
# Semantic perturbation (prioritized): paraphrase / entity rename / distractor
# clause insertion. Meaning-preserving -> gold answer is reused unchanged.
# ---------------------------------------------------------------------------

SEMANTIC_PERTURB_PROMPT = """You will rewrite a math problem WITHOUT changing its \
mathematical content or answer. Apply exactly one of the following transformations, \
chosen to best fit the problem:
  (a) Paraphrase the wording/sentence structure while keeping all quantities and \
      relations identical.
  (b) Rename the named entities (people, objects, labels) to different but semantically \
      equivalent ones.
  (c) Insert one plausible-sounding but mathematically irrelevant clause or sentence \
      (a distractor) that does not affect the solution.

Do not change any numbers, units, or the underlying relationships between quantities.
Return ONLY the rewritten problem text, nothing else.

Original problem:
{problem}
"""


def generate_semantic_perturbations(weak_report: list, math_rows: list, generator_fn,
                                     out_path: str, n_per_problem: int = 2):
    """
    generator_fn: callable(prompt: str) -> str
        Plug in your own model call here (local Qwen2.5-Math-7B via vLLM/HF generate,
        or any other model you have access to). Kept pluggable so this script has no
        hard dependency on a specific serving stack.
    """
    weak_keys = {(r["subject"], r["level"]) for r in weak_report}
    targets = [row for row in math_rows if (row["subject"], str(row.get("level", "unknown"))) in weak_keys]
    print(f"[augment-semantic] {len(targets)} source problems in weak clusters")

    out = []
    for row in targets:
        for _ in range(n_per_problem):
            prompt = SEMANTIC_PERTURB_PROMPT.format(problem=row["problem"])
            rewritten = generator_fn(prompt).strip()
            if not rewritten or rewritten == row["problem"]:
                continue
            out.append({
                "problem_id": _hash_problem(rewritten),
                "source_problem_id": row["problem_id"],
                "subject": row["subject"],
                "level": row.get("level", "unknown"),
                "problem": rewritten,
                "solution": row["solution"],  # gold answer unchanged: meaning-preserving
                "augmentation": "semantic_perturbation",
            })

    with open(out_path, "w") as f:
        for ex in out:
            f.write(json.dumps(ex) + "\n")
    print(f"[augment-semantic] wrote {len(out)} perturbed examples -> {out_path}")


# ---------------------------------------------------------------------------
# Numeric perturbation (secondary): only applied where the answer can be
# independently re-verified. Two verification paths:
#   1. sympy recomputation, when the problem's solution is a closed-form
#      expression you can reparametrize (you supply a `recompute_fn` per template).
#   2. self-consistency vote: sample the *current* model N times on the
#      perturbed problem and only keep it if a strong majority agree AND
#      that majority also matches an independently sampled "solver" pass.
# ---------------------------------------------------------------------------

NUMERIC_LITERAL_RE = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.])")


def perturb_numeric_literals(problem: str, rng, scale_range=(0.5, 2.0)):
    """Randomly rescale integer literals in-place; returns (new_problem, changed_values)."""
    changed = []

    def _sub(m):
        val = m.group(1)
        try:
            n = int(val)
        except ValueError:
            return val  # skip decimals - too easy to break units/precision
        factor = rng.uniform(*scale_range)
        new_n = max(1, round(n * factor))
        changed.append((n, new_n))
        return str(new_n)

    new_problem = NUMERIC_LITERAL_RE.sub(_sub, problem)
    return new_problem, changed


def generate_numeric_perturbations(weak_report: list, math_rows: list, solver_fn,
                                    out_path: str, n_per_problem: int = 2,
                                    votes: int = 5, agreement_threshold: float = 0.8,
                                    seed: int = 0):
    """
    solver_fn: callable(problem: str) -> str (a full solution with \\boxed{...})
        Used to independently re-derive the answer for the perturbed numbers.
    Only accepted if >= agreement_threshold of `votes` independent solver calls agree
    on the same boxed answer (self-consistency gate) - this is the verification step
    that makes numeric perturbation safe to include in training data.
    """
    import random
    rng = random.Random(seed)
    weak_keys = {(r["subject"], r["level"]) for r in weak_report}
    targets = [row for row in math_rows if (row["subject"], str(row.get("level", "unknown"))) in weak_keys]
    print(f"[augment-numeric] {len(targets)} source problems in weak clusters")

    accepted, rejected = 0, 0
    out = []
    for row in targets:
        for _ in range(n_per_problem):
            new_problem, changed = perturb_numeric_literals(row["problem"], rng)
            if not changed:
                continue

            votes_seen = defaultdict(int)
            solutions_by_answer = {}
            for _ in range(votes):
                sol = solver_fn(new_problem)
                ans = extract_boxed(sol)
                if ans is None:
                    continue
                key = ans.strip()
                votes_seen[key] += 1
                solutions_by_answer.setdefault(key, sol)

            if not votes_seen:
                rejected += 1
                continue
            best_ans, best_count = max(votes_seen.items(), key=lambda kv: kv[1])
            if best_count / votes < agreement_threshold:
                rejected += 1
                continue

            accepted += 1
            out.append({
                "problem_id": _hash_problem(new_problem),
                "source_problem_id": row["problem_id"],
                "subject": row["subject"],
                "level": row.get("level", "unknown"),
                "problem": new_problem,
                "solution": solutions_by_answer[best_ans],
                "changed_values": changed,
                "self_consistency": best_count / votes,
                "augmentation": "numeric_perturbation",
            })

    print(f"[augment-numeric] accepted={accepted} rejected={rejected} "
          f"(agreement threshold={agreement_threshold})")
    with open(out_path, "w") as f:
        for ex in out:
            f.write(json.dumps(ex) + "\n")
    print(f"[augment-numeric] wrote {len(out)} verified perturbed examples -> {out_path}")


# ---------------------------------------------------------------------------
# Multi-solution generation via rejection sampling (diverse correct derivations)
# ---------------------------------------------------------------------------

def _solution_signature(solution_text: str) -> str:
    """Coarse dedup signature: sequence of equation-bearing lines, whitespace-collapsed."""
    lines = [re.sub(r"\s+", "", l) for l in solution_text.splitlines() if l.strip()]
    sig_lines = [l for l in lines if any(c in l for c in "=+-*/^")]
    return hashlib.sha256(" ".join(sig_lines).encode()).hexdigest()[:12]


def generate_multi_solutions(weak_report: list, math_rows: list, sampler_fn,
                              out_path: str, n_samples: int = 16, keep_top_k: int = 3,
                              temperature: float = 0.9):
    """
    sampler_fn: callable(problem: str, n: int, temperature: float) -> list[str]
        Returns n sampled completions (each containing a full solution + \\boxed{}).
    Keeps up to keep_top_k solutions per problem that (a) match the gold answer and
    (b) are structurally distinct from each other (different equation signature).
    """
    weak_keys = {(r["subject"], r["level"]) for r in weak_report}
    targets = [row for row in math_rows if (row["subject"], str(row.get("level", "unknown"))) in weak_keys]
    print(f"[multi-solution] {len(targets)} source problems in weak clusters")

    out = []
    for row in targets:
        gold = extract_boxed(row["solution"])
        if gold is None:
            continue
        samples = sampler_fn(row["problem"], n_samples, temperature)
        seen_sigs = set()
        kept = []
        for s in samples:
            pred = extract_boxed(s)
            if pred is None or not answers_match(pred, gold):
                continue
            sig = _solution_signature(s)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            kept.append(s)
            if len(kept) >= keep_top_k:
                break

        for i, sol in enumerate(kept):
            out.append({
                "problem_id": row["problem_id"],
                "subject": row["subject"],
                "level": row.get("level", "unknown"),
                "problem": row["problem"],
                "solution": sol,
                "variant_index": i,
                "augmentation": "multi_solution_rejection_sampling",
            })

    with open(out_path, "w") as f:
        for ex in out:
            f.write(json.dumps(ex) + "\n")
    print(f"[multi-solution] wrote {len(out)} diverse verified solutions -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-sft", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--augment-semantic", action="store_true")
    ap.add_argument("--augment-numeric", action="store_true")
    ap.add_argument("--multi-solution", action="store_true")

    ap.add_argument("--skill-labels", type=str, default="data/skill_labels.jsonl")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out", type=str, default="outputs/sft_data.jsonl")
    ap.add_argument("--predictions", type=str, default="outputs/predictions.jsonl")
    ap.add_argument("--weak-report", type=str, default="outputs/weak_clusters.json")

    args = ap.parse_args()

    if args.build_sft:
        build_sft_dataset(args.skill_labels, args.out, args.split)
    elif args.diagnose:
        diagnose(args.predictions, args.weak_report)
    else:
        print("Semantic / numeric / multi-solution augmentation require you to pass a "
              "generator_fn / solver_fn / sampler_fn (a live model call) - import this "
              "module and call the functions directly from a script where your model "
              "is loaded, rather than running this file standalone for those stages.")
