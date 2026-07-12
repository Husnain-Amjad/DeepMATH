"""
Data pipeline for the skill-metacognition + GRPO training run.

Stages implemented here:
  --build-sft          : build SFT jsonl from the skill-labeled dataset (+ hendrycks_math fallback)
  --diagnose            : evaluate a checkpoint, report accuracy by (subject, level)
  --augment-semantic     : semantic-preserving perturbation of weak clusters (prioritized)
  --augment-numeric       : numeric-literal perturbation with verification (secondary)
  --multi-solution        : rejection-sampled diverse correct solutions for weak clusters

Design notes:
  - "think" section = short skill label + semantic extraction (entities/quantities/
    relations/operation type), not just a category tag - built from the skill-labeled
    dataset's own reasoning_trace, which already carries inline [SKILL: ...] tags.
  - Semantic perturbation is meaning-preserving -> gold answer is reused as-is.
  - Numeric perturbation requires verification (self-consistency vote) before being
    accepted into the training set.
  - All augmentation functions batch model calls across problems rather than looping
    one problem at a time - see run_augmentation.py for the model-connected drivers.
  - Every write path auto-creates its parent directory; every read path fails with a
    clear message if missing, rather than silently creating an empty directory next
    to a file that was never there.
"""

import argparse
import json
import hashlib
import re
from collections import defaultdict

from datasets import load_dataset

from reward_fn import extract_boxed, answers_match
from storage_utils import ensure_output_path, require_input_path, add_destination_args, dispatch_destination

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

DEFAULT_SKILL_REPO = "HusnainAmjad/Skill_MATH"


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
    <think> section doesn't duplicate the boxed answer that appears in <solution>."""
    return _ANSWER_LINE_RE.sub("", reasoning_trace).strip()


def build_think_section(row: dict) -> str:
    """
    Builds the <think> content directly from the skill-labeled reasoning_trace,
    which already contains inline [SKILL: ...] tags before each step - kept close
    to verbatim rather than reduced to a single category tag. A short header line
    names the minimal skill set so it's independently checkable at eval time.
    """
    min_skills = clean_skill_list(row.get("minimum_skills", ""))
    if not min_skills:
        min_skills = clean_skill_list(row.get("_extraction_raw", "")) or row.get("skills_used_in_steps", "")

    trace = strip_trailing_answer_line(row.get("reasoning_trace", ""))
    header = f"Relevant skills: {min_skills}" if min_skills else ""
    return f"{header}\n{trace}".strip() if header else trace


def load_skill_labels(split: str = "train", repo_id: str = DEFAULT_SKILL_REPO,
                       local_path: str = None):
    """
    Loads the skill-labeled dataset either from a local jsonl/csv file (if
    local_path is given - useful for offline work or a custom labeling batch),
    or from a Hugging Face Hub dataset repo otherwise (default: HusnainAmjad/Skill_MATH).
    Returns a dict keyed by a hash of the problem text.
    """
    labels = {}
    if local_path:
        local_path = str(require_input_path(local_path))
        if local_path.endswith(".csv"):
            import csv
            with open(local_path, newline="") as f:
                reader = csv.DictReader(f)
                for obj in reader:
                    labels[_hash_problem(obj["problem"])] = obj
        else:
            with open(local_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    labels[_hash_problem(obj["problem"])] = obj
        print(f"[load_skill_labels] loaded {len(labels)} rows from local file {local_path}")
        return labels

    ds = load_dataset(repo_id, split=split)
    for row in ds:
        row = dict(row)
        labels[_hash_problem(row["problem"])] = row
    print(f"[load_skill_labels] loaded {len(labels)} rows from hf://{repo_id} (split={split})")
    return labels


def build_sft_dataset(out_path: str, split: str = "train",
                       include_unlabeled_fallback: bool = True,
                       skill_repo: str = DEFAULT_SKILL_REPO, skill_labels_file: str = None):
    """
    Builds SFT examples directly from the skill-labeled dataset (self-contained:
    it already carries problem/solution/subject/level, no join needed). Optionally
    tops up with plain hendrycks_math rows (generic think section) for problems
    the skill-labeling pass hasn't covered yet, so training data isn't capped by
    labeling progress.
    """
    labels = load_skill_labels(split, repo_id=skill_repo, local_path=skill_labels_file)
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

    ensure_output_path(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[build-sft] wrote {len(examples)} examples -> {out_path}")


def diagnose(predictions_path: str, out_report_path: str):
    """
    predictions_path: jsonl with fields {problem_id, subject, level, prediction, gold_boxed}
    produced by run_eval.py after a training stage. Outputs per (subject, level)
    accuracy to find weak clusters.
    """
    stats = defaultdict(lambda: [0, 0])  # (subject, level) -> [correct, total]

    # READ path: fail clearly if missing, rather than creating a directory next
    # to a file that was never produced (that was the actual bug - a write-path
    # helper doesn't fix a missing read-path file).
    predictions_path = str(require_input_path(predictions_path))

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

    ensure_output_path(out_report_path)
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


def generate_semantic_perturbations(weak_report: list, math_rows: list, batch_generate_fn,
                                     out_path: str, n_per_problem: int = 2, batch_size: int = 128):
    """
    batch_generate_fn: callable(prompts: list[str]) -> list[str]
        One completion per prompt, called on a BATCH of prompts at once - this is what
        actually gives you vLLM's continuous-batching throughput. Looping a single-prompt
        generate() call once per problem pays engine scheduling overhead thousands of
        times over and is dramatically slower than a handful of large batched calls.
    """
    weak_keys = {(r["subject"], r["level"]) for r in weak_report}
    targets = [row for row in math_rows if (row["subject"], str(row.get("level", "unknown"))) in weak_keys]
    print(f"[augment-semantic] {len(targets)} source problems in weak clusters")

    jobs = []  # (row, prompt_text)
    for row in targets:
        for _ in range(n_per_problem):
            jobs.append((row, SEMANTIC_PERTURB_PROMPT.format(problem=row["problem"])))

    print(f"[augment-semantic] generating {len(jobs)} perturbations in batches of {batch_size} "
          f"({(len(jobs) + batch_size - 1) // batch_size} model calls total, not {len(jobs)})")
    completions = []
    for i in range(0, len(jobs), batch_size):
        batch_prompts = [p for _, p in jobs[i:i + batch_size]]
        completions.extend(batch_generate_fn(batch_prompts))
        print(f"[augment-semantic] {min(i + batch_size, len(jobs))}/{len(jobs)} generated")

    out = []
    for (row, _), rewritten in zip(jobs, completions):
        rewritten = (rewritten or "").strip()
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

    ensure_output_path(out_path)
    with open(out_path, "w") as f:
        for ex in out:
            f.write(json.dumps(ex) + "\n")
    print(f"[augment-semantic] wrote {len(out)} perturbed examples -> {out_path}")


# ---------------------------------------------------------------------------
# Numeric perturbation (secondary): only applied where the answer can be
# independently re-verified via a self-consistency vote.
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


def generate_numeric_perturbations(weak_report: list, math_rows: list, batch_solver_fn,
                                    out_path: str, n_per_problem: int = 2,
                                    votes: int = 5, agreement_threshold: float = 0.8,
                                    seed: int = 0, batch_size: int = 64):
    """
    batch_solver_fn: callable(problems: list[str], votes: int) -> list[list[str]]
        Returns `votes` independently-sampled full solutions (each with \\boxed{...})
        PER PROBLEM, for a BATCH of problems at once - implemented via vLLM's n=votes
        sampling parameter under the hood, so every vote for every problem comes back
        in one engine call instead of `votes * n_problems` separate single-prompt calls.
    Only accepted if >= agreement_threshold of `votes` independent solver calls agree
    on the same boxed answer (self-consistency gate) - this is the verification step
    that makes numeric perturbation safe to include in training data.
    """
    import random
    rng = random.Random(seed)
    weak_keys = {(r["subject"], r["level"]) for r in weak_report}
    targets = [row for row in math_rows if (row["subject"], str(row.get("level", "unknown"))) in weak_keys]
    print(f"[augment-numeric] {len(targets)} source problems in weak clusters")

    jobs = []  # (row, new_problem, changed)
    for row in targets:
        for _ in range(n_per_problem):
            new_problem, changed = perturb_numeric_literals(row["problem"], rng)
            if changed:
                jobs.append((row, new_problem, changed))

    print(f"[augment-numeric] verifying {len(jobs)} perturbed problems in batches of "
          f"{batch_size} ({(len(jobs) + batch_size - 1) // batch_size} model calls total, "
          f"each returning {votes} votes per problem)")

    accepted, rejected = 0, 0
    out = []
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i + batch_size]
        batch_problems = [p for _, p, _ in batch]
        batch_votes = batch_solver_fn(batch_problems, votes)  # list[list[str]]
        for (row, new_problem, changed), sols in zip(batch, batch_votes):
            votes_seen = defaultdict(int)
            solutions_by_answer = {}
            for sol in sols:
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
        print(f"[augment-numeric] {min(i + batch_size, len(jobs))}/{len(jobs)} verified "
              f"(accepted={accepted} rejected={rejected})")

    print(f"[augment-numeric] accepted={accepted} rejected={rejected} "
          f"(agreement threshold={agreement_threshold})")
    ensure_output_path(out_path)
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


def generate_multi_solutions(weak_report: list, math_rows: list, batch_sampler_fn,
                              out_path: str, n_samples: int = 16, keep_top_k: int = 3,
                              temperature: float = 0.9, batch_size: int = 32):
    """
    batch_sampler_fn: callable(problems: list[str], n: int, temperature: float) -> list[list[str]]
        Returns n sampled completions per problem, for a BATCH of problems at once.
    Keeps up to keep_top_k solutions per problem that (a) match the gold answer and
    (b) are structurally distinct from each other (different equation signature).
    """
    weak_keys = {(r["subject"], r["level"]) for r in weak_report}
    targets = [row for row in math_rows if (row["subject"], str(row.get("level", "unknown"))) in weak_keys]
    targets = [row for row in targets if extract_boxed(row["solution"]) is not None]
    print(f"[multi-solution] {len(targets)} source problems in weak clusters")
    print(f"[multi-solution] sampling in batches of {batch_size} "
          f"({(len(targets) + batch_size - 1) // batch_size} model calls total, "
          f"each returning {n_samples} samples per problem)")

    out = []
    for i in range(0, len(targets), batch_size):
        batch = targets[i:i + batch_size]
        batch_problems = [row["problem"] for row in batch]
        batch_samples = batch_sampler_fn(batch_problems, n_samples, temperature)

        for row, samples in zip(batch, batch_samples):
            gold = extract_boxed(row["solution"])
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

            for vi, sol in enumerate(kept):
                out.append({
                    "problem_id": row["problem_id"],
                    "subject": row["subject"],
                    "level": row.get("level", "unknown"),
                    "problem": row["problem"],
                    "solution": sol,
                    "variant_index": vi,
                    "augmentation": "multi_solution_rejection_sampling",
                })
        print(f"[multi-solution] {min(i + batch_size, len(targets))}/{len(targets)} problems sampled")

    ensure_output_path(out_path)
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

    ap.add_argument("--skill-repo", type=str, default=DEFAULT_SKILL_REPO,
                     help="Hugging Face Hub dataset repo id for skill labels")
    ap.add_argument("--skill-labels-file", type=str, default=None,
                     help="optional local jsonl/csv override instead of --skill-repo")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out", type=str, default="outputs/sft_data.jsonl")
    ap.add_argument("--predictions", type=str, default="outputs/predictions.jsonl")
    ap.add_argument("--weak-report", type=str, default="outputs/weak_clusters.json")
    add_destination_args(ap, default_repo_type="dataset")

    args = ap.parse_args()

    if args.build_sft:
        build_sft_dataset(
            out_path=args.out,
            split=args.split,
            skill_repo=args.skill_repo,
            skill_labels_file=args.skill_labels_file,
        )
        dispatch_destination(args.out, args)
    elif args.diagnose:
        diagnose(args.predictions, args.weak_report)
        dispatch_destination(args.weak_report, args)
    else:
        print("Semantic / numeric / multi-solution augmentation require you to pass a "
              "generator_fn / solver_fn / sampler_fn (a live model call) - import this "
              "module and call the functions directly from a script where your model "
              "is loaded (see run_augmentation.py), rather than running this file "
              "standalone for those stages.")
