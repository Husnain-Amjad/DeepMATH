"""
Evaluation module: scores model predictions along four dimensions, then
supports comparing multiple runs (ablations) against a baseline with
statistical-significance-aware charts and tables.

Metric families:
  1. final_correct          - boxed-answer match against gold (always available)
  2. intermediate reasoning  - rule-based arithmetic-consistency proxy on each
     correctness                step's extractable "A = B" assertions (sympy-verified),
                                plus an OPTIONAL LLM-judge hook for a stronger check
                                (see judge_steps_batch / run_judge_eval.py)
  3. skill prediction         - does the model's own predicted skill set (parsed from
     correctness                its <think> section) match the reference minimum_skills
                                for that problem (exact + fuzzy set metrics)
  4. correct skill usage      - for each step where the model tagged a skill, is that
                                skill plausible for this problem's known skill vocabulary
                                (fuzzy-matched against the reference) - a cheaper proxy
                                for "did it use a sensible skill tag" distinct from #3's
                                problem-level set match. A full per-step-position check
                                needs the LLM-judge hook.

IMPORTANT CAVEAT: skill labels (from load_skill_labels) currently only cover the MATH
TRAIN split (per the labeling work described by the user). Metrics #3 and #4 are only
meaningful for predictions on problems that ARE in the labeled set - for predictions on
the test split (which has no skill labels), those metrics come back as None per-example
and are excluded from aggregates, with a coverage percentage printed so this is visible
rather than silently producing a misleading number. final_correct (#1) and the rule-based
arithmetic-consistency proxy (#2) do not depend on the skill-label split at all.

CLI:
  python evaluator.py --score --predictions outputs/predictions_run1.jsonl \
      --split train --out-detailed outputs/eval_run1_detailed.jsonl \
      --out-summary outputs/eval_run1_summary.json

  python evaluator.py --compare --run baseline=outputs/eval_baseline_summary.json \
      --run lora_run1=outputs/eval_run1_summary.json --baseline baseline \
      --out-dir outputs/comparison
"""

import argparse
import json
import re
import math
import difflib
from collections import defaultdict
from pathlib import Path

from reward_fn import extract_boxed, answers_match
from data_pipeline import load_skill_labels, clean_skill_list, DEFAULT_SKILL_REPO
from storage_utils import ensure_output_path, ensure_dir, require_input_path, add_destination_args, dispatch_destination

_SKILL_TAG_RE = re.compile(r"\[SKILL:\s*([^\]]+)\]")
_RELEVANT_SKILLS_HEADER_RE = re.compile(r"Relevant skills:\s*(.+)", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Parsing model output
# ---------------------------------------------------------------------------

def split_think_solution(text: str):
    m_think = _THINK_RE.search(text)
    m_sol = _SOLUTION_RE.search(text)
    think = m_think.group(1).strip() if m_think else None
    solution = m_sol.group(1).strip() if m_sol else None
    return think, solution


def extract_predicted_skills(text: str) -> set:
    skills = set()
    header = _RELEVANT_SKILLS_HEADER_RE.search(text)
    if header:
        for s in header.group(1).split("|"):
            s = s.strip().rstrip(".")
            if s:
                skills.add(s)
    for m in _SKILL_TAG_RE.finditer(text):
        s = m.group(1).strip()
        if s:
            skills.add(s)
    return skills


def split_into_tagged_steps(text: str):
    """Splits on [SKILL: ...] markers -> list of (skill_or_None, step_text)."""
    parts = _SKILL_TAG_RE.split(text)
    if len(parts) == 1:
        return [(None, text.strip())] if text.strip() else []
    steps = []
    pre = parts[0].strip()
    if pre:
        steps.append((None, pre))
    for i in range(1, len(parts), 2):
        skill = parts[i].strip()
        step_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if step_text:
            steps.append((skill, step_text))
    return steps


def parse_model_output(text: str) -> dict:
    think, solution = split_think_solution(text)
    source_for_skills = think if think is not None else text
    predicted_skills = extract_predicted_skills(source_for_skills)
    steps = split_into_tagged_steps(source_for_skills)
    final_answer = extract_boxed(solution) if solution is not None else extract_boxed(text)
    return {
        "think": think, "solution": solution,
        "predicted_skills": predicted_skills, "steps": steps,
        "final_answer": final_answer,
        "has_think": think is not None, "has_boxed": final_answer is not None,
    }


# ---------------------------------------------------------------------------
# Skill-set metrics (metric family #3)
# ---------------------------------------------------------------------------

def normalize_skill(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _f1(p, r):
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def skill_set_metrics(predicted: set, reference: set, fuzzy_threshold: float = 0.8) -> dict:
    pred_norm = {normalize_skill(s) for s in predicted if s}
    ref_norm = {normalize_skill(s) for s in reference if s}

    exact_overlap = pred_norm & ref_norm
    exact_precision = (len(exact_overlap) / len(pred_norm)) if pred_norm else (1.0 if not ref_norm else 0.0)
    exact_recall = (len(exact_overlap) / len(ref_norm)) if ref_norm else (1.0 if not pred_norm else 0.0)

    fuzzy_matched_ref, fuzzy_matched_pred = set(), set()
    for r in ref_norm:
        for p in pred_norm:
            if difflib.SequenceMatcher(None, r, p).ratio() >= fuzzy_threshold:
                fuzzy_matched_ref.add(r)
                fuzzy_matched_pred.add(p)
    fuzzy_precision = (len(fuzzy_matched_pred) / len(pred_norm)) if pred_norm else (1.0 if not ref_norm else 0.0)
    fuzzy_recall = (len(fuzzy_matched_ref) / len(ref_norm)) if ref_norm else (1.0 if not pred_norm else 0.0)

    return {
        "exact_precision": exact_precision, "exact_recall": exact_recall,
        "exact_f1": _f1(exact_precision, exact_recall),
        "fuzzy_precision": fuzzy_precision, "fuzzy_recall": fuzzy_recall,
        "fuzzy_f1": _f1(fuzzy_precision, fuzzy_recall),
        "exact_match": pred_norm == ref_norm,
        "n_predicted": len(pred_norm), "n_reference": len(ref_norm),
    }


# ---------------------------------------------------------------------------
# Rule-based intermediate-reasoning-correctness proxy (metric family #2)
# ---------------------------------------------------------------------------

def _clean_expr_for_sympy(x: str) -> str:
    x = x.strip()
    x = x.replace("$", "").replace("\\left", "").replace("\\right", "")
    x = x.replace("\\cdot", "*").replace("\\times", "*")
    x = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", x)
    x = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", x)
    x = x.replace("\\pi", "pi")
    x = x.replace("^", "**")
    x = x.replace("\\", "")
    return x


def arithmetic_consistency_score(step_text: str):
    """
    Crude, rule-based proxy: finds single 'A = B' assertions per line and checks
    numeric/symbolic equality via sympy. Returns (n_checkable, n_correct) - most
    step text isn't a clean checkable equality (verbal reasoning, multi-equals
    chains, etc.), so this is a lower-bound signal, not a full step-correctness
    judge. Use judge_steps_batch for a stronger LLM-based check.
    """
    from sympy import sympify, simplify
    n_checkable, n_correct = 0, 0
    for line in step_text.splitlines():
        line = line.strip()
        if line.count("=") != 1:
            continue
        lhs, rhs = line.split("=")
        lhs, rhs = _clean_expr_for_sympy(lhs), _clean_expr_for_sympy(rhs)
        if not lhs or not rhs:
            continue
        try:
            vl, vr = sympify(lhs), sympify(rhs)
            n_checkable += 1
            if simplify(vl - vr) == 0:
                n_correct += 1
        except Exception:
            continue
    return n_checkable, n_correct


# ---------------------------------------------------------------------------
# Skill-usage validity proxy (metric family #4)
# ---------------------------------------------------------------------------

def skill_usage_validity(steps, reference_skills: set, fuzzy_threshold: float = 0.8):
    """
    Proxy: for each step where the model tagged a skill, checks whether that
    skill plausibly belongs to this problem's known skill vocabulary (fuzzy
    match against reference minimum_skills/skills_used_in_steps). Catches the
    cheap, common failure mode of a fabricated/off-vocabulary tag anywhere in
    the trace - it does NOT verify the tag is right for that step's specific
    position in the derivation (that needs judge_steps_batch).
    Returns None if the model tagged no steps at all (nothing to evaluate).
    """
    tagged = [(skill, text) for skill, text in steps if skill]
    if not tagged:
        return None
    ref_norm = {normalize_skill(s) for s in reference_skills if s}
    if not ref_norm:
        return None
    n_valid = 0
    for skill, _ in tagged:
        s_norm = normalize_skill(skill)
        if s_norm in ref_norm or any(
            difflib.SequenceMatcher(None, s_norm, r).ratio() >= fuzzy_threshold for r in ref_norm
        ):
            n_valid += 1
    return n_valid / len(tagged)


# ---------------------------------------------------------------------------
# Optional LLM-judge hook for stronger step-level correctness (metric family #2, strong version)
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = (
    "You are grading ONE reasoning step from a student's solution to a math problem.\n"
    "Problem: {problem}\n"
    "Reference correct solution (for your context only): {reference_solution}\n"
    "Step to grade: \"{step_text}\"\n"
    "Is this step mathematically valid and consistent with a correct solution path? "
    "Answer with exactly one word: CORRECT or INCORRECT."
)


def judge_steps_batch(examples_with_steps: list, batch_judge_fn, batch_size: int = 32):
    """
    examples_with_steps: list of {'problem', 'reference_solution', 'steps': [str, ...]}
    batch_judge_fn: callable(prompts: list[str]) -> list[str] (batched model call -
        reuse run_augmentation.build_backend to supply this; see run_judge_eval.py)
    Returns a list of (n_steps_judged, n_judged_correct) aligned to examples_with_steps.
    """
    jobs = []
    for ex_idx, ex in enumerate(examples_with_steps):
        for step_text in ex["steps"]:
            jobs.append((ex_idx, JUDGE_PROMPT_TEMPLATE.format(
                problem=ex["problem"], reference_solution=ex["reference_solution"],
                step_text=step_text,
            )))

    tallies = defaultdict(lambda: [0, 0])  # ex_idx -> [n_total, n_correct]
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i + batch_size]
        outs = batch_judge_fn([p for _, p in batch])
        for (ex_idx, _), out in zip(batch, outs):
            tallies[ex_idx][0] += 1
            out_up = (out or "").upper()
            if "CORRECT" in out_up and "INCORRECT" not in out_up:
                tallies[ex_idx][1] += 1

    return [tuple(tallies.get(i, [0, 0])) for i in range(len(examples_with_steps))]


# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------

def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def evaluate_run(predictions_path: str, out_detailed_path: str, out_summary_path: str,
                  base_split: str = "train", skill_repo: str = DEFAULT_SKILL_REPO,
                  skill_labels_file: str = None, fuzzy_threshold: float = 0.8):
    predictions_path = str(require_input_path(predictions_path))
    base_labels = load_skill_labels(base_split, repo_id=skill_repo, local_path=skill_labels_file)

    detailed = []
    n_with_ref = 0
    with open(predictions_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pid = row["problem_id"]
            ref = base_labels.get(pid)

            parsed = parse_model_output(row["prediction"])
            final_correct = parsed["final_answer"] is not None and answers_match(
                parsed["final_answer"], row["gold_boxed"]
            )

            n_checkable, n_correct = 0, 0
            for _, step_text in parsed["steps"]:
                nc, ncorr = arithmetic_consistency_score(step_text)
                n_checkable += nc
                n_correct += ncorr

            m = {
                "problem_id": pid, "subject": row["subject"], "level": str(row["level"]),
                "final_correct": final_correct,
                "has_think": parsed["has_think"], "has_boxed": parsed["has_boxed"],
                "n_predicted_skills": len(parsed["predicted_skills"]),
                "n_steps": len(parsed["steps"]),
                "step_checkable": n_checkable, "step_arith_correct": n_correct,
                "step_arith_consistency": (n_correct / n_checkable) if n_checkable else None,
                "has_reference": ref is not None,
            }

            if ref is not None:
                n_with_ref += 1
                ref_skills_raw = clean_skill_list(ref.get("minimum_skills", "")) or ref.get("skills_used_in_steps", "")
                ref_skills = {s.strip() for s in ref_skills_raw.split("|") if s.strip()}
                sk = skill_set_metrics(parsed["predicted_skills"], ref_skills, fuzzy_threshold)
                for k, v in sk.items():
                    m[f"skill_{k}"] = v
                m["skill_usage_validity"] = skill_usage_validity(parsed["steps"], ref_skills, fuzzy_threshold)
            else:
                for k in ["exact_precision", "exact_recall", "exact_f1", "fuzzy_precision",
                          "fuzzy_recall", "fuzzy_f1", "exact_match", "n_predicted", "n_reference"]:
                    m[f"skill_{k}"] = None
                m["skill_usage_validity"] = None

            detailed.append(m)

    coverage = (n_with_ref / len(detailed) * 100) if detailed else 0.0
    print(f"[evaluate_run] {len(detailed)} examples scored, {n_with_ref} had skill ground truth "
          f"({coverage:.1f}% coverage - skill metrics only meaningful for these)")
    if n_with_ref == 0:
        print("[evaluate_run] WARNING: 0 examples had skill ground truth. Skill labels currently "
              "only cover the MATH train split - evaluate on (a subset of) train if you need the "
              "skill-prediction / skill-usage metrics; final_correct and arithmetic-consistency "
              "are still valid on any split.")

    ensure_output_path(out_detailed_path)
    with open(out_detailed_path, "w") as f:
        for m in detailed:
            f.write(json.dumps(m) + "\n")

    summary = aggregate_summary(detailed)
    ensure_output_path(out_summary_path)
    with open(out_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[evaluate_run] wrote detailed -> {out_detailed_path}, summary -> {out_summary_path}")
    return summary


def aggregate_summary(detailed: list) -> dict:
    def agg(rows):
        ref_rows = [r for r in rows if r["has_reference"]]
        arith_vals = [r["step_arith_consistency"] for r in rows if r["step_arith_consistency"] is not None]
        usage_vals = [r["skill_usage_validity"] for r in ref_rows if r["skill_usage_validity"] is not None]
        return {
            "n": len(rows),
            "final_accuracy": _mean([r["final_correct"] for r in rows]),
            "format_compliance": _mean([r["has_think"] and r["has_boxed"] for r in rows]),
            "arith_consistency": _mean(arith_vals),
            "arith_coverage": (len(arith_vals) / len(rows)) if rows else 0.0,
            "n_with_skill_ref": len(ref_rows),
            "skill_exact_f1": _mean([r["skill_exact_f1"] for r in ref_rows]) if ref_rows else None,
            "skill_fuzzy_f1": _mean([r["skill_fuzzy_f1"] for r in ref_rows]) if ref_rows else None,
            "skill_exact_match_rate": _mean([r["skill_exact_match"] for r in ref_rows]) if ref_rows else None,
            "skill_usage_validity": _mean(usage_vals) if usage_vals else None,
        }

    by_cluster = defaultdict(list)
    for r in detailed:
        by_cluster[(r["subject"], r["level"])].append(r)

    clusters = {f"{subj}|{lvl}": agg(rows) for (subj, lvl), rows in sorted(by_cluster.items())}
    return {"overall": agg(detailed), "clusters": clusters}


# ---------------------------------------------------------------------------
# Cross-run comparison / ablation studies
# ---------------------------------------------------------------------------

def standard_error(p, n):
    if n is None or n == 0 or p is None:
        return None
    if p <= 0 or p >= 1:
        return 0.0001
    return math.sqrt(p * (1 - p) / n)


OVERALL_METRIC_KEYS = [
    "final_accuracy", "format_compliance", "arith_consistency",
    "skill_exact_f1", "skill_fuzzy_f1", "skill_usage_validity",
]


def compare_runs(run_summaries: dict, out_dir: str, baseline: str = None):
    """run_summaries: {run_name: summary_dict} as produced by evaluate_run."""
    ensure_dir(out_dir)
    names = list(run_summaries.keys())
    baseline = baseline or names[0]
    if baseline not in run_summaries:
        raise SystemExit(f"[compare_runs] baseline '{baseline}' not found among runs: {names}")

    overall_table = []
    for name in names:
        row = {"run": name, "n": run_summaries[name]["overall"]["n"]}
        row.update({k: run_summaries[name]["overall"].get(k) for k in OVERALL_METRIC_KEYS})
        overall_table.append(row)
    with open(Path(out_dir) / "overall_comparison.json", "w") as f:
        json.dump(overall_table, f, indent=2)

    cluster_keys = set()
    for name in names:
        cluster_keys |= set(run_summaries[name]["clusters"].keys())

    ablation_rows = []
    for ck in sorted(cluster_keys):
        subj, lvl = ck.split("|")
        base_cluster = run_summaries[baseline]["clusters"].get(ck)
        if base_cluster is None or base_cluster["final_accuracy"] is None:
            continue
        for name in names:
            if name == baseline:
                continue
            cur_cluster = run_summaries[name]["clusters"].get(ck)
            if cur_cluster is None or cur_cluster["final_accuracy"] is None:
                continue
            base_acc, cur_acc = base_cluster["final_accuracy"], cur_cluster["final_accuracy"]
            base_n, cur_n = base_cluster["n"], cur_cluster["n"]
            p_avg, n_avg = (base_acc + cur_acc) / 2, (base_n + cur_n) / 2
            se = standard_error(p_avg, n_avg)
            diff = cur_acc - base_acc
            se_units = (abs(diff) / se) if se else None
            ablation_rows.append({
                "subject": subj, "level": lvl, "run": name, "baseline": baseline,
                "baseline_acc": round(base_acc, 4), "run_acc": round(cur_acc, 4),
                "diff_pp": round(diff * 100, 2),
                "n_baseline": base_n, "n_run": cur_n,
                "se_units": round(se_units, 2) if se_units is not None else None,
                "likely_real_difference": bool(se_units is not None and se_units > 2),
            })
    with open(Path(out_dir) / "ablation_deltas.json", "w") as f:
        json.dump(ablation_rows, f, indent=2)

    n_flagged = sum(1 for r in ablation_rows if r["likely_real_difference"])
    print(f"[compare_runs] {n_flagged}/{len(ablation_rows)} cluster-level diffs vs baseline "
          f"'{baseline}' exceed 2 standard errors (likely real, not just eval noise)")

    _plot_accuracy_by_level(run_summaries, out_dir)
    _plot_overall_metrics(run_summaries, out_dir)

    print(f"[compare_runs] wrote overall_comparison.json, ablation_deltas.json, and charts -> {out_dir}")
    return overall_table, ablation_rows


def _plot_accuracy_by_level(run_summaries: dict, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subjects = sorted({ck.split("|")[0] for s in run_summaries.values() for ck in s["clusters"]})
    for subj in subjects:
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted_any = False
        for name, summary in run_summaries.items():
            xs, ys = [], []
            for lvl_num in range(1, 6):
                for lvl_key in (f"Level {lvl_num}", str(lvl_num)):
                    c = summary["clusters"].get(f"{subj}|{lvl_key}")
                    if c is not None and c["final_accuracy"] is not None:
                        xs.append(f"L{lvl_num}")
                        ys.append(c["final_accuracy"])
                        break
            if xs:
                ax.plot(xs, ys, marker="o", label=name)
                plotted_any = True
        if not plotted_any:
            plt.close(fig)
            continue
        ax.set_title(f"{subj}: accuracy by level")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"accuracy_by_level_{subj}.png", dpi=150)
        plt.close(fig)


def _plot_overall_metrics(run_summaries: dict, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(run_summaries.keys())
    x = np.arange(len(OVERALL_METRIC_KEYS))
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, name in enumerate(names):
        vals = [run_summaries[name]["overall"].get(k) or 0 for k in OVERALL_METRIC_KEYS]
        ax.bar(x + i * width, vals, width=width, label=name)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(OVERALL_METRIC_KEYS, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Overall metric comparison across runs")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "overall_metrics_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_run_arg(run_args):
    runs = {}
    for item in run_args or []:
        if "=" not in item:
            raise SystemExit(f"[evaluator] --run expects name=path, got: {item}")
        name, path = item.split("=", 1)
        runs[name] = path
    return runs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--compare", action="store_true")

    # --score args
    ap.add_argument("--predictions", type=str, default=None)
    ap.add_argument("--split", type=str, default="train",
                     help="split to load skill labels from - use 'train' if your "
                          "predictions are also on the train split (required for skill "
                          "metrics, since labels only cover train currently)")
    ap.add_argument("--skill-repo", type=str, default=DEFAULT_SKILL_REPO)
    ap.add_argument("--skill-labels-file", type=str, default=None)
    ap.add_argument("--out-detailed", type=str, default="outputs/eval_detailed.jsonl")
    ap.add_argument("--out-summary", type=str, default="outputs/eval_summary.json")
    ap.add_argument("--fuzzy-threshold", type=float, default=0.8)

    # --compare args
    ap.add_argument("--run", action="append", default=[],
                     help="repeatable: --run name=path/to/summary.json")
    ap.add_argument("--baseline", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="outputs/comparison")

    add_destination_args(ap, default_repo_type="dataset")
    args = ap.parse_args()

    if args.score:
        if not args.predictions:
            raise SystemExit("[evaluator] --score requires --predictions")
        evaluate_run(
            args.predictions, args.out_detailed, args.out_summary,
            base_split=args.split, skill_repo=args.skill_repo,
            skill_labels_file=args.skill_labels_file, fuzzy_threshold=args.fuzzy_threshold,
        )
        dispatch_destination(args.out_summary, args)

    elif args.compare:
        run_paths = _parse_run_arg(args.run)
        if not run_paths:
            raise SystemExit("[evaluator] --compare requires at least one --run name=path")
        run_summaries = {}
        for name, path in run_paths.items():
            with open(str(require_input_path(path))) as f:
                run_summaries[name] = json.load(f)
        compare_runs(run_summaries, args.out_dir, baseline=args.baseline)
        dispatch_destination(args.out_dir, args)

    else:
        print("Pass --score (single-run evaluation) or --compare (multi-run ablation "
              "comparison with charts). See module docstring for examples.")
