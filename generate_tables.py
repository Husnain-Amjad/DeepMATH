"""
Generates publication-ready tables (LaTeX/booktabs + Markdown, side by side)
directly from experiment_ledger.py's ledger and evaluator.py's summary files -
no manual transcription from the console output into the paper, which is where
numbers usually drift from what was actually run.

Usage:
  # main results table: one row per logged run
  python generate_tables.py --table main_results --ledger outputs/experiment_ledger.jsonl \
      --out outputs/tables/main_results

  # per-subject/level breakdown for one run
  python generate_tables.py --table subject_level --ledger outputs/experiment_ledger.jsonl \
      --run_id qwen7b_skill_lora_e05 --out outputs/tables/subject_level_qwen7b

  # skill-wise performance for one run (needs evaluator.py's skill_wise_accuracy,
  # i.e. the run's summary.json must have been produced after that feature was added)
  python generate_tables.py --table skill_wise --ledger outputs/experiment_ledger.jsonl \
      --run_id qwen7b_skill_lora_e05 --out outputs/tables/skill_wise_qwen7b --top_n 20

  # side-by-side comparison of specific runs (e.g. an ablation)
  python generate_tables.py --table compare --ledger outputs/experiment_ledger.jsonl \
      --run_ids baseline skill_sft skill_sft_no_replay --out outputs/tables/ablation_replay
"""

import argparse

from experiment_ledger import load_ledger, find_record
from storage_utils import ensure_output_path


def _fmt(v, pct=False):
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v * 100:.2f}" if pct else f"{v:.4f}"
    return str(v)


def _latex_escape(s):
    return str(s).replace("%", "\\%").replace("_", "\\_")


def _write_both(rows, headers, caption, label, out_path):
    """headers should be PLAIN (no LaTeX escaping) - this function escapes for
    the .tex output and leaves the .md output clean, rather than sharing one
    pre-escaped header list that would leak backslashes into Markdown."""
    ensure_output_path(out_path + ".tex")
    ensure_output_path(out_path + ".md")

    latex_headers = [_latex_escape(h) for h in headers]
    latex_rows = [[_latex_escape(c) for c in row] for row in rows]

    # --- LaTeX (booktabs) ---
    col_spec = "l" + "r" * (len(headers) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(latex_headers) + " \\\\",
        "\\midrule",
    ]
    for row in latex_rows:
        lines.append(" & ".join(str(c) for c in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(out_path + ".tex", "w") as f:
        f.write("\n".join(lines) + "\n")

    # --- Markdown (plain, no LaTeX escaping) ---
    md = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        md.append("| " + " | ".join(str(c) for c in row) + " |")
    with open(out_path + ".md", "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"[generate_tables] wrote {out_path}.tex and {out_path}.md ({len(rows)} rows)")


def table_main_results(ledger_path, out_path):
    records = load_ledger(ledger_path)
    if not records:
        raise SystemExit(f"[generate_tables] no records in {ledger_path}")

    headers = ["Run", "Model", "Mode", "n", "Accuracy (%)", "Format", "Arith.", "Skill F1", "Delta vs baseline (pp)"]
    rows = []
    for r in records:
        cfg = r.get("training_config", {})
        o = r["overall"]
        delta = r.get("increment_overall", {}).get("final_accuracy")
        rows.append([
            r["run_id"], cfg.get("model", "?"), cfg.get("mode", "?"), o.get("n", "?"),
            _fmt(o.get("final_accuracy"), pct=True), _fmt(o.get("format_compliance"), pct=True),
            _fmt(o.get("arith_consistency"), pct=True), _fmt(o.get("skill_exact_f1"), pct=True),
            f"{delta*100:+.2f}" if isinstance(delta, float) else "--",
        ])
    _write_both(rows, headers, "Main results across all logged runs.", "tab:main-results", out_path)


def table_subject_level(ledger_path, run_id, out_path):
    record = find_record(ledger_path, run_id)
    if record is None:
        raise SystemExit(f"[generate_tables] run_id '{run_id}' not found in {ledger_path}")

    headers = ["Subject", "Level", "n", "Accuracy (%)"]
    rows = []
    for ck, c in sorted(record.get("clusters", {}).items()):
        subj, lvl = ck.split("|") if "|" in ck else (ck, "")
        rows.append([subj, lvl, c.get("n", "?"), _fmt(c.get("final_accuracy"), pct=True)])
    _write_both(rows, headers, f"Accuracy by subject and difficulty level - {run_id}.",
                f"tab:subject-level-{run_id}", out_path)


def table_skill_wise(ledger_path, run_id, out_path, top_n=20):
    record = find_record(ledger_path, run_id)
    if record is None:
        raise SystemExit(f"[generate_tables] run_id '{run_id}' not found in {ledger_path}")
    skills = record.get("skills")
    if not skills:
        raise SystemExit(f"[generate_tables] run '{run_id}' has no 'skills' breakdown in its "
                          f"logged summary - re-log it from an evaluator.py summary.json that "
                          f"includes skill_wise_accuracy (see evaluator.py's evaluate_run()).")

    headers = ["Skill", "n", "Accuracy (%)"]
    rows = [[skill, s.get("n", "?"), _fmt(s.get("final_accuracy"), pct=True)]
            for skill, s in list(skills.items())[:top_n]]
    _write_both(rows, headers, f"Per-skill accuracy (top {top_n} by frequency) - {run_id}.",
                f"tab:skill-wise-{run_id}", out_path)


def table_compare(ledger_path, run_ids, out_path):
    records = {rid: find_record(ledger_path, rid) for rid in run_ids}
    missing = [rid for rid, r in records.items() if r is None]
    if missing:
        raise SystemExit(f"[generate_tables] run_id(s) not found in ledger: {missing}")

    headers = ["Run", "Accuracy (%)", "Format", "Arith.", "Skill F1", "Skill Usage Val."]
    rows = []
    for rid in run_ids:
        o = records[rid]["overall"]
        rows.append([rid, _fmt(o.get("final_accuracy"), pct=True), _fmt(o.get("format_compliance"), pct=True),
                     _fmt(o.get("arith_consistency"), pct=True), _fmt(o.get("skill_exact_f1"), pct=True),
                     _fmt(o.get("skill_usage_validity"), pct=True)])
    _write_both(rows, headers, "Side-by-side comparison of selected runs.", "tab:compare", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True,
                     choices=["main_results", "subject_level", "skill_wise", "compare"])
    ap.add_argument("--ledger", default="outputs/experiment_ledger.jsonl")
    ap.add_argument("--run_id", default=None, help="for subject_level / skill_wise")
    ap.add_argument("--run_ids", nargs="*", default=None, help="for compare")
    ap.add_argument("--top_n", type=int, default=20, help="for skill_wise")
    ap.add_argument("--out", required=True, help="output path WITHOUT extension - writes .tex and .md")
    args = ap.parse_args()

    if args.table == "main_results":
        table_main_results(args.ledger, args.out)
    elif args.table == "subject_level":
        if not args.run_id:
            raise SystemExit("--table subject_level requires --run_id")
        table_subject_level(args.ledger, args.run_id, args.out)
    elif args.table == "skill_wise":
        if not args.run_id:
            raise SystemExit("--table skill_wise requires --run_id")
        table_skill_wise(args.ledger, args.run_id, args.out, top_n=args.top_n)
    elif args.table == "compare":
        if not args.run_ids:
            raise SystemExit("--table compare requires --run_ids")
        table_compare(args.ledger, args.run_ids, args.out)
