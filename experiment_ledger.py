"""
Append-only experiment ledger: one structured record per (model, fine-tune mode,
data config, checkpoint) combination, capturing the full training-pipeline spec
alongside the resulting overall/domain/level accuracy AND the increment versus
a named baseline record - this is the "record not being saved" gap: previously
each run's predictions/summary lived in its own file with no persistent link
back to the config that produced it, and no computed delta over time.

Storage: JSONL (one record per line, append-only - safe even if a session dies
mid-write, since completed lines before the crash are still intact and valid)
plus a --export-csv flattened view for spreadsheet use (Excel/Google Sheets).

CLI:
  # log one run (after sft_train.py + run_eval.py + evaluator.py --score)
  python experiment_ledger.py --log \
      --run_id qwen7b_skill_lora_e05 \
      --training_config ckpts/qwen7b_run/training_config.json \
      --eval_summary outputs/eval_qwen7b_lora_e05_summary.json \
      --baseline_run_id qwen7b_baseline \
      --ledger outputs/experiment_ledger.jsonl

  # view the ledger as a table
  python experiment_ledger.py --print --ledger outputs/experiment_ledger.jsonl

  # export to CSV for Excel/Sheets
  python experiment_ledger.py --export-csv --ledger outputs/experiment_ledger.jsonl \
      --out outputs/experiment_ledger.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

from storage_utils import ensure_output_path, require_input_path, add_destination_args, dispatch_destination


def load_ledger(ledger_path: str) -> list:
    p = Path(ledger_path)
    if not p.exists():
        return []
    records = []
    with open(p) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def find_record(ledger_path: str, run_id: str):
    for r in load_ledger(ledger_path):
        if r["run_id"] == run_id:
            return r
    return None


def compute_deltas(overall: dict, baseline_overall: dict) -> dict:
    """Per-metric delta (current - baseline), for whichever numeric metrics both share."""
    deltas = {}
    for k, v in overall.items():
        bv = baseline_overall.get(k)
        if isinstance(v, (int, float)) and isinstance(bv, (int, float)):
            deltas[k] = round(v - bv, 4)
    return deltas


def compute_cluster_deltas(clusters: dict, baseline_clusters: dict) -> dict:
    """Per (subject,level) cluster delta in final_accuracy, where both runs have that cluster."""
    out = {}
    for ck, c in clusters.items():
        bc = baseline_clusters.get(ck)
        if bc and c.get("final_accuracy") is not None and bc.get("final_accuracy") is not None:
            out[ck] = round(c["final_accuracy"] - bc["final_accuracy"], 4)
    return out


def append_record(ledger_path: str, run_id: str, training_config_path: str,
                   eval_summary_path: str, baseline_run_id: str = None, notes: str = ""):
    training_config = json.loads(require_input_path(training_config_path).read_text())
    eval_summary = json.loads(require_input_path(eval_summary_path).read_text())

    record = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": notes,
        "training_config": training_config,
        "overall": eval_summary.get("overall", {}),
        "clusters": eval_summary.get("clusters", {}),
        "baseline_run_id": baseline_run_id,
        "increment_overall": {},
        "increment_clusters": {},
    }

    if baseline_run_id:
        baseline = find_record(ledger_path, baseline_run_id)
        if baseline is None:
            print(f"[experiment_ledger] WARNING: baseline_run_id '{baseline_run_id}' not found "
                  f"in ledger yet - increments will be empty. Log the baseline run first, or "
                  f"double check the run_id spelling.")
        else:
            record["increment_overall"] = compute_deltas(record["overall"], baseline["overall"])
            record["increment_clusters"] = compute_cluster_deltas(record["clusters"], baseline["clusters"])

    existing = find_record(ledger_path, run_id)
    if existing is not None:
        print(f"[experiment_ledger] WARNING: run_id '{run_id}' already exists in the ledger - "
              f"appending anyway (ledger is append-only; use the latest matching entry when reading).")

    ensure_output_path(ledger_path)
    with open(ledger_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"[experiment_ledger] logged run_id='{run_id}' -> {ledger_path}")
    if record["increment_overall"]:
        print(f"[experiment_ledger] increment vs '{baseline_run_id}': {record['increment_overall']}")
    return record


def print_ledger(ledger_path: str):
    records = load_ledger(ledger_path)
    if not records:
        print(f"[experiment_ledger] no records yet in {ledger_path}")
        return
    header = f"{'run_id':30s} {'mode':8s} {'model':30s} {'acc':>8s} {'vs baseline':>12s} {'baseline':20s}"
    print(header)
    print("-" * len(header))
    for r in records:
        cfg = r.get("training_config", {})
        model = str(cfg.get("model", "?"))[:30]
        mode = str(cfg.get("mode", "?"))
        acc = r["overall"].get("final_accuracy")
        acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else "?"
        delta = r["increment_overall"].get("final_accuracy")
        delta_str = f"{delta:+.4f}" if isinstance(delta, (int, float)) else "-"
        baseline = r.get("baseline_run_id") or "-"
        print(f"{r['run_id']:30s} {mode:8s} {model:30s} {acc_str:>8s} {delta_str:>12s} {baseline:20s}")


def export_csv(ledger_path: str, out_path: str):
    records = load_ledger(ledger_path)
    if not records:
        print(f"[experiment_ledger] no records to export from {ledger_path}")
        return

    rows = []
    for r in records:
        cfg = r.get("training_config", {})
        base_row = {
            "run_id": r["run_id"], "timestamp": r["timestamp"],
            "model": cfg.get("model"), "mode": cfg.get("mode"),
            "replay_strategy": cfg.get("replay_strategy"), "replay_ratio": cfg.get("replay_ratio"),
            "epochs": cfg.get("epochs"), "seed": cfg.get("seed"),
            "lora_r": cfg.get("lora_r"), "lora_alpha": cfg.get("lora_alpha"),
            "baseline_run_id": r.get("baseline_run_id"),
            "cluster": "OVERALL", "subject": "", "level": "",
            "n": r["overall"].get("n"),
            "final_accuracy": r["overall"].get("final_accuracy"),
            "delta_vs_baseline": r["increment_overall"].get("final_accuracy"),
            "skill_exact_f1": r["overall"].get("skill_exact_f1"),
            "skill_usage_validity": r["overall"].get("skill_usage_validity"),
        }
        rows.append(base_row)
        for ck, c in r.get("clusters", {}).items():
            subj, lvl = ck.split("|") if "|" in ck else (ck, "")
            rows.append({
                "run_id": r["run_id"], "timestamp": r["timestamp"],
                "model": cfg.get("model"), "mode": cfg.get("mode"),
                "replay_strategy": cfg.get("replay_strategy"), "replay_ratio": cfg.get("replay_ratio"),
                "epochs": cfg.get("epochs"), "seed": cfg.get("seed"),
                "lora_r": cfg.get("lora_r"), "lora_alpha": cfg.get("lora_alpha"),
                "baseline_run_id": r.get("baseline_run_id"),
                "cluster": ck, "subject": subj, "level": lvl,
                "n": c.get("n"), "final_accuracy": c.get("final_accuracy"),
                "delta_vs_baseline": r.get("increment_clusters", {}).get(ck),
                "skill_exact_f1": c.get("skill_exact_f1"),
                "skill_usage_validity": c.get("skill_usage_validity"),
            })

    ensure_output_path(out_path)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[experiment_ledger] exported {len(rows)} rows ({len(records)} runs, overall + per-cluster) -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--print", action="store_true", dest="print_ledger")
    ap.add_argument("--export-csv", action="store_true")

    ap.add_argument("--ledger", default="outputs/experiment_ledger.jsonl")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--training_config", default=None)
    ap.add_argument("--eval_summary", default=None)
    ap.add_argument("--baseline_run_id", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", default="outputs/experiment_ledger.csv")
    add_destination_args(ap, default_repo_type="dataset")
    args = ap.parse_args()

    if args.log:
        if not (args.run_id and args.training_config and args.eval_summary):
            raise SystemExit("[experiment_ledger] --log requires --run_id, --training_config, --eval_summary")
        append_record(args.ledger, args.run_id, args.training_config, args.eval_summary,
                      baseline_run_id=args.baseline_run_id, notes=args.notes)
        dispatch_destination(args.ledger, args)
    elif args.print_ledger:
        print_ledger(args.ledger)
    elif args.export_csv:
        export_csv(args.ledger, args.out)
        dispatch_destination(args.out, args)
    else:
        print("Pass --log, --print, or --export-csv. See module docstring for examples.")
