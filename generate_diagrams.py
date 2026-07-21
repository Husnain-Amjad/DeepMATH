"""
Generates the CONCEPTUAL/architecture diagrams from the paper outline (Figures
1-4: overall framework architecture, skill taxonomy, replay mechanism, training
pipeline). These are illustrative, not data-driven - see generate_figures.py
for the DATA figures (retention curves, skill-wise performance) sourced from
actual experiment results.

Requires graphviz (both the Python package AND the `dot` binary - on most
systems: apt install graphviz / brew install graphviz, then pip install graphviz).

Usage:
  python generate_diagrams.py --diagram framework_overview --out outputs/figures/framework_overview
  python generate_diagrams.py --diagram training_pipeline --out outputs/figures/training_pipeline
  python generate_diagrams.py --diagram replay_mechanism --out outputs/figures/replay_mechanism
  python generate_diagrams.py --all --out_dir outputs/figures/
"""

import argparse
import os

import graphviz


def diagram_framework_overview():
    g = graphviz.Digraph("framework_overview", format="png")
    g.attr(rankdir="TB", fontsize="11", fontname="Helvetica")
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="10")

    g.node("data", "MATH Dataset\n(Hendrycks et al.)", fillcolor="#E8EEF7")
    g.node("labeling", "Skill-Aware\nData Labeling", fillcolor="#D6E4F5")
    g.node("augment", "Semantic + Numeric\nAugmentation", fillcolor="#D6E4F5")
    g.node("replay", "Domain-Level\nReplay Buffer", fillcolor="#D6E4F5")
    g.node("sft", "Skill-Aware SFT\n(Full FT / LoRA)", fillcolor="#F5E6D6")
    g.node("grpo", "GRPO Optimization\n(Correctness + Format +\nPersistence + Chain Stability)", fillcolor="#F5E6D6")
    g.node("eval", "Evaluation\n(4 metric families)", fillcolor="#E6F5E6")
    g.node("model", "Continually-Adapted\nReasoning Model", fillcolor="#F5D6D6", shape="ellipse")

    g.edge("data", "labeling")
    g.edge("labeling", "augment")
    g.edge("augment", "replay")
    g.edge("replay", "sft")
    g.edge("sft", "grpo")
    g.edge("grpo", "eval")
    g.edge("eval", "model")
    g.edge("eval", "labeling", label="  ablation feedback", style="dashed", constraint="false")
    return g


def diagram_skill_taxonomy():
    g = graphviz.Digraph("skill_taxonomy", format="png")
    g.attr(rankdir="TB", fontsize="11", fontname="Helvetica")
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="10")

    g.node("root", "Canonical Skill Vocabulary\n(~45 skills)", fillcolor="#E8EEF7", shape="ellipse")
    subjects = {
        "algebra": ["Solving Linear Equations", "Quadratic Equations", "Polynomial Ops"],
        "geometry": ["Triangle Geometry", "Circle Geometry", "3D Geometry/Volume"],
        "number_theory": ["Modular Arithmetic", "GCD/LCM", "Prime Factorization"],
    }
    for subj, skills in subjects.items():
        g.node(subj, subj.replace("_", " ").title(), fillcolor="#D6E4F5")
        g.edge("root", subj)
        for sk in skills:
            node_id = subj + "_" + sk.replace(" ", "_").replace("/", "_")
            g.node(node_id, sk, fillcolor="#F5F0D6", shape="box", fontsize="9")
            g.edge(subj, node_id)
    g.node("etc", "... (remaining subjects/skills)", shape="plaintext", fontsize="9")
    g.edge("root", "etc", style="dashed")
    return g


def diagram_replay_mechanism():
    g = graphviz.Digraph("replay_mechanism", format="png")
    g.attr(rankdir="LR", fontsize="11", fontname="Helvetica")
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="10")

    g.node("orig", "Original\nSkill-Labeled Data", fillcolor="#E8EEF7")
    g.node("aug", "Augmented Data\n(Semantic + Numeric)", fillcolor="#E8EEF7")
    g.node("strategy", "Replay Strategy\n(none / random /\nbalanced / skill)", fillcolor="#F5E6D6", shape="diamond")
    g.node("buffer", "Mixed Training Batch\n(replay_ratio-controlled)", fillcolor="#D6E4F5")
    g.node("train", "SFT Training Step", fillcolor="#E6F5E6")

    g.edge("orig", "strategy")
    g.edge("aug", "strategy")
    g.edge("strategy", "buffer")
    g.edge("buffer", "train")
    g.edge("train", "buffer", label="  next step", style="dashed", constraint="false")
    return g


def diagram_training_pipeline():
    g = graphviz.Digraph("training_pipeline", format="png")
    g.attr(rankdir="TB", fontsize="11", fontname="Helvetica")
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="10")

    steps = [
        ("s0", "build_sft_dataset\n(data_pipeline.py)"),
        ("s1", "sft_train.py\n--mode full/lora"),
        ("s2", "run_eval.py\n(generate predictions)"),
        ("s3", "evaluator.py --score\n(4 metric families)"),
        ("s4", "experiment_ledger.py --log\n(record + increment)"),
        ("s5", "grpo_train.py\n(4-component reward)"),
        ("s6", "generate_tables.py /\ngenerate_figures.py"),
    ]
    for i, (nid, label) in enumerate(steps):
        g.node(nid, label, fillcolor="#D6E4F5" if i % 2 == 0 else "#F5E6D6")
    for (a, _), (b, _) in zip(steps, steps[1:]):
        g.edge(a, b)
    return g


DIAGRAMS = {
    "framework_overview": diagram_framework_overview,
    "skill_taxonomy": diagram_skill_taxonomy,
    "replay_mechanism": diagram_replay_mechanism,
    "training_pipeline": diagram_training_pipeline,
}


def render(diagram_name, out_path):
    g = DIAGRAMS[diagram_name]()
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    base = out_path[:-4] if out_path.endswith(".png") else out_path
    rendered_path = g.render(filename=base, cleanup=True)
    print(f"[generate_diagrams] wrote {rendered_path}")
    return rendered_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagram", choices=list(DIAGRAMS.keys()), default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=None, help="output path for a single --diagram")
    ap.add_argument("--out_dir", default="outputs/figures", help="output dir for --all")
    args = ap.parse_args()

    if args.all:
        for name in DIAGRAMS:
            render(name, os.path.join(args.out_dir, name))
    elif args.diagram:
        if not args.out:
            raise SystemExit("--diagram requires --out")
        render(args.diagram, args.out)
    else:
        raise SystemExit("pass --diagram <name> --out <path>, or --all --out_dir <dir>")
