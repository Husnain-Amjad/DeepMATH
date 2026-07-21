# Paper Reproduction Tutorial: Template Use -> GRPO -> Tables/Figures

This is the master walkthrough tying every paper/thesis section to the exact
commands that produce its evidence, which files get written at each step, and
which table/figure generator turns those files into paper-ready output. Read
alongside `USAGE.md` (stage-by-stage command reference) and `TUTORIAL.md`
(save/load/push mechanics) - this file is the connective tissue between "run
commands" and "here is Table 5.1 / Figure 6".

Run everything below per model (repeat the whole sequence once per entry in
your model list) and per fine-tune mode (`--mode full` and `--mode lora` each
get their own run, for the side-by-side comparison the paper is built around).

---

## Section 3.4 / 3.5 (Template Design + SFT): from raw data to a trained checkpoint

**What changed vs. earlier in this pipeline's history:** `data_pipeline.py` no
longer bakes one hardcoded chat template into the dataset. It stores raw
`problem`/`think`/`solution`/`skills` fields, and `templates.py` renders them
using **whichever tokenizer you actually pass to `sft_train.py`** - this is
the "template integration when model changes" mechanism. You do not edit any
template string when you switch models; you just point `--model` at a
different repo id.

```bash
# Step 1: build the raw (template-agnostic) SFT dataset once
python data_pipeline.py --build-sft --split train --out outputs/sft_data.jsonl
#   -> writes outputs/sft_data.jsonl (raw fields, reusable across ALL models)

# Step 2: train - repeat once per model, once per mode
python sft_train.py --model Qwen/Qwen2.5-Math-7B --data outputs/sft_data.jsonl \
    --mode full --output_dir ckpts/qwen7b_full --save_every_epochs 0.5 \
    --replay_strategy random --replay_ratio 0.7
#   -> writes ckpts/qwen7b_full/training_config.json   (Section 5.2 hyperparameter table source)
#   -> writes ckpts/qwen7b_full/checkpoint-{N}/         (one per 0.5-epoch interval)
#   -> writes ckpts/qwen7b_full_merged/                 (final checkpoint, standalone - for full FT
#                                                          this is identical to output_dir, since
#                                                          full FT has no adapter to merge)

python sft_train.py --model Qwen/Qwen2.5-Math-7B --data outputs/sft_data.jsonl \
    --mode lora --output_dir ckpts/qwen7b_lora --save_every_epochs 0.5 \
    --replay_strategy random --replay_ratio 0.7
#   -> writes ckpts/qwen7b_lora/checkpoint-{N}/         (adapter-only)
#   -> writes ckpts/qwen7b_lora_merged/                 (final checkpoint, auto-merged & standalone)
```

Swap `--model` for any of your 7 target models and repeat - `templates.py`
detects whether the tokenizer has a chat template (Qwen-Instruct, NuminaMath)
or not (Qwen base, deepseek-math-7b-base) and renders correctly either way,
with no other change needed. See `dump_model_template.py` if you want to
confirm ahead of time what a given model's rendered prompt looks like.

**A100 / H100 / ROCm / cluster note:** every script above prints a hardware
report at startup (`hardware_utils.py`) confirming what it detected and which
defaults it picked - check this line first if a run behaves unexpectedly
differently across machines. For multi-GPU, see the Cluster section near the
bottom of this file instead of running the plain `python` command above.

---

## Section 3.2 (Mixed Perturbation) + Section 3.3 (Replay): augmentation

```bash
# diagnose weak clusters from the checkpoint you want to augment around
python run_eval.py --model ckpts/qwen7b_lora_merged --split test \
    --out outputs/predictions_qwen7b_lora.jsonl
python data_pipeline.py --diagnose --predictions outputs/predictions_qwen7b_lora.jsonl \
    --weak-report outputs/weak_clusters.json
#   -> writes outputs/predictions_qwen7b_lora.jsonl   (Section 6 raw eval output)
#   -> writes outputs/weak_clusters.json               (input to augmentation below)

# semantic + numeric augmentation (Section 3.2)
python run_augmentation.py --stage semantic --model ckpts/qwen7b_lora_merged \
    --weak_report outputs/weak_clusters.json --out outputs/semantic_aug.jsonl
python run_augmentation.py --stage numeric --model ckpts/qwen7b_lora_merged \
    --weak_report outputs/weak_clusters.json --out outputs/numeric_aug.jsonl
#   -> writes outputs/semantic_aug.jsonl, outputs/numeric_aug.jsonl

# re-train WITH replay, mixing original + augmented data (Section 3.3)
python sft_train.py --model Qwen/Qwen2.5-Math-7B --data outputs/sft_data.jsonl \
    --extra_data outputs/semantic_aug.jsonl outputs/numeric_aug.jsonl \
    --replay_strategy skill --replay_ratio 0.7 \
    --mode lora --output_dir ckpts/qwen7b_lora_replay --save_every_epochs 0.5

# ABLATION: the same run WITHOUT replay, for the forgetting-mitigation comparison
# (Section 6.3 / Figure 5 - this is what makes the "with vs without replay" curve possible)
python sft_train.py --model Qwen/Qwen2.5-Math-7B --data outputs/sft_data.jsonl \
    --extra_data outputs/semantic_aug.jsonl outputs/numeric_aug.jsonl \
    --replay_strategy none \
    --mode lora --output_dir ckpts/qwen7b_lora_noreplay --save_every_epochs 0.5
```

`--replay_strategy` has four values (Section 5.4/6.4 "Replay Analysis" -
random/balanced/skill/none) - re-run the augmented-training command once per
strategy, changing only `--replay_strategy` and `--output_dir`, to populate
that comparison directly.

---

## Section 3.6 / 4.5-4.6 (GRPO Optimization): the reward decomposition

```bash
python grpo_train.py --model ckpts/qwen7b_lora_replay_merged --data outputs/sft_data.jsonl \
    --output_dir ckpts/qwen7b_grpo \
    --w_correctness 1.0 --w_format 0.2 --w_persistence 0.15 --w_chain_stability 0.25
#   -> writes ckpts/qwen7b_grpo/  (final RL-tuned model)
```

The four reward weights map directly to the paper's reward decomposition
(Section 4.5): correctness, format fidelity, persistence, and chain stability
each have their own `--w_*` flag and are computed by independent, separately
testable functions in `grpo_train.py` (`correctness_reward_fn`,
`format_reward_fn`, `persistence_reward_fn`, `chain_stability_reward_fn`) -
report these four weights directly in your Implementation Details table
(Section 5.6).

---

## Section 6 (Results): evaluating and logging every run

This is the step that makes every later table/figure automatic rather than a
manual transcription job. Do this **after every checkpoint you want reported**
- baseline, each SFT config, each replay strategy, each GRPO run:

```bash
# 1. generate predictions
python run_eval.py --model ckpts/qwen7b_grpo --split test \
    --out outputs/predictions_qwen7b_grpo.jsonl

# 2. score against the four metric families (final-answer, arithmetic
#    consistency, skill-prediction, skill-usage) - this ALSO computes the
#    skill-wise breakdown (Section 6.5) automatically
python evaluator.py --score --predictions outputs/predictions_qwen7b_grpo.jsonl \
    --split train --out-detailed outputs/eval_qwen7b_grpo_detailed.jsonl \
    --out-summary outputs/eval_qwen7b_grpo_summary.json
#   NOTE: --split train is required to get skill-prediction/skill-usage metrics,
#   since skill labels currently only cover the MATH train split - see the
#   coverage percentage evaluator.py prints; final-answer accuracy works on
#   any split regardless.

# 3. log it into the persistent experiment ledger, with the increment vs a named baseline
python experiment_ledger.py --log --run_id qwen7b_grpo \
    --training_config ckpts/qwen7b_grpo/training_config.json \
    --eval_summary outputs/eval_qwen7b_grpo_summary.json \
    --baseline_run_id qwen7b_baseline \
    --ledger outputs/experiment_ledger.jsonl
#   -> appends one record to outputs/experiment_ledger.jsonl containing:
#      the full training config, overall accuracy, per-(subject,level) accuracy,
#      per-skill accuracy, AND the computed increment over the named baseline -
#      this one file is the source of truth for every table below.
```

Repeat step 3's `--run_id` for every run you did - baseline, skill-SFT,
+semantic, +numeric, +both, each replay strategy, each GRPO variant. Give each
a clear, distinct `run_id` (e.g. `qwen7b_baseline`, `qwen7b_skillsft_lora_e05`,
`qwen7b_replay_skill_lora`, `qwen7b_grpo`) - these names are what you'll pass
to the table/figure generators next, so name them something you'll recognize
in six months.

---

## Turning the ledger into paper tables (LaTeX + Markdown, no manual transcription)

```bash
# Main results table (Section 6.1 / your thesis Table 5.6-equivalent):
# one row per logged run, straight from the ledger
python generate_tables.py --table main_results --ledger outputs/experiment_ledger.jsonl \
    --out outputs/tables/main_results
#   -> outputs/tables/main_results.tex   (paste into the paper directly)
#   -> outputs/tables/main_results.md    (paste into the thesis / GitHub README)

# Per-subject/level breakdown for one run (Section 6.6 Difficulty Analysis)
python generate_tables.py --table subject_level --run_id qwen7b_grpo \
    --ledger outputs/experiment_ledger.jsonl --out outputs/tables/subject_level_grpo

# Per-skill breakdown for one run (Section 6.5 Skill-wise Performance)
python generate_tables.py --table skill_wise --run_id qwen7b_grpo --top_n 20 \
    --ledger outputs/experiment_ledger.jsonl --out outputs/tables/skill_wise_grpo

# Side-by-side ablation comparison (Section 6.2 Ablation Analysis) -
# pass exactly the run_ids you want compared, in the order you want them shown
python generate_tables.py --table compare \
    --run_ids qwen7b_baseline qwen7b_skillsft_lora qwen7b_replay_skill_lora qwen7b_grpo \
    --ledger outputs/experiment_ledger.jsonl --out outputs/tables/ablation_main
```

---

## Turning the ledger into paper figures

```bash
# Retention/forgetting curve WITH vs WITHOUT replay (Section 6.3, Figure 5) -
# this is the single most important figure for the replay-mitigation claim
python generate_figures.py --figure retention_curve \
    --run_ids qwen7b_replay_e05 qwen7b_replay_e1 qwen7b_replay_e15 qwen7b_replay_e2 \
    --x_values 0.5 1.0 1.5 2.0 --x_label "Epoch" \
    --baseline_run_id qwen7b_baseline \
    --ledger outputs/experiment_ledger.jsonl --out outputs/figures/retention_with_replay.png

python generate_figures.py --figure retention_curve \
    --run_ids qwen7b_noreplay_e05 qwen7b_noreplay_e1 qwen7b_noreplay_e15 qwen7b_noreplay_e2 \
    --x_values 0.5 1.0 1.5 2.0 --x_label "Epoch" \
    --baseline_run_id qwen7b_baseline \
    --ledger outputs/experiment_ledger.jsonl --out outputs/figures/retention_no_replay.png

# Skill-wise performance bar chart (Figure 6)
python generate_figures.py --figure skill_wise --run_id qwen7b_grpo --top_n 15 \
    --ledger outputs/experiment_ledger.jsonl --out outputs/figures/skill_wise_grpo.png
```

If you want the with/without-replay comparison as ONE chart (two lines on the
same axes) rather than two separate images, use `plot_retention_comparison()`
directly from a short Python script - it's a function in `generate_figures.py`
not yet wired to its own CLI flag; the two-call version above is the CLI path
and produces equivalent information as two files.

## Conceptual architecture diagrams (Figures 1-4 - not data-driven)

```bash
python generate_diagrams.py --all --out_dir outputs/figures/
#   -> outputs/figures/framework_overview.png
#   -> outputs/figures/skill_taxonomy.png
#   -> outputs/figures/replay_mechanism.png
#   -> outputs/figures/training_pipeline.png
```
These are illustrative diagrams built from the pipeline's own structure, not
from experiment results - edit `generate_diagrams.py`'s node/edge definitions
directly if you want to adjust wording or add boxes, rather than regenerating
from data.

---

## Cluster / multi-GPU note

Everything above assumes one GPU. For multi-GPU:
```bash
accelerate launch --config_file cluster_configs/accelerate_fsdp_full.yaml \
    sft_train.py --model Qwen/Qwen2.5-Math-7B --data outputs/sft_data.jsonl \
    --mode full --output_dir ckpts/qwen7b_full_cluster
# (use accelerate_ddp_lora.yaml instead for --mode lora)
```
and for multi-GPU inference (sharding a large model across GPUs for
eval/augmentation), add `--tensor_parallel_size <N>` to `run_eval.py` /
`run_augmentation.py`. See `CLUSTER_TUTORIAL.md` for the full walkthrough,
`ROCM_TUTORIAL.md` if any of the cluster's nodes are AMD GPUs instead of NVIDIA.

---

## Quick sanity check: does the whole chain actually connect?

```bash
python check_environment.py                       # confirms your env is ready
python -c "from experiment_ledger import load_ledger; print(len(load_ledger('outputs/experiment_ledger.jsonl')), 'runs logged so far')"
```
If a table/figure generator complains a `run_id` isn't found, this second
command tells you what's actually in the ledger right now - the generators
never invent or interpolate missing data, they fail loudly with the exact
missing `run_id` named, by design.
