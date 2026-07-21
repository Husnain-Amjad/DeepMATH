# Multi-GPU / Cluster Tutorial

Covers running this pipeline across multiple GPUs on one node, and across
multiple nodes. Read `ROCM_TUTORIAL.md` too if any node is an AMD GPU - the
two are complementary (this file covers *how many GPUs*, that file covers
*which vendor*).

## 0. Decide: do you actually need this?

A single 80GB A100/H100 handles the 1.5B and 7B models in this pipeline
comfortably (see the VRAM budget table in `README.md`). Multi-GPU is worth the
added complexity mainly for: (a) training larger models than fit on one GPU,
(b) parallelizing across your model list (running 2+ of the 7 target models
simultaneously on different GPUs, rather than for speed on any single one),
or (c) sharding vLLM inference to speed up evaluation/augmentation on a big
test set. If none of those apply, skip this file and just run the single-GPU
commands in `PAPER_REPRODUCTION_TUTORIAL.md`.

## 1. Training: full fine-tuning (FSDP) vs. LoRA (DDP)

Full fine-tuning needs the optimizer states and gradients **sharded** across
GPUs (FSDP) since they don't fit replicated on each GPU for a 7B+ model. LoRA's
trainable parameter count is small enough that plain DDP (each GPU holds a full
copy of the adapter + frozen base weights) is simpler and just as fast.

```bash
# Full fine-tuning, multi-GPU, one node
accelerate launch --config_file cluster_configs/accelerate_fsdp_full.yaml \
    sft_train.py --model Qwen/Qwen2.5-Math-7B --data outputs/sft_data.jsonl \
    --mode full --output_dir ckpts/qwen7b_full_cluster --save_every_epochs 0.5

# LoRA, multi-GPU, one node
accelerate launch --config_file cluster_configs/accelerate_ddp_lora.yaml \
    sft_train.py --model Qwen/Qwen2.5-Math-7B --data outputs/sft_data.jsonl \
    --mode lora --output_dir ckpts/qwen7b_lora_cluster --save_every_epochs 0.5
```

**Before running either:** open `cluster_configs/accelerate_fsdp_full.yaml` or
`accelerate_ddp_lora.yaml` and edit `num_processes` to match your actual GPU
count on this node (`nvidia-smi -L | wc -l` or `rocm-smi --showid | grep -c GPU`
to check). The shipped configs default to 4 as a placeholder.

Everything else about the command is identical to the single-GPU version -
`--save_every_epochs`, `--push_every_checkpoint`, `--replay_strategy`, all
work the same way under `accelerate launch` as under plain `python`.

## 2. Multi-node (more than one machine)

Generate a config per node with `accelerate config` interactively (answer
"more than one machine" when prompted), or edit the shipped YAML files
directly:
```yaml
num_machines: 2          # total machine count
machine_rank: 0          # 0 on the main node, 1/2/3... on the others
main_process_ip: <IP of machine_rank 0>
main_process_port: 29500
```
Launch the identical `accelerate launch --config_file ...` command on every
machine - each one reads its own `machine_rank` from its local config file and
coordinates over the network automatically. Confirm all nodes can reach
`main_process_ip:main_process_port` (check firewall/security-group rules)
before launching, or the run will hang at startup with no useful error.

## 3. Multi-GPU inference (evaluation / augmentation / GRPO rollouts)

This is `tensor_parallel_size`, not `accelerate launch` - vLLM shards one
model's weights across GPUs for faster/larger-batch generation:

```bash
python run_eval.py --model ckpts/qwen7b_full_cluster_merged --split test \
    --out outputs/predictions.jsonl --tensor_parallel_size 4

python run_augmentation.py --stage semantic --model ckpts/qwen7b_full_cluster_merged \
    --weak_report outputs/weak_clusters.json --out outputs/semantic_aug.jsonl \
    --tensor_parallel_size 4
```
`--tensor_parallel_size` must evenly divide the model's attention-head count -
if vLLM errors about head divisibility, try a smaller power-of-2 value (2
instead of 4, etc). This flag has no effect on the HF-generate fallback path
(single-GPU only) - it's vLLM-specific.

`grpo_train.py`'s vLLM-backed rollouts pick this up automatically if you also
run it under `accelerate launch` with a multi-GPU config; check
`trl`'s current GRPOTrainer docs for the exact interaction between
`accelerate launch` process count and vLLM's own tensor-parallel setting in
your installed `trl` version, since this detail has changed across `trl`
releases.

## 4. Parallelizing across your model list instead of within one run

If you have multiple GPUs and want to train several of your 7 target models
at once rather than making one training run faster, the simplest approach is
just one `CUDA_VISIBLE_DEVICES`-pinned process per GPU, no `accelerate` needed:

```bash
CUDA_VISIBLE_DEVICES=0 python sft_train.py --model Qwen/Qwen2.5-Math-7B ... &
CUDA_VISIBLE_DEVICES=1 python sft_train.py --model deepseek-ai/deepseek-math-7b-base ... &
CUDA_VISIBLE_DEVICES=2 python sft_train.py --model vanillaOVO/WizardMath-7B-V1.0 ... &
wait
```
Each process only ever sees its assigned GPU (hardware_utils.py's report will
show `gpu_count=1` for each, correctly - not the node's total). This is
usually the better use of multiple GPUs for this pipeline's actual bottleneck,
which is running many *different* model/config combinations, not making one
7B training run itself faster.

## 5. Checking it actually used all the GPUs you expected

```bash
# during training, in another terminal:
watch -n1 nvidia-smi        # CUDA
watch -n1 rocm-smi           # ROCm
```
All GPUs you intended to use should show non-trivial utilization. If only GPU
0 is busy, you launched with plain `python` instead of `accelerate launch` -
`sft_train.py` prints an explicit warning for exactly this case (checks
`WORLD_SIZE` and tells you if it sees multiple GPUs but isn't running under a
multi-process launcher).

## 6. Logging cluster runs into the same experiment ledger

No special handling needed - `experiment_ledger.py --log` reads
`training_config.json` and the eval summary the same way regardless of
whether the run was single-GPU, multi-GPU, or multi-node. Give cluster runs a
`run_id` that makes this visible (e.g. `qwen7b_full_4xA100`) so it's traceable
later without needing to cross-reference which launch command produced it.
