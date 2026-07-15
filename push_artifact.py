"""
Push something that ALREADY EXISTS on local disk to Hugging Face Hub or Drive,
without re-running whatever produced it. This is the script you actually want
for `python storage_utils.py --push_to hf ...` - storage_utils.py is a shared
library imported by the other scripts, not something you run directly.

Usage:
  # push an already-trained model directory
  python push_artifact.py --path merged_model --push_to hf \
      --hf_repo_id HusnainAmjad/Qwen_2.5_math_7B_instruct_sft --hf_repo_type model

  # push a dataset file
  python push_artifact.py --path outputs/sft_data.jsonl --push_to hf \
      --hf_repo_id HusnainAmjad/skill-math-sft --hf_repo_type dataset

  # copy to an already-mounted Google Drive path instead
  python push_artifact.py --path merged_model --push_to gdrive \
      --gdrive_path /content/drive/MyDrive/DeepMATH/merged_model
"""

import argparse

from storage_utils import require_input_path, add_destination_args, dispatch_destination

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="local file or directory to push")
    add_destination_args(ap, default_repo_type="model")
    args = ap.parse_args()

    if args.push_to == "none":
        raise SystemExit("[push_artifact] pass --push_to hf or --push_to gdrive - "
                          "'none' means nothing to do.")

    local_path = str(require_input_path(args.path))
    dispatch_destination(local_path, args)
