"""
Shared storage helpers used by every script in this pipeline:
  - ensure_output_path / ensure_dir: never fail because a folder doesn't exist yet.
  - require_input_path: fail EARLY with a clear message when a *read* path is
    missing, instead of silently creating an empty directory next to it.
  - add_destination_args / dispatch_destination: optional --push_to {hf,gdrive}
    support so any produced artifact (SFT jsonl, augmentation jsonl, weak-cluster
    report, trained model directory) can be sent somewhere other than local disk.

Google Drive note: this does NOT implement an OAuth flow. It assumes Drive is
already mounted as a normal filesystem path (Colab: /content/drive/MyDrive/...,
or an rclone/gdrive-fuse mount elsewhere) and just copies to that path.
"""

import os
import shutil
from pathlib import Path


def ensure_output_path(filepath: str) -> Path:
    """For a FILE you're about to write: creates the parent directory if missing."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(dirpath: str) -> Path:
    """For a DIRECTORY you're about to write into (e.g. a model output_dir)."""
    path = Path(dirpath)
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_input_path(filepath: str) -> Path:
    """
    For a file you're about to READ: fails immediately with a clear message if
    missing, instead of a bare FileNotFoundError, and instead of the actual
    bug this replaces (calling ensure_output_path on a read path, which
    creates the parent dir but does nothing about the missing file itself).
    """
    path = Path(filepath)
    if not path.exists():
        raise SystemExit(
            f"[storage] required input file not found: '{path}' "
            f"(resolved to '{path.resolve()}'). Check the path, or run the "
            f"stage that produces this file first."
        )
    return path


def push_to_hf(local_path: str, repo_id: str, repo_type: str = "dataset",
               private: bool = False, path_in_repo: str = None,
               token: str = None, commit_message: str = None):
    """Uploads a local file or directory to a Hugging Face Hub repo, creating it if needed.
    For folders, path_in_repo places the contents under that subfolder in the repo instead
    of the repo root - this is what makes it safe to push multiple checkpoints from the
    same training run to one repo without each one overwriting the last."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)

    p = Path(local_path)
    msg = commit_message or f"Upload {p.name}"
    if p.is_dir():
        api.upload_folder(folder_path=str(p), repo_id=repo_id, repo_type=repo_type,
                           path_in_repo=path_in_repo, commit_message=msg)
    else:
        api.upload_file(path_or_fileobj=str(p), path_in_repo=path_in_repo or p.name,
                         repo_id=repo_id, repo_type=repo_type, commit_message=msg)
    dest_note = f"/{path_in_repo}" if path_in_repo else ""
    print(f"[storage] pushed '{local_path}' -> hf://{repo_id}{dest_note} ({repo_type})")


def copy_to_path(local_path: str, dest_path: str):
    """Copies a local file or directory to another filesystem path (covers the
    'save to Drive' case when Drive is already mounted)."""
    src = Path(local_path)
    dest = Path(dest_path)
    if src.is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    print(f"[storage] copied '{local_path}' -> '{dest_path}'")


def add_destination_args(parser, default_repo_type: str = "dataset"):
    """Adds the common --push_to / --hf_* / --gdrive_path flags to an argparse parser."""
    parser.add_argument("--push_to", choices=["none", "hf", "gdrive"], default="none",
                         help="optionally send the produced artifact somewhere after "
                              "saving it locally. Local save always happens regardless.")
    parser.add_argument("--hf_repo_id", type=str, default=None,
                         help="e.g. 'yourname/skill-math-sft' - required if --push_to hf")
    parser.add_argument("--hf_repo_type", choices=["dataset", "model"], default=default_repo_type)
    parser.add_argument("--hf_private", action="store_true", default=False)
    parser.add_argument("--hf_token", type=str, default=None,
                         help="explicit HF write token. If omitted, falls back to the "
                              "HF_TOKEN environment variable, then to any cached "
                              "`huggingface-cli login` credentials.")
    parser.add_argument("--hf_path_in_repo", type=str, default=None,
                         help="destination path inside the repo - filename for a single "
                              "file, or subfolder for a directory (e.g. 'checkpoint-113' "
                              "so multiple checkpoints pushed to the same repo don't "
                              "overwrite each other)")
    parser.add_argument("--gdrive_path", type=str, default=None,
                         help="destination path under your already-mounted Drive - "
                              "required if --push_to gdrive")
    return parser


def dispatch_destination(local_path: str, args):
    """Call after saving local_path, passing the parsed argparse Namespace."""
    push_to = getattr(args, "push_to", "none")
    if push_to == "none":
        return
    if push_to == "hf":
        if not getattr(args, "hf_repo_id", None):
            raise SystemExit("[storage] --push_to hf requires --hf_repo_id")
        token = getattr(args, "hf_token", None) or os.environ.get("HF_TOKEN")
        push_to_hf(local_path, args.hf_repo_id, repo_type=args.hf_repo_type,
                   private=args.hf_private, path_in_repo=args.hf_path_in_repo, token=token)
    elif push_to == "gdrive":
        if not getattr(args, "gdrive_path", None):
            raise SystemExit("[storage] --push_to gdrive requires --gdrive_path")
        copy_to_path(local_path, args.gdrive_path)
