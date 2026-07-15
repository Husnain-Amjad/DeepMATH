"""
Ground-truth template/EOS extraction: loads each model's actual tokenizer and
reports what it will really do, rather than relying on documentation (which
can drift from the shipped tokenizer_config.json, and doesn't exist at all
for some of these models - several have no chat template whatsoever).

Usage:
  python dump_model_template.py --models \
      "Qwen/Qwen2.5-Math-1.5B" "Qwen/Qwen2.5-Math-1.5B-Instruct" \
      "Qwen/Qwen2.5-Math-7B-Instruct" "deepseek-ai/deepseek-math-7b-base" \
      "deepseek-ai/deepseek-math-7b-rl" "vanillaOVO/WizardMath-7B-V1.0" \
      "AI-MO/NuminaMath-7B-CoT" \
      --out outputs/model_templates.json
"""

import argparse
import json

from transformers import AutoTokenizer
from storage_utils import ensure_output_path


def probe_model(model_id: str) -> dict:
    info = {"model_id": model_id}
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return info

    info["eos_token"] = tok.eos_token
    info["eos_token_id"] = tok.eos_token_id
    info["bos_token"] = tok.bos_token
    info["pad_token"] = tok.pad_token
    info["has_chat_template"] = tok.chat_template is not None
    info["chat_template_raw"] = tok.chat_template

    if tok.chat_template is not None:
        sample_msgs = [
            {"role": "system", "content": "SYSTEM_PLACEHOLDER"},
            {"role": "user", "content": "USER_PLACEHOLDER"},
        ]
        try:
            info["rendered_with_system"] = tok.apply_chat_template(
                sample_msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            info["rendered_with_system_error"] = f"{type(e).__name__}: {e}"
            sample_msgs = [{"role": "user", "content": "USER_PLACEHOLDER"}]
            try:
                info["rendered_user_only"] = tok.apply_chat_template(
                    sample_msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception as e2:
                info["rendered_user_only_error"] = f"{type(e2).__name__}: {e2}"
    else:
        info["note"] = "No chat template - this is a base/completion model. " \
                        "Use plain text prompts, not apply_chat_template."

    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/model_templates.json")
    args = ap.parse_args()

    results = {}
    for model_id in args.models:
        print(f"[dump_model_template] probing {model_id} ...")
        results[model_id] = probe_model(model_id)
        if "error" in results[model_id]:
            print(f"  ERROR: {results[model_id]['error']}")
        else:
            print(f"  eos_token={results[model_id]['eos_token']!r} "
                  f"has_chat_template={results[model_id]['has_chat_template']}")

    ensure_output_path(args.out)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[dump_model_template] wrote full details -> {args.out}")
