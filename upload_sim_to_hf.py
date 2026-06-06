"""upload_sim_to_hf.py — push the sim PRISM-JEPA bundle (Cube or PushT) to HuggingFace.

Publishes the weights that reproduce the headline PRISM-MPPI numbers in the
paper (s=1, injection_mode=pog, sigma_floor=0.05, MPPI K=128). Bundle contains
the JEPA world model, the PRISM prior head, the Python source needed to
unpickle the world model, and a README pointing back to the GitHub repo.

Prereqs:
  1. `hf auth login` (write-scope token), OR `HF_TOKEN=hf_...` exported.
  2. Bundle files exist at canonical project locations:
       .stable-wm/<task>/lewm_object.ckpt
       prior_head_<task>.pt
       jepa.py, module.py, prior_head.py
       HF_README_sim_<task>.md
       requirements.txt

Usage:
    python upload_sim_to_hf.py --task cube              # push Cube bundle
    python upload_sim_to_hf.py --task pusht             # push PushT bundle
    python upload_sim_to_hf.py --task cube --dry-run    # inspect only
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path("/home/yuhai/project/prism-jepa")
REPO_TYPE = "model"


def build_file_map(task: str) -> dict[Path, str]:
    return {
        ROOT / f".stable-wm/{task}/lewm_object.ckpt": "lewm_object.ckpt",
        ROOT / f"prior_head_{task}.pt":               f"prior_head_{task}.pt",
        ROOT / "jepa.py":                             "jepa.py",
        ROOT / "module.py":                           "module.py",
        ROOT / "prior_head.py":                       "prior_head.py",
        ROOT / "requirements.txt":                    "requirements.txt",
        ROOT / f"HF_README_sim_{task}.md":            "README.md",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["cube", "pusht"])
    ap.add_argument("--dry-run", action="store_true",
                    help="Only print the bundle inventory; don't push.")
    args = ap.parse_args()

    task = args.task
    repo_id = f"YuhaiW/prism-jepa-{task}"
    file_map = build_file_map(task)

    missing = [src for src in file_map if not src.exists()]
    if missing:
        print("[error] missing files:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    print(f"[bundle] task={task}  repo={repo_id}")
    total = 0
    for src, name in file_map.items():
        sz = src.stat().st_size
        total += sz
        print(f"  {src.name:40s} -> {name:30s}  ({sz / 1e6:.2f} MB)")
    print(f"[bundle] total {total / 1e6:.1f} MB")

    if args.dry_run:
        print("\n[dry-run] would push to:")
        print(f"  https://huggingface.co/{repo_id}  (public)")
        sys.exit(0)

    api = HfApi()
    try:
        whoami = api.whoami()
        print(f"[hf] authenticated as: {whoami.get('name', '???')}")
    except Exception as e:
        print(f"[error] not authenticated to HF: {e}")
        print("  run `hf auth login` first (or set HF_TOKEN env var)")
        sys.exit(1)

    print(f"\n[push] ensuring repo exists: {repo_id}  (public)")
    api.create_repo(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        private=False,
        exist_ok=True,
    )
    # Force public — HF accounts with default-private setting silently ignore
    # private=False on create_repo. Explicit update_repo_settings flips it.
    api.update_repo_settings(repo_id=repo_id, repo_type=REPO_TYPE, private=False)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for src, name in file_map.items():
            dst = td_path / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)

        print(f"\n[push] uploading bundle from {td_path}…")
        url = api.upload_folder(
            repo_id=repo_id,
            folder_path=str(td_path),
            repo_type=REPO_TYPE,
            commit_message=f"Initial PRISM-JEPA sim {task} bundle (paper PRISM-MPPI, s=1)",
            commit_description=(
                f"LeWM (JEPA encoder + ARPredictor) + PRISM prior head trained on "
                f"the sim {task} task, reproducing the headline PRISM-MPPI numbers "
                f"from the paper (s=1, sigma_floor=0.05, MPPI K=128, β-NLL head). "
                f"See https://github.com/YuhaiW/prism-jepa for the eval pipeline."
            ),
        )
        print(f"\n[done] uploaded to: {url}")


if __name__ == "__main__":
    main()
