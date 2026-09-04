#!/usr/bin/env python3
from pathlib import Path
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
VENV = ROOT / ".venv"
MODEL_DIR = ROOT / "models" / "segmentation"
REPO = MODEL_DIR / "sam2_repo"
CHECKPOINT = MODEL_DIR / "sam2.1_hiera_tiny.pt"

REPO_URL = "https://github.com/facebookresearch/sam2.git"
CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/"
    "092824/sam2.1_hiera_tiny.pt"
)

uv = shutil.which("uv")
if not uv:
    raise SystemExit("uv was not found in PATH.")

python = next(
    (
        p for p in (
            VENV / "bin" / "python3",
            VENV / "bin" / "python",
        )
        if p.exists()
    ),
    None,
)

if python is None:
    raise SystemExit(
        f"Project venv not found at {VENV}. Run `uv venv` first."
    )

MODEL_DIR.mkdir(parents=True, exist_ok=True)

if not (REPO / "sam2").is_dir():
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(REPO)],
        check=True,
    )

subprocess.run(
    [
        uv, "pip", "install",
        "--python", str(python.resolve()),
        "-e", str(REPO),
    ],
    check=True,
)

if not CHECKPOINT.exists() or CHECKPOINT.stat().st_size <= 100 * 1024 * 1024:
    tmp = CHECKPOINT.with_name(CHECKPOINT.name + ".download")
    if tmp.exists():
        tmp.unlink()
    print(f"Downloading checkpoint to {CHECKPOINT}")
    urllib.request.urlretrieve(CHECKPOINT_URL, tmp)
    tmp.replace(CHECKPOINT)

print("SAM 2.1 setup complete.")
print("Venv:", python.resolve())
print("Repo:", REPO)
print("Checkpoint:", CHECKPOINT)
