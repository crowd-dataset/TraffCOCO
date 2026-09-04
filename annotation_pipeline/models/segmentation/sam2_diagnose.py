#!/usr/bin/env python3
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
VENV = ROOT / ".venv"
MODEL_DIR = ROOT / "models" / "segmentation"
REPO = MODEL_DIR / "sam2_repo"
CKPT = MODEL_DIR / "sam2.1_hiera_tiny.pt"

print("Project root:", ROOT)
print("Current Python:", sys.executable)
print("uv:", shutil.which("uv"))
print("Venv:", VENV.exists())

project_python = next(
    (p for p in (VENV / "bin" / "python3", VENV / "bin" / "python") if p.exists()),
    None,
)
print("Project Python:", project_python)

print("SAM2 repo:", REPO.exists(), REPO)
print("Checkpoint:", CKPT.exists(), CKPT)
if CKPT.exists():
    print("Checkpoint MB:", round(CKPT.stat().st_size / 1024 / 1024, 1))

try:
    import sam2
    print("Current-process sam2 import: OK")
    print("Location:", Path(sam2.__file__).resolve())
except Exception as exc:
    print("Current-process sam2 import: FAILED:", exc)

if project_python:
    r = subprocess.run(
        [str(project_python), "-c", "import sam2; print(sam2.__file__)"],
        capture_output=True,
        text=True,
    )
    print("Project-venv sam2 import:", "OK" if r.returncode == 0 else "FAILED")
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.stderr.strip():
        print(r.stderr.strip())
