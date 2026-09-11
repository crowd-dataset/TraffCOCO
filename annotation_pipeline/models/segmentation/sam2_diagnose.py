#!/usr/bin/env python3

from pathlib import Path
import shutil
import subprocess
import sys


def find_project_root() -> Path:
    """
    Find the TraffCOCO project root by locating the repository markers.
    """
    current = Path(__file__).resolve()

    for candidate in (current.parent, *current.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "annotation_pipeline").is_dir()
        ):
            return candidate

    raise RuntimeError(
        "Could not locate the TraffCOCO project root. "
        "Expected a directory containing both 'pyproject.toml' "
        "and 'annotation_pipeline'."
    )


ROOT = find_project_root()

VENV = ROOT / ".venv"

MODEL_DIR = (
    ROOT
    / "annotation_pipeline"
    / "models"
    / "segmentation"
)

REPO = MODEL_DIR / "sam2_repo"
CHECKPOINT = MODEL_DIR / "sam2.1_hiera_tiny.pt"


# ----------------------------------------------------------------------
# Basic environment information
# ----------------------------------------------------------------------

print("Project root:", ROOT)
print("Current Python:", sys.executable)
print("Platform:", sys.platform)
print("uv:", shutil.which("uv"))
print("Venv:", VENV.exists())


# ----------------------------------------------------------------------
# Project Python
# ----------------------------------------------------------------------

if sys.platform == "win32":
    candidates = (
        VENV / "Scripts" / "python.exe",
    )
else:
    candidates = (
        VENV / "bin" / "python3",
        VENV / "bin" / "python",
    )


project_python = next(
    (
        path
        for path in candidates
        if path.exists()
    ),
    None,
)


print("Project Python:", project_python)


# ----------------------------------------------------------------------
# SAM 2 files
# ----------------------------------------------------------------------

print("SAM2 repo:", REPO.exists(), REPO)

print(
    "Checkpoint:",
    CHECKPOINT.exists(),
    CHECKPOINT,
)

if CHECKPOINT.exists():
    print(
        "Checkpoint MB:",
        round(
            CHECKPOINT.stat().st_size / 1024 / 1024,
            1,
        ),
    )


# ----------------------------------------------------------------------
# Current-process SAM 2 import
# ----------------------------------------------------------------------

try:
    import sam2

    print("Current-process sam2 import: OK")
    print(
        "Location:",
        Path(sam2.__file__).resolve(),
    )

except Exception as exc:
    print(
        "Current-process sam2 import: FAILED:",
        exc,
    )


# ----------------------------------------------------------------------
# Project-venv SAM 2 import
# ----------------------------------------------------------------------

if project_python:

    result = subprocess.run(
        [
            str(project_python),
            "-c",
            "import sam2; print(sam2.__file__)",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    print(
        "Project-venv sam2 import:",
        "OK" if result.returncode == 0 else "FAILED",
    )

    if result.stdout.strip():
        print(result.stdout.strip())

    if result.stderr.strip():
        print(result.stderr.strip())