#!/usr/bin/env python3

from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request


def find_project_root() -> Path:
    """
    Find the TraffCOCO project root without assuming a Linux/Windows path
    layout beyond the repository structure.
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

# Keep this identical to sam2_segmenter.py.
MODEL_DIR = (
    ROOT
    / "annotation_pipeline"
    / "models"
    / "segmentation"
)

REPO = MODEL_DIR / "sam2_repo"
CHECKPOINT = MODEL_DIR / "sam2.1_hiera_tiny.pt"

REPO_URL = "https://github.com/facebookresearch/sam2.git"

CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/"
    "092824/sam2.1_hiera_tiny.pt"
)


# ----------------------------------------------------------------------
# Executables
# ----------------------------------------------------------------------

uv = shutil.which("uv")

if not uv:
    raise SystemExit(
        "uv was not found in PATH."
    )


if sys.platform == "win32":
    candidates = (
        VENV / "Scripts" / "python.exe",
    )
else:
    candidates = (
        VENV / "bin" / "python3",
        VENV / "bin" / "python",
    )


python = next(
    (
        path
        for path in candidates
        if path.exists()
    ),
    None,
)


if python is None:
    raise SystemExit(
        f"Project venv not found at {VENV}. "
        f"Expected one of: "
        f"{', '.join(str(path) for path in candidates)}. "
        "Run `uv venv` first."
    )


# ----------------------------------------------------------------------
# Directories
# ----------------------------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ----------------------------------------------------------------------
# SAM 2 repository
# ----------------------------------------------------------------------

if not (REPO / "sam2").is_dir():
    print(
        f"Cloning SAM 2 repository into:\n"
        f"  {REPO}"
    )

    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            REPO_URL,
            str(REPO),
        ],
        check=True,
    )


# ----------------------------------------------------------------------
# Install SAM 2 into the TraffCOCO project venv
# ----------------------------------------------------------------------

print(
    f"Installing SAM 2 into TraffCOCO venv:\n"
    f"  {python}"
)

subprocess.run(
    [
        uv,
        "pip",
        "install",
        "--python",
        str(python),
        "-e",
        str(REPO),
    ],
    check=True,
    cwd=str(ROOT),
)


# ----------------------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------------------

if (
    not CHECKPOINT.exists()
    or CHECKPOINT.stat().st_size <= 100 * 1024 * 1024
):
    tmp = CHECKPOINT.with_name(
        CHECKPOINT.name + ".download"
    )

    if tmp.exists():
        tmp.unlink()

    print(
        f"Downloading SAM 2.1 Hiera Tiny checkpoint to:\n"
        f"  {CHECKPOINT}"
    )

    urllib.request.urlretrieve(
        CHECKPOINT_URL,
        tmp,
    )

    tmp.replace(CHECKPOINT)


# ----------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------

print()
print("SAM 2.1 setup complete.")
print("Project root:", ROOT)
print("Venv:", python)
print("Repo:", REPO)
print("Checkpoint:", CHECKPOINT)