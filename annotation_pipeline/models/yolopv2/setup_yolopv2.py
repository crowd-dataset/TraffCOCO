"""Bootstrap the official YOLOPv2 inference repository and pretrained weights.

The official repository is intentionally installed as an external dependency.
The pretrained TorchScript model is downloaded separately.

This setup avoids installing YOLOPv2's historical torch==1.7.0 /
torchvision==0.8.0 requirements into the TraffCOCO environment. The official
demo loads the released model directly with torch.jit.load(), so this adapter
uses that model format and implements the small amount of required postprocess
logic locally.
"""

from __future__ import annotations

import os
import subprocess
import urllib.request
from pathlib import Path


OFFICIAL_REPO_URL = "https://github.com/CAIC-AD/YOLOPv2.git"
OFFICIAL_WEIGHTS_URL = (
    "https://github.com/CAIC-AD/YOLOPv2/releases/download/"
    "V0.0.1/yolopv2.pt"
)


def project_root() -> Path:
    # setup_yolopv2.py lives at:
    # annotation_pipeline/models/yolopv2/setup_yolopv2.py
    return Path(__file__).resolve().parents[3]


def model_root() -> Path:
    return project_root() / "annotation_pipeline" / "models" / "yolopv2"


def repo_root() -> Path:
    return model_root() / "yolopv2_repo"


def weights_path() -> Path:
    return model_root() / "yolopv2.pt"


def ensure_repo() -> Path:
    repo = repo_root()
    repo.parent.mkdir(parents=True, exist_ok=True)

    if (repo / ".git").exists() or (repo / "demo.py").is_file():
        return repo

    subprocess.run(
        ["git", "clone", "--depth", "1", OFFICIAL_REPO_URL, str(repo)],
        check=True,
    )
    return repo


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                print(
                    f"Downloading YOLOPv2 weights: "
                    f"{downloaded / 1024 / 1024:.1f}/"
                    f"{total / 1024 / 1024:.1f} MiB"
                )

    partial.replace(destination)
    return destination


def ensure_weights() -> Path:
    custom = os.environ.get("YOLOPV2_WEIGHTS")
    if custom:
        path = Path(custom).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"YOLOPV2_WEIGHTS does not exist: {path}")
        return path

    return _download(OFFICIAL_WEIGHTS_URL, weights_path())


def ensure_assets() -> tuple[Path, Path]:
    repo = ensure_repo()
    weights = ensure_weights()
    return repo, weights


if __name__ == "__main__":
    repo, weights = ensure_assets()
    print(f"YOLOPv2 repository: {repo}")
    print(f"YOLOPv2 weights:     {weights}")
