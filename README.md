# TraffCOCO

## Getting started
[![Python Version](https://img.shields.io/badge/python-3.12.13-blue.svg)](https://www.python.org/downloads/release/python-31213/)
[![Package Manager: uv](https://img.shields.io/badge/package%20manager-uv-green)](https://docs.astral.sh/uv/)

Tested with **Python 3.12.13** and the [`uv`](https://docs.astral.sh/uv/) package manager.
Follow these steps to set up the project.

**Step 1:** Install `uv`. `uv` is a fast Python package and environment manager. Install it using one of the following methods:

**macOS / Linux (bash/zsh):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Alternative (if you already have Python and pip):**
```bash
pip install uv
```

**Step 2:** Fix permissions (if needed):

Sometimes `uv` needs to create a folder under `~/.local/share/uv/python` (macOS/Linux) or `%LOCALAPPDATA%\uv\python` (Windows).
If this folder was created by another tool (e.g. `sudo`), you may see an error like:
```lua
error: failed to create directory ... Permission denied (os error 13)
```

To fix it, ensure you own the directory:

### macOS / Linux
```bash
mkdir -p ~/.local/share/uv
chown -R "$(id -un)":"$(id -gn)" ~/.local/share/uv
chmod -R u+rwX ~/.local/share/uv
```

### Windows
```powershell
# Create directory if it doesn't exist
New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\uv"

# Ensure you (the current user) own it
# (usually not needed, but if permissions are broken)
icacls "$env:LOCALAPPDATA\uv" /grant "$($env:UserName):(OI)(CI)F"
```

**Step 3:** After installing, verify:
```bash
uv --version
```

**Step 4:** Clone the repository:
```command line
git clone https://github.com/crowd-dataset/crowd-city
cd crowd-city
```

**Step 5:** Ensure correct Python version. If you don’t already have Python 3.12.13 installed, let `uv` fetch it:
```command line
uv python install 3.12.13
```
The repo should contain a .python-version file so `uv` will automatically use this version.

**Step 6:** Create and sync the virtual environment. This will create **.venv** in the project folder and install dependencies exactly as locked in **uv.lock**:
```command line
uv sync --frozen
```

**Step 7:** Activate the virtual environment:

**macOS / Linux (bash/zsh):**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**
```bat
.\.venv\Scripts\activate.bat
```

**Step 8:** Ensure that dataset are present. Place required datasets (including **mapping.csv**) into the **data/** directory:


**Step 9:** Run the code for extracting the frame:
```command line
python3 frame-extractor.py


### Configuration of project

Configuration of the project needs to be defined in `config`. Please use the `default.config` file for the required structure of the file. If no custom config file is provided, `default.config` is used. The config file has the following parameters:

- **`frames`**: Directory where the extracted frame data is stored.
- **`videos`**: Directory containing the videos used to generate the data.
- **`mapping`**: CSV file containing the mapping information used by the project.
- **`interval_seconds`**: Time interval, in seconds, used while extracting frames from videos.
- **`base_url`**: Base URL of the FTP or HTTP file server where the frame data is hosted.
- **`delete_downloaded_videos`**: Boolean flag to decide whether downloaded videos should be deleted after processing.
- **`logger_level`**: Logging level for the project, for example `info`, `debug`, or `error`.
- **`num_images`**: Number of random frame images to sample.
- **`local_output_root`**: Local directory where the sampled random frames will be saved.
- **`save_random_frames_csv`**: Boolean flag to decide whether a CSV file should be generated with information about the sampled images.
- **`DAY_NIGHTS`**: List of allowed time categories to sample from, for example `Day` and `Night`. An empty list means all available values are considered.
- **`VEHICLES`**: List of vehicle categories to sample from, for example `Car`, `Bus`, and `Truck`. An empty list means all available vehicle types are considered.
- **`VIDEO_IDS`**: List of specific video IDs to sample from. An empty list means all available video IDs are considered.
- **`LOCATION_TREE`**: Nested location filter used to select frames by continent, country, state, and city. If the state is missing in the dataset, use `unknown` as the state name.

Example:

```json
"LOCATION_TREE": {
  "Asia": {
    "India": {
      "DL": ["New Delhi", "Old Delhi"],
      "KA": ["Bengaluru"]
    },
    "Japan": {
      "Tokyo": ["Tokyo"]
    }
  },
  "Europe": {
    "Netherlands": {
      "unknown": ["Eindhoven", "Tilburg", "Amsterdam"]
    },
    "Germany": {
      "unknown": ["München"]
    }
  }
}

## License
This project is licensed under the MIT License - see the LICENSE file for details.