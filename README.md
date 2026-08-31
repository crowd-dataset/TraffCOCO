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

Users can look into **mapping.csv** to see the available continents, countries, states, localities/cities, video IDs, time categories, and vehicle categories before updating the configuration.

**Step 9:** Use the active project entry point for the current annotation pipeline instead of the legacy frame extractor:
```bash
uv run python main.py --images 5
```

> The repository currently implements the VLM-first traffic-scene annotation pipeline in `main.py`. The older `frame-extractor.py` flow is not the active runtime path for this codebase.

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
```

## Current TraffCOCO pipeline architecture

The repository in its current state is not a database or frame-extraction project in the classic sense. It is a traffic-scene annotation pipeline built around a staged VLM + ontology + grounding workflow.

The active runtime flow defined in `main.py` is:

```text
Random Frame Download (if enabled)
        │
        ▼
Image Discovery
        │
        ▼
Scene Understanding
        │
        ▼
Pipeline Cache
        │
        ▼
Ontology Reasoning
        │
        ▼
Pipeline Cache
        │
        ▼
Prompt Builder
        │
        ▼
Locate Anything grounding
        │
        ▼
Annotation / ID reconciliation
        │
        ▼
Final visualization
```

### Stage responsibilities

- `SceneUnderstandingEngine` reads the configured image set and performs object discovery using the selected vision-language model.
- `OntologyEngine` reads each discovered scene object from the pipeline cache, resolves ontology metadata, and stores the canonical `class_id`, `class_name`, and `grounding_prompt`.
- `PromptBuilder` constructs the Locate Anything category prompt from the ontology-grounded labels rather than reconstructing categories from noisy scene output.
- `LocateAnythingEngine` runs the grounding model and saves raw, parsed, and visualization outputs to the configured outputs folders.
- `AnnotationEngine` consumes the cached scene + ontology + grounding data and produces the final per-object detections used for visualization and downstream annotation work.

### Architectural rules enforced by the current code

The current implementation makes these rules explicit:

- `ontology` is the source of truth for the canonical class identity.
- `grounding_prompt` is authoritative for the prompt sent to Locate Anything.
- `object_id=None` from Locate Anything is not treated as a hard failure; it is resolved only when the canonical scene cache provides a valid same-label object ID.
- The final annotation flow must prefer the canonical object ID already associated with the object in the pipeline cache.
- Repeated valid bounding boxes for the same object are retained instead of being deduplicated away.

This is implemented in `annotation_pipeline/pipeline/annotation.py`, which merges cache-grounding results from per-object entries and unmatched detections, validates bounding boxes, and resolves final object IDs before drawing the annotation overlay.

### Runtime configuration

The active runtime is driven by `default.config` and loaded via `annotation_pipeline/configs/settings.py`. Key flags include:

- `download_random_frames`
- `run_scene_understanding`
- `run_ontology_reasoning`
- `run_prompt_builder`
- `run_grounding`
- `run_segmentation`
- `run_annotation`
- `save_intermediate_cache`
- `save_pipeline_cache`
- `save_raw_outputs`
- `save_visualizations`

The project also uses the `uv` lockfile and `pyproject.toml` as the canonical dependency basis for the Python environment.

### Typical execution

From the repository root:

```bash
uv sync --frozen
source .venv/bin/activate
uv run python main.py --images 5
```

This reads the random frames directory, builds the scene-understanding cache, runs ontology reasoning, grounds objects with Locate Anything, and writes the final visualized annotation outputs under the configured output folders.

### Output structure

The current project writes intermediate and final artifacts under `annotation_pipeline/outputs` and related subfolders, including:

- scene cache entries
- ontology cache entries
- pipeline cache JSON files
- grounding prompts and raw output logs
- final annotated visuals
- annotation outputs

The repository is therefore best understood as a structured annotation-generation pipeline rather than a single standalone data-extraction script.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
