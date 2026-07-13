## TraffCOCO

## Implementation Plan

> Master design document for the automatic traffic annotation pipeline combining
> Vision Language Models (Gemma 4), a Traffic Knowledge Graph, semantic reasoning,
> Locate Anything grounding, and SAM2 segmentation.

---

## 1. Project Overview

This project implements a modular, research-quality annotation pipeline that:

1. Uses **Gemma 4** (multimodal VLM) for semantic scene understanding
2. Maps detected objects to a **Traffic Knowledge Graph** (1,270+ classes)
3. Constructs descriptive prompts for visual grounding
4. Uses **LocateAnything-3B** (NVIDIA) for bounding box localization
5. Uses **SAM2** for pixel-level mask refinement
6. Produces **COCO-format** annotation JSON with ontology class labels

Two parallel pipeline variants are planned:
- `vlm_first/` — VLM drives initial detection (current target)
- `sam_first/` — SAM drives initial segmentation (future)

---

## 2. Architecture Diagram

```
                         Image
                           │
                           ▼
                       Gemma 4
                 (Scene Understanding)
                           │
                           ▼
              Scene Understanding Cache
                           │
               ┌───────────┴────────────┐
               ▼                        ▼
      Knowledge Graph             Prompt Builder
       (Class Mapping)     (Description + Context)
                                        │
                                        ▼
                               Locate Anything
                                (Localization)
               └───────────┬────────────┘
                           ▼
                    Pipeline Cache
                           │
                           ▼
                      SAM / SAM2
                   (Mask Refinement)
                           │
                           ▼
              Annotation Post Processing
                           │
                           ▼
                      Visualization
                           │
                           ▼
                Final COCO Annotation JSON
```

---

## 3. Folder Structure

```
SAMPLE/                              # Project root
├── random_frames/                   # Input traffic images (99 PNG files)
├── annotation_pipeline/
│   ├── implementation_plan.md       # This file
│   ├── progress.md                  # Session-by-session progress tracker
│   │
│   ├── common/                      # Shared modules (both pipelines)
│   │   ├── __init__.py
│   │   ├── ontology/
│   │   │   ├── __init__.py
│   │   │   ├── load_ontology.py     # Ontology loader + search
│   │   │   ├── traffic_ontology_v2.json
│   │   │   ├── build_embeddings.py
│   │   │   ├── embedding_texts.json
│   │   │   └── faiss_index.bin
│   │   ├── prompts/
│   │   │   ├── __init__.py
│   │   │   ├── load_prompt.py       # Dynamic prompt template loader
│   │   │   └── scene_understanding_prompt.txt
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── image_utils.py       # Image loading, validation
│   │   │   └── logger.py            # Centralized logging
│   │   ├── configs/
│   │   │   ├── __init__.py
│   │   │   ├── settings.py          # Dataclass-based config
│   │   │   └── default.config # JSON configuration
│   │   └── cache/
│   │       ├── __init__.py
│   │       └── pipeline_cache.py    # Shared pipeline cache
│   │
│   ├── vlm_first/                   # VLM-first pipeline (current target)
│   │   ├── __init__.py
│   │   ├── main.py                  # Pipeline orchestrator
│   │   ├── pipeline/
│   │   │   ├── __init__.py
│   │   │   ├── scene_understanding.py
│   │   │   ├── ontology_reasoning.py
│   │   │   ├── prompt_builder.py
│   │   │   ├── locate_anything.py
│   │   │   ├── sam_refinement.py
│   │   │   ├── annotation_postprocess.py
│   │   │   └── visualize_results.py
│   │   └── outputs/                 # Per-image results
│   │
│   ├── sam_first/                   # SAM-first pipeline (future)
│   │   ├── pipeline/
│   │   ├── outputs/
│   │   └── main.py
│   │
│   └── experiments/
│
├── pyproject.toml                   # uv project definition
└── .python-version                  # Python 3.11
```

---

## 4. Pipeline Stages — Implementation Order

### Stage 0: Foundation Infrastructure
- [ ] Project configuration (`common/configs/`)
- [ ] Centralized logging (`common/utils/logger.py`)
- [ ] Image utilities (`common/utils/image_utils.py`)
- [ ] Prompt template loader (`common/prompts/`)
- [ ] Scene understanding prompt template
- [ ] All `__init__.py` files
- [ ] Update `pyproject.toml` (Python 3.11, dependencies)

### Stage 1: Gemma Scene Understanding
- [ ] Scene understanding module (`vlm_first/pipeline/scene_understanding.py`)
- [ ] Gemma 4 integration via `google-genai` SDK
- [ ] Per-image JSON output matching prescribed schema
- [ ] Scene Understanding Cache (JSON per image)
- [ ] Independent test function

### Stage 2: Knowledge Graph Reasoning
- [ ] Ontology loader (`common/ontology/load_ontology.py`)
- [ ] Ontology reasoning module (`vlm_first/pipeline/ontology_reasoning.py`)
- [ ] Attribute-weighted scoring for class matching
- [ ] Complete ontology entry preservation in pipeline cache
- [ ] Independent test function

### Stage 3: Prompt Builder
- [ ] Prompt builder module (`vlm_first/pipeline/prompt_builder.py`)
- [ ] Deterministic — NO LLM calls
- [ ] Descriptive prompts (not taxonomy names)
- [ ] Merged comma-separated string output
- [ ] Independent test function

### Stage 4: Pipeline Cache
- [ ] Pipeline cache module (`common/cache/pipeline_cache.py`)
- [ ] Object-ID-based progressive accumulation
- [ ] Merge Gemma + KG + LocateAnything + SAM stages
- [ ] Independent test function

### Stage 5: Locate Anything Integration
- [ ] Locate Anything module (`vlm_first/pipeline/locate_anything.py`)
- [ ] LocateAnything-3B via HuggingFace transformers
- [ ] Bounding box extraction + confidence
- [ ] Visualization + annotation JSON output
- [ ] Independent test function

### Stage 6: SAM/SAM2 Refinement
- [ ] SAM refinement module (`vlm_first/pipeline/sam_refinement.py`)
- [ ] SAM2ImagePredictor with box prompts
- [ ] Mask generation + storage
- [ ] Independent test function

### Stage 7: Annotation Post-Processing
- [ ] Post-processing module (`vlm_first/pipeline/annotation_postprocess.py`)
- [ ] Replace description labels → ontology class names
- [ ] Preserve bboxes, confidence, masks
- [ ] Independent test function

### Stage 8: Visualization
- [ ] Visualization module (`vlm_first/pipeline/visualize_results.py`)
- [ ] Overlay bboxes + masks + class labels
- [ ] Save to `vlm_first/outputs/`
- [ ] Independent test function

### Stage 9: Main Pipeline Orchestrator
- [ ] `vlm_first/main.py` — full orchestration
- [ ] Sequential stage execution
- [ ] Final COCO annotation JSON export
- [ ] End-to-end test

---

## 5. Model Dependencies

| Model | Source | Package | Purpose |
|-------|--------|---------|---------|
| Gemma 4 (12B/27B) | Google | `google-genai` | Scene understanding |
| LocateAnything-3B | NVIDIA/HuggingFace | `transformers` | Bounding box localization |
| SAM2 | Meta | `sam2` / `segment-anything-2` | Mask refinement |

### External Repositories
- LocateAnything: `nvidia/LocateAnything-3B` on HuggingFace
- SAM2: `facebookresearch/sam2` on GitHub

---

## 6. Configuration Schema

```yaml
# pipeline_config.yaml
project:
  name: "kgflm-traffic-annotation"
  version: "0.1.0"

paths:
  project_root: "."                              # Resolved at runtime
  random_frames: "random_frames"
  ontology: "annotation_pipeline/common/ontology/traffic_ontology_v2.json"
  vlm_outputs: "annotation_pipeline/vlm_first/outputs"
  scene_cache: "annotation_pipeline/vlm_first/outputs/scene_cache"
  pipeline_cache: "annotation_pipeline/vlm_first/outputs/pipeline_cache"

models:
  gemma:
    model_name: "gemma-4-27b-it"
    api_provider: "google-genai"               # or "local-vllm"
    temperature: 0.1
    max_output_tokens: 8192

  locate_anything:
    model_name: "nvidia/Eagle-X5-13B-Chat"
    device: "cuda"
    checkpoint: null                            # Auto-download from HF

  sam:
    model_name: "sam2.1_hiera_large"
    checkpoint: null                            # Path to sam2 checkpoint
    config: null                                # Path to sam2 config
    device: "cuda"

pipeline:
  confidence_threshold: 30
  max_objects_per_image: 50
  save_visualizations: true
  save_intermediate_cache: true
```

---

## 7. Shared Module Contracts

### Ontology Loader
- Input: Path to `traffic_ontology_v2.json`
- Output: Indexed ontology with fast lookup by `class_id`, `class_name`, keywords
- Methods: `get_class_by_id()`, `search_by_attributes()`, `get_all_classes()`

### Pipeline Cache
- Key: `(image_name, object_id)` tuple
- Progressive: Each stage adds its own key (`gemma`, `kg`, `locate_anything`, `sam`)
- Serializable: JSON export/import for checkpoint-resume

### Prompt Loader
- Input: Template filename (e.g., `scene_understanding_prompt.txt`)
- Output: String with optional variable substitution
- Location: `common/prompts/`

---

## 8. Key Design Decisions

1. **Directory naming**: The existing on-disk folder is `VLM FIRST` (space, uppercase).
   The code should reference it as-is to avoid breaking the existing structure.
   Python packages will use import-safe `__init__.py` inside the folder.

2. **Python version**: Spec says 3.11, existing `.python-version` says 3.12.
   → Will update to 3.11 per spec requirements.

3. **Ontology is read-only**: The ontology JSON is never modified at runtime.
   A Python class wraps it for efficient querying.

4. **No fuzzy matching**: Object IDs are used as strict keys throughout the cache.

5. **Gemma via API**: Use `google-genai` SDK for Gemma 4 access (simplest path).
   Local vLLM serving is an alternative (configurable).

---

## 9. Known Limitations

- LocateAnything-3B and SAM2 require GPU (CUDA) for inference
- Gemma 4 API requires a Google AI Studio API key
- Input images are assumed to be `.png` files from `random_frames/`
- No video support in this version
- Pipeline is single-image sequential (parallelism planned for future)

---

## 10. Future Improvements

- [ ] Batch processing with multiprocessing/async
- [ ] FAISS-based embedding search for ontology matching
- [ ] `sam_first/` pipeline variant
- [ ] Active learning feedback loop
- [ ] GPU memory optimization (model loading/unloading)
- [ ] COCO evaluation metrics integration
- [ ] Web-based annotation review interface

---

## 11. Research Notes

- The ontology contains **36,573 lines** with **1,270+ traffic object classes** spanning
  48 countries, 9 categories (Road Users, Vehicles, Traffic Signs, Traffic Signals,
  Road Markings, Infrastructure, Temporary Objects, Animals, Country-Specific).
- Each class includes: visual attributes, text info, installation context, relationships,
  semantic keywords, inference cues, distinctive features, negative cues, attribute
  importance weights, and detection priority ordering.
- The `embedding_text` field per class is pre-computed for FAISS indexing.
