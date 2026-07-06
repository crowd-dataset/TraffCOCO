"""
vlm_first — VLM-first traffic annotation pipeline.

Pipeline order:
    Gemma Scene Understanding → Knowledge Graph + Prompt Builder →
    Locate Anything → SAM2 → Post-Processing → Visualization → COCO JSON
"""
