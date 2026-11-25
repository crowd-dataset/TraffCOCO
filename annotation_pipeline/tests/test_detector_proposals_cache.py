import json

import pytest

pytest.importorskip("torch")

from annotation_pipeline.models.detection.detector_proposals import DetectorProposalEngine  # noqa: E402


def test_cache_reused_only_for_the_same_classes(tmp_path, monkeypatch):
    engine = DetectorProposalEngine(tmp_path, device="cpu")
    runs = []

    def fake(paths):
        runs.append([p.name for p in paths])
        return {}

    monkeypatch.setattr(engine, "_run_yolo", fake)
    monkeypatch.setattr(engine, "_run_panoptic", fake)
    image = tmp_path / "a.png"
    (tmp_path / "a.json").write_text(json.dumps({"image": "a.png", "proposals": []}))
    engine.process_images([image])  # old cache without labels: run again
    assert runs == [["a.png"], ["a.png"]]
    engine.process_images([image])  # same classes: reused
    assert len(runs) == 2
    assert engine.checks_lamp_heads
