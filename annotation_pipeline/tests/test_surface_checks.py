import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

from annotation_pipeline.models.segmentation.surface_segmenter import (  # noqa: E402
    remove_surface_covered_markings,
    roadside_unsupported,
)

CLASSES = ["Road", "Sidewalk", "Curb", "Guard Rail", "Barrier", "Fence", "Wall", "Vegetation"]
ROAD, CURB, RAIL, FENCE, WALL, VEG = 1, 3, 4, 6, 7, 8


def frame(fill=0):
    return np.full((100, 200), fill, dtype=np.uint8)


def test_ground_area_boxes_dropped():
    detections = {"f.png": [{"class_name": "curb"}, {"class_name": "sidewalk"}, {"class_name": "bollard"}]}
    assert remove_surface_covered_markings(detections) == 2
    assert [d["class_name"] for d in detections["f.png"]] == ["bollard"]


def test_bollard_on_wall_or_road():
    m = frame()
    m[50:60, 10:20] = WALL
    m[50:60, 30:40] = ROAD
    assert roadside_unsupported("bollard", [10, 50, 20, 60], m, CLASSES)
    assert roadside_unsupported("bollard", [30, 50, 40, 60], m, CLASSES)
    assert roadside_unsupported("bollard", [60, 50, 70, 60], m, CLASSES) is None  # object pixels


def test_guardrail_needs_a_rail():
    m = frame(ROAD)
    m[40:60, 0:100] = VEG
    assert roadside_unsupported("guardrail", [0, 40, 100, 70], m, CLASSES)
    m[55:60, 0:100] = RAIL
    assert roadside_unsupported("guardrail", [0, 40, 100, 70], m, CLASSES) is None


def test_median_needs_barrier_or_curb():
    m = frame(ROAD)
    assert roadside_unsupported("road_median", [0, 60, 100, 80], m, CLASSES)
    m[70:80, 0:100] = CURB
    assert roadside_unsupported("road_median", [0, 60, 100, 80], m, CLASSES) is None


def test_flexible_post_shape():
    m = frame()
    assert roadside_unsupported("flexible_post", [10, 5, 15, 80], m, CLASSES)  # reaches the top: a pole
    assert roadside_unsupported("flexible_post", [10, 50, 50, 80], m, CLASSES)  # too wide
    assert roadside_unsupported("flexible_post", [10, 50, 15, 80], m, CLASSES) is None
    assert roadside_unsupported("car", [10, 5, 15, 80], m, CLASSES) is None
