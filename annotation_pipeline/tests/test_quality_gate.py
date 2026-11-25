import math

from tools.quality_gate import FEATURES, box_group, box_source, gate_key, image_features, passes, score


def test_box_source():
    assert box_source({"object_id": "det_f_3", "detector_confidence": 0.9}) == "added"
    assert box_source({"object_id": "obj_2", "detector_confidence": 0.9}) == "matched"
    assert box_source({"object_id": "obj_2"}) == "grnd"


def test_box_group():
    assert box_group("taxi", "vehicle") == "car"
    assert box_group("stop_line", "road_marking") == "stop_line"


def test_gate_key_pools_rare_groups():
    assert gate_key("taxi", "vehicle", "added", set()) == "car|added"
    assert gate_key("stop_line", "road_marking", "grnd", set()) == "stop_line|grnd"
    assert gate_key("stop_line", "road_marking", "grnd", {"stop_line|grnd"}) == "rare|grnd"


def test_passes():
    table = {"car|added": 0.6, "rare|grnd": None, "person|matched": 0.0}
    box = {"bbox": [0, 0, 40, 40], "detector_confidence": 0.7}
    assert passes(box, "car|added", table, 24)
    assert not passes({**box, "detector_confidence": 0.5}, "car|added", table, 24)
    assert not passes(box, "rare|grnd", table, 24)
    assert passes(box, "person|matched", table, 24)
    assert passes(box, "auto_rickshaw|grnd", table, 24)  # never measured: kept
    assert not passes({**box, "bbox": [0, 0, 40, 20]}, "person|matched", table, 24)


def test_image_features():
    names = {1: "taxi", 2: "pedestrian"}
    supers = {"taxi": "vehicle", "pedestrian": "road_user"}
    annotations = [
        {"id": 1, "category_id": 1, "bbox": [0, 0, 100, 50], "object_id": "det_x_0",
         "detector_confidence": 0.8, "source": "yolo11x", "sam_score": 0.9},
        {"id": 2, "category_id": 2, "bbox": [10, 10, 20, 40], "object_id": "obj_1"},
    ]
    rows = image_features(annotations, 200, 100, names, supers)
    assert all(len(r) == len(FEATURES) for r in rows)
    car = dict(zip(FEATURES, rows[0]))
    person = dict(zip(FEATURES, rows[1]))
    assert car["group_car"] == 1 and car["source_added"] == 1 and car["detector_yolo"] == 1
    assert car["detector_confidence"] == 0.8 and person["detector_confidence"] == -1
    assert math.isclose(car["log_width"], math.log(0.5))
    assert person["inside_other_group"] == 1.0 and car["contains_other_group"] == 1.0
    assert car["boxes_in_image"] == 2 and car["same_group_in_image"] == 0


def test_score():
    tree = [
        {"leaf": False, "feature": 0, "threshold": 0.5, "left": 1, "right": 2},
        {"leaf": True, "value": -1.0},
        {"leaf": True, "value": 1.0},
    ]
    model = {"baseline": 0.0, "trees": [tree]}
    assert math.isclose(score(model, [0.0]), 1 / (1 + math.e))
    assert math.isclose(score(model, [1.0]), 1 / (1 + math.exp(-1)))
