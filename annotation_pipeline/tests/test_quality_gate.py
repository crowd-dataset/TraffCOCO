from tools.quality_gate import box_source, gate_key, passes


def test_box_source():
    assert box_source({"object_id": "det_f_3", "detector_confidence": 0.9}) == "added"
    assert box_source({"object_id": "obj_2", "detector_confidence": 0.9}) == "matched"
    assert box_source({"object_id": "obj_2"}) == "grnd"


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
