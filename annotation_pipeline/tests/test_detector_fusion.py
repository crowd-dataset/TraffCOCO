from annotation_pipeline.pipeline.detector_fusion import fuse_proposals, fusion_group

GROUPS = {
    "car": "vehicle",
    "taxi": "vehicle",
    "pedestrian": "road_user",
    "traffic_light_red": "traffic_signal",
    "pedestrian_signal": "traffic_signal",
    "stop_sign": "traffic_sign",
    "unreadable_traffic_sign": "traffic_sign",
    "streetlight": "road_infrastructure",
}
IDS = {"car": "V001", "unreadable_traffic_sign": "S001"}


def test_groups():
    assert fusion_group("taxi", "vehicle") == "car"
    assert fusion_group("traffic_light_red", "traffic_signal") == "traffic_light"
    assert fusion_group("pedestrian_signal", "traffic_signal") is None
    assert fusion_group("stop_sign", "traffic_sign") == "traffic_sign"
    assert fusion_group("streetlight", "road_infrastructure") is None


def test_match_keeps_label_and_takes_detector_box():
    detections = [{"class_name": "taxi", "bbox": [0, 0, 100, 100]}]
    proposals = [{"class_name": "car", "bbox": [2, 2, 100, 100], "detector": "yolo11x", "detector_confidence": 0.9}]
    fused, counts = fuse_proposals(detections, proposals, GROUPS, IDS)
    assert counts["matched"] == 1 and len(fused) == 1
    assert fused[0]["class_name"] == "taxi"
    assert fused[0]["bbox"] == [2, 2, 100, 100]
    assert fused[0]["grounding_bbox"] == [0, 0, 100, 100]


def test_unmatched_proposal_added_and_grounding_box_dropped():
    detections = [
        {"class_name": "car", "bbox": [0, 0, 50, 50]},
        {"class_name": "streetlight", "bbox": [500, 0, 520, 200]},
    ]
    proposals = [{"class_name": "unreadable_traffic_sign", "bbox": [300, 10, 330, 40]},
                 {"class_name": "car", "bbox": [200, 200, 300, 260]}]
    fused, counts = fuse_proposals(detections, proposals, GROUPS, IDS, image_name="f.png")
    names = sorted(d["class_name"] for d in fused)
    assert names == ["car", "streetlight", "unreadable_traffic_sign"]
    assert counts == {"matched": 0, "added": 2, "dropped": 1, "kept_unmatched": 0, "in_billboard": 0,
                      "streetlight_no_lamp": 0}
    added = [d for d in fused if d.get("object_id", "").startswith("det_")]
    assert {d["class_id"] for d in added} == {"V001", "S001"}


def test_add_mode_keeps_unmatched_grounding_box():
    detections = [{"class_name": "car", "bbox": [0, 0, 50, 50]}]
    proposals = [{"class_name": "car", "bbox": [200, 200, 300, 260]}]
    fused, counts = fuse_proposals(detections, proposals, GROUPS, IDS, mode="add")
    assert len(fused) == 2 and counts["kept_unmatched"] == 1


def test_uncovered_group_untouched():
    detections = [{"class_name": "stop_sign", "bbox": [0, 0, 20, 20]}]
    fused, counts = fuse_proposals(detections, [{"class_name": "car", "bbox": [100, 100, 200, 200]}], GROUPS, IDS)
    assert any(d["class_name"] == "stop_sign" for d in fused)
    assert counts["dropped"] == 0


def _verify(candidates):
    return {"kind": "verify", "candidates": candidates, "partners": {}}


def test_conflict_check_keeps_sure_detector_box():
    from annotation_pipeline.pipeline.conflict_check import NOT_A_SIGN, decide

    groups = {**GROUPS, "zebra_crossing": "road_marking", "stop_line": "road_marking"}
    detections = [
        {"class_name": "unreadable_traffic_sign", "bbox": [0, 0, 20, 20],
         "detector": "mask2former_vistas_panoptic", "detector_confidence": 0.9},
        {"class_name": "unreadable_traffic_sign", "bbox": [50, 0, 70, 20],
         "detector": "mask2former_vistas_panoptic", "detector_confidence": 0.6},
        {"class_name": "unreadable_traffic_sign", "bbox": [100, 0, 120, 20]},
    ]
    conflicts = {i: _verify(["unreadable_traffic_sign", NOT_A_SIGN]) for i in range(3)}
    kept = decide(detections, conflicts, {0: NOT_A_SIGN, 1: NOT_A_SIGN, 2: NOT_A_SIGN}, IDS, groups)
    assert [d["bbox"] for d in kept] == [[0, 0, 20, 20]]
    assert kept[0]["conflict_check"]["action"] == "kept_detector_box"


def test_conflict_check_keeps_detector_group():
    from annotation_pipeline.pipeline.conflict_check import decide

    groups = {**GROUPS, "zebra_crossing": "road_marking", "stop_line": "road_marking",
              "traffic_light_green": "traffic_signal"}
    detections = [
        {"class_name": "zebra_crossing", "bbox": [0, 100, 300, 140], "detector": "mask2former_vistas_panoptic"},
        {"class_name": "traffic_light_red", "bbox": [0, 0, 10, 30], "detector": "yolo11x"},
        {"class_name": "zebra_crossing", "bbox": [400, 100, 700, 140]},
    ]
    conflicts = {
        0: _verify(["zebra_crossing", "stop_line"]),
        1: _verify(["traffic_light_red", "traffic_light_green"]),
        2: _verify(["zebra_crossing", "stop_line"]),
    }
    kept = decide(detections, conflicts, {0: "stop_line", 1: "traffic_light_green", 2: "stop_line"}, IDS, groups)
    assert [d["class_name"] for d in kept] == ["zebra_crossing", "traffic_light_green", "stop_line"]
    assert kept[0]["conflict_check"]["action"] == "kept_detector_group"


def test_person_on_billboard_dropped():
    detections = [{"class_name": "pedestrian", "bbox": [110, 20, 140, 90]}]
    proposals = [
        {"class_name": "pedestrian", "bbox": [110, 20, 140, 90], "detector": "yolo11x"},
        {"class_name": "pedestrian", "bbox": [400, 300, 430, 380], "detector": "yolo11x"},
        {"class_name": "billboard", "bbox": [100, 0, 300, 100], "detector_label": "Billboard"},
    ]
    groups = {**GROUPS, "billboard": "commercial_sign"}
    fused, counts = fuse_proposals(detections, proposals, groups, IDS, image_name="f.png")
    assert counts["in_billboard"] == 1
    assert sorted(d["class_name"] for d in fused) == ["billboard", "pedestrian"]


def test_conflict_check_lets_light_become_pedestrian_signal():
    from annotation_pipeline.pipeline.conflict_check import decide

    detections = [{"class_name": "traffic_light_red", "bbox": [0, 0, 10, 30], "detector": "yolo11x"}]
    conflicts = {0: _verify(["traffic_light_red", "pedestrian_signal"])}
    kept = decide(detections, conflicts, {0: "pedestrian_signal"}, IDS, GROUPS)
    assert kept[0]["class_name"] == "pedestrian_signal"


def test_streetlight_needs_a_lamp_head():
    lamp = {"class_name": "streetlight", "bbox": [500, 0, 520, 200]}
    pole_only = {"class_name": "streetlight", "bbox": [700, 100, 715, 200]}
    head_only = {"class_name": "streetlight", "bbox": [900, 0, 960, 30]}
    heads = [
        {"class_name": "streetlight", "detector_label": "Street Light", "bbox": [505, 5, 518, 12]},
        {"class_name": "streetlight", "detector_label": "Street Light", "bbox": [910, 5, 950, 20]},
        {"class_name": "streetlight", "detector_label": "Street Light", "bbox": [700, 20, 712, 28]},
    ]
    detections = [lamp, pole_only, head_only]
    fused, counts = fuse_proposals(detections, heads, GROUPS, IDS, check_lamps=True)
    assert [d["bbox"] for d in fused] == [lamp["bbox"]]
    assert counts["streetlight_no_lamp"] == 2 and counts["added"] == 0
    # Without lamp heads from the detector the check is off.
    fused, counts = fuse_proposals(detections, [], GROUPS, IDS)
    assert len(fused) == 3 and counts["streetlight_no_lamp"] == 0
