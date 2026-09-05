"""YOLOPv2 auxiliary road-perception integration."""

from .yolopv2_segmentation import (
    YOLOPv2SegmentationEngine,
    YOLOPV2_CLASSES,
    YOLOPV2_IMPORTED_CLASS_IDS,
    YOLOPV2_EXCLUDED_CLASS_IDS,
    YOLOPV2_EXCLUDED_CLASSES,
    YOLOPV2_GENERIC_CLASSES,
)

__all__ = [
    "YOLOPv2SegmentationEngine",
    "YOLOPV2_CLASSES",
    "YOLOPV2_IMPORTED_CLASS_IDS",
    "YOLOPV2_EXCLUDED_CLASS_IDS",
    "YOLOPV2_EXCLUDED_CLASSES",
    "YOLOPV2_GENERIC_CLASSES",
]
