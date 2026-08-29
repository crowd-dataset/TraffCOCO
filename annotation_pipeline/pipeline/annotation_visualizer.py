"""Visualization for final ontology-resolved annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


class AnnotationVisualizer:
    """Draw final labels on the ORIGINAL image coordinates."""

    def __init__(self) -> None:
        try:
            self.font = ImageFont.truetype(
                "DejaVuSans.ttf",
                14,
            )
        except OSError:
            self.font = ImageFont.load_default()

    def visualize(
        self,
        image_path: str | Path,
        detections: list[dict[str, Any]],
        output_path: str | Path,
    ) -> Path:
        image_path = Path(image_path)
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with Image.open(image_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)

            for detection in detections:
                bbox = detection.get("bbox")
                label = str(
                    detection.get(
                        "final_label",
                        "unknown",
                    )
                )
                object_id = detection.get(
                    "object_id"
                )

                if (
                    not isinstance(
                        bbox,
                        (list, tuple),
                    )
                    or len(bbox) != 4
                ):
                    continue

                x1, y1, x2, y2 = map(
                    float,
                    bbox,
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                draw.rectangle(
                    (x1, y1, x2, y2),
                    outline="red",
                    width=3,
                )

                text = (
                    f"ID:{object_id} | {label}"
                    if object_id is not None
                    else label
                )

                text_bbox = draw.textbbox(
                    (x1, y1),
                    text,
                    font=self.font,
                )

                draw.rectangle(
                    text_bbox,
                    fill="red",
                )

                draw.text(
                    (x1, y1),
                    text,
                    fill="white",
                    font=self.font,
                )

            image.save(output_path)

        return output_path
