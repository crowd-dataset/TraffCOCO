"""
visualize_grounding.py

Visualization utilities for Locate Anything.

This module visualizes grounding predictions by drawing bounding boxes
and object labels onto the original image.

Pipeline
--------
Parsed Grounding Output
        │
        ▼
Bounding Box Renderer
        │
        ▼
Annotated Image
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
)

from custom_logger import CustomLogger

logger = CustomLogger(__name__)


class GroundingVisualizer:
    """
    Visualize Locate Anything predictions.
    """

    def __init__(
        self,
        line_width: int = 3,
        font_size: int = 18,
        seed: int = 42,
    ) -> None:

        self.line_width = line_width

        self.font_size = font_size

        self.random = random.Random(seed)

        # ----------------------------------------------------------
        # Unique color assigned to every object class
        # ----------------------------------------------------------

        self.class_colors: dict[str, str] = {}

        try:

            self.font = ImageFont.truetype(
                "arial.ttf",
                self.font_size,
            )

        except Exception:

            self.font = ImageFont.load_default()

        logger.info(
            "Initialized GroundingVisualizer."
        )

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def visualize(
        self,
        image: Image.Image,
        detections: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        """
        Draw valid Locate Anything detections on an image and save it.

        Invalid, malformed, non-finite, reversed, out-of-bounds, or
        degenerate bounding boxes are handled safely by _draw_detection().
        """

        # Work on a copy so visualization does not modify the original
        # image object used elsewhere in the pipeline.
        image = image.copy()

        draw = ImageDraw.Draw(image)

        image_width, image_height = image.size

        logger.info(
            "Visualizing {} grounding detection(s).",
            len(detections),
        )

        for detection in detections:

            self._draw_detection(
                draw=draw,
                detection=detection,
                image_width=image_width,
                image_height=image_height,
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image.save(
            output_path,
        )

        logger.info(
            "Saved visualization to '{}'.",
            output_path,
        )

    # ----------------------------------------------------------
    # Drawing
    # ----------------------------------------------------------

    def _draw_detection(
        self,
        draw: ImageDraw.ImageDraw,
        detection: dict[str, Any],
        image_width: int,
        image_height: int,
    ) -> None:
        """
        Draw one Locate Anything detection safely.

        Bounding boxes are:
        - validated
        - converted to numeric coordinates
        - checked for finite values
        - reordered if necessary
        - clamped to image boundaries
        - rejected if degenerate
        """

        bbox = detection.get(
            "bbox",
        )

        if bbox is None:

            logger.warning(
                "Skipping detection without bounding box."
            )

            return

        # ----------------------------------------------------------
        # Validate bbox structure
        # ----------------------------------------------------------

        if not isinstance(
            bbox,
            (list, tuple),
        ) or len(bbox) != 4:

            logger.warning(
                "Skipping detection with invalid bounding box: {}",
                bbox,
            )

            return

        # ----------------------------------------------------------
        # Convert coordinates to floats
        # ----------------------------------------------------------

        try:

            x1, y1, x2, y2 = (
                float(value)
                for value in bbox
            )

        except (
            TypeError,
            ValueError,
        ):

            logger.warning(
                "Skipping detection with non-numeric "
                "bounding box: {}",
                bbox,
            )

            return

        # ----------------------------------------------------------
        # Reject NaN / infinite coordinates
        # ----------------------------------------------------------

        import math

        if not all(
            math.isfinite(value)
            for value in (
                x1,
                y1,
                x2,
                y2,
            )
        ):

            logger.warning(
                "Skipping detection with non-finite "
                "bounding box: {}",
                bbox,
            )

            return

        # ----------------------------------------------------------
        # Normalize coordinate ordering
        # ----------------------------------------------------------

        x1, x2 = sorted(
            (
                x1,
                x2,
            )
        )

        y1, y2 = sorted(
            (
                y1,
                y2,
            )
        )

        # ----------------------------------------------------------
        # Clamp to image boundaries
        # ----------------------------------------------------------

        x1 = max(
            0.0,
            min(
                x1,
                float(image_width - 1),
            ),
        )

        x2 = max(
            0.0,
            min(
                x2,
                float(image_width - 1),
            ),
        )

        y1 = max(
            0.0,
            min(
                y1,
                float(image_height - 1),
            ),
        )

        y2 = max(
            0.0,
            min(
                y2,
                float(image_height - 1),
            ),
        )

        # ----------------------------------------------------------
        # Reject degenerate boxes BEFORE integer conversion
        # ----------------------------------------------------------

        if x2 <= x1 or y2 <= y1:

            logger.warning(
                "Skipping degenerate bounding box: {}",
                bbox,
            )

            return

        # ----------------------------------------------------------
        # Convert to integer pixel coordinates
        # ----------------------------------------------------------

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        # Rounding can collapse a very small box.
        if x2 <= x1 or y2 <= y1:

            logger.warning(
                "Skipping bounding box collapsed after "
                "integer conversion: {}",
                bbox,
            )

            return

        # ----------------------------------------------------------
        # Object information
        # ----------------------------------------------------------

        object_id = detection.get(
            "object_id",
            "?",
        )

        object_name = detection.get(
            "object_name",
            "Unknown",
        )

        object_name = str(
            object_name
        )

        color = self._get_color(
            object_name,
        )

        label = (
            f"ID:{object_id} | {object_name}"
        )

        # ----------------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------------

        draw.rectangle(
            [
                (x1, y1),
                (x2, y2),
            ],
            outline=color,
            width=self.line_width,
        )

        # ----------------------------------------------------------
        # Calculate label dimensions
        # ----------------------------------------------------------

        left, top, right, bottom = draw.textbbox(
            (x1, y1),
            label,
            font=self.font,
        )

        padding = 3

        label_left = left - padding
        label_top = top - padding
        label_right = right + padding
        label_bottom = bottom + padding

        # ----------------------------------------------------------
        # Keep label background inside image
        # ----------------------------------------------------------

        label_left = max(
            0,
            label_left,
        )

        label_top = max(
            0,
            label_top,
        )

        label_right = min(
            image_width - 1,
            label_right,
        )

        label_bottom = min(
            image_height - 1,
            label_bottom,
        )

        # ----------------------------------------------------------
        # Draw label background and text
        # ----------------------------------------------------------

        if (
            label_right >= label_left
            and label_bottom >= label_top
        ):

            draw.rectangle(
                [
                    (
                        label_left,
                        label_top,
                    ),
                    (
                        label_right,
                        label_bottom,
                    ),
                ],
                fill=color,
            )

            text_x = max(
                0,
                label_left + padding,
            )

            text_y = max(
                0,
                label_top + padding,
            )

            draw.text(
                (
                    text_x,
                    text_y,
                ),
                label,
                fill="white",
                font=self.font,
            )
    # ----------------------------------------------------------
    # Colors
    # ----------------------------------------------------------

    def _get_color(
        self,
        object_name: str,
    ) -> str:
        """
        Return a unique, consistent color for every object class.

        The same object label always receives the same color.
        """

        if object_name not in self.class_colors:

            self.class_colors[
                object_name
            ] = self._generate_color()

        return self.class_colors[
            object_name
        ]

    def _generate_color(
        self,
    ) -> str:
        """
        Generate a bright random RGB color.

        Bright colors are easier to distinguish on road scenes.
        """

        r = self.random.randint(
            80,
            255,
        )

        g = self.random.randint(
            80,
            255,
        )

        b = self.random.randint(
            80,
            255,
        )

        return f"#{r:02x}{g:02x}{b:02x}"