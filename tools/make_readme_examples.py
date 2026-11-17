"""
Build the README example figure from a finished TraffCOCO run.

Writes, next to each other:

  figures/examples.png    original frame | annotated result, one row per example
  figures/examples.html   interactive version (Plotly embedded, so it renders
                          through htmlpreview): pick an example, zoom, and
                          hover a box to see its class, sign text/meaning and
                          scores

Usage:
  uv run python tools/make_readme_examples.py \
      --frames random_frames --outputs outputs --out figures

``--outputs`` is the run's outputs folder (FINAL/visualizations and
COCO_FORMAT/annotations/instances_final.json are read from it).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

# Frame stem -> caption. Locations and day/night are from mapping.csv.
EXAMPLES = [
    ("1J0i_X1Qk1I_2271", "Tokyo, Japan (day)"),
    ("iryEo-mkNJw_140", "Toyokawa, Aichi, Japan (night), Japanese guide sign"),
    ("4-RJNtuAL8U_80", "Bengaluru, India (day)"),
    ("ZXO-SvYjCxA_3375", "Amsterdam, Netherlands (day)"),
    ("1nC-7OsXgyM_2417", "Eindhoven, Netherlands (night)"),
    ("Uuiqz2gha7o_27", "Tilburg, Netherlands (day): surfaces only, grounding found no objects in this 640x360 frame"),
]

PANEL_W, PANEL_H = 640, 360
GAP, CAPTION_H, MARGIN = 12, 30, 16


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _panel(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((PANEL_W, PANEL_H), Image.LANCZOS)


def _data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def make_png(examples, frames: Path, finals: Path, out: Path) -> None:
    rows = len(examples)
    width = MARGIN * 2 + PANEL_W * 2 + GAP
    height = MARGIN + 34 + rows * (CAPTION_H + PANEL_H + GAP)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    head, caption = _font(20), _font(16)
    draw.text((MARGIN, MARGIN), "Original CROWD frame", fill="black", font=head)
    draw.text((MARGIN + PANEL_W + GAP, MARGIN), "TraffCOCO annotation", fill="black", font=head)

    y = MARGIN + 34
    for stem, text in examples:
        draw.text((MARGIN, y + 6), text, fill=(60, 60, 60), font=caption)
        y += CAPTION_H
        sheet.paste(_panel(frames / f"{stem}.png"), (MARGIN, y))
        sheet.paste(_panel(finals / f"{stem}_FINAL.png"), (MARGIN + PANEL_W + GAP, y))
        y += PANEL_H + GAP

    sheet.save(out)


def _hover(annotation: dict, category: str) -> str:
    lines = [f"<b>{category}</b>"]
    attributes = annotation.get("attributes") or {}
    for key in ("text", "language", "meaning"):
        if attributes.get(key):
            lines.append(f"{key}: {attributes[key]}")
    if annotation.get("alternative_labels"):
        lines.append("also proposed: " + ", ".join(annotation["alternative_labels"]))
    for key in ("ontology_score", "sam_score"):
        if annotation.get(key) is not None:
            lines.append(f"{key}: {annotation[key]:.2f}")
    return "<br>".join(lines)


def make_html(examples, frames: Path, finals: Path, coco_path: Path, out: Path) -> None:
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images = {Path(i["file_name"]).stem: i for i in coco["images"]}
    by_image: dict[int, list] = {}
    for annotation in coco["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)

    figure = go.Figure()
    groups = []

    for index, (stem, _) in enumerate(examples):
        first = len(figure.data)
        original = Image.open(frames / f"{stem}.png").convert("RGB")
        width, height = original.size
        final = Image.open(finals / f"{stem}_FINAL.png").convert("RGB").resize((width, height))
        visible = index == 0

        figure.add_trace(go.Image(source=_data_uri(original), x0=0, y0=0, dx=1, dy=1,
                                  xaxis="x", yaxis="y", hoverinfo="skip", visible=visible))
        figure.add_trace(go.Image(source=_data_uri(final), x0=0, y0=0, dx=1, dy=1,
                                  xaxis="x2", yaxis="y2", hoverinfo="skip", visible=visible))

        image = images.get(stem)
        for annotation in by_image.get(image["id"], []) if image else []:
            x, y, w, h = annotation["bbox"]
            figure.add_trace(go.Scatter(
                x=[x, x + w, x + w, x, x], y=[y, y, y + h, y + h, y],
                mode="lines", fill="toself", fillcolor="rgba(255,255,255,0.02)",
                line={"width": 0}, hoveron="fills", hoverinfo="text",
                text=_hover(annotation, categories[annotation["category_id"]]),
                xaxis="x2", yaxis="y2", showlegend=False, visible=visible,
            ))

        groups.append((first, len(figure.data)))

    total = len(figure.data)
    buttons = []
    for (stem, text), (start, end) in zip(examples, groups):
        flags = [start <= i < end for i in range(total)]
        buttons.append({"label": text, "method": "update", "args": [{"visible": flags}]})

    axis = {"showgrid": False, "zeroline": False, "showticklabels": False}
    figure.update_layout(
        title={"text": "TraffCOCO examples: original frame (left) and annotation (right). "
                       "Hover a box on the right for its labels.", "font": {"size": 15}},
        xaxis={**axis, "domain": [0, 0.49]},
        yaxis={**axis, "autorange": "reversed", "scaleanchor": "x"},
        xaxis2={**axis, "domain": [0.51, 1]},
        yaxis2={**axis, "autorange": "reversed", "scaleanchor": "x2"},
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0, "y": 1.12,
                      "xanchor": "left", "yanchor": "top"}],
        margin={"l": 10, "r": 10, "t": 110, "b": 10},
        height=560, plot_bgcolor="white", paper_bgcolor="white",
        hoverlabel={"align": "left"},
    )
    figure.write_html(out, include_plotlyjs=True, full_html=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    finals = args.outputs / "FINAL" / "visualizations"
    coco = args.outputs / "COCO_FORMAT" / "annotations" / "instances_final.json"

    make_png(EXAMPLES, args.frames, finals, args.out / "examples.png")
    make_html(EXAMPLES, args.frames, finals, coco, args.out / "examples.html")
    print("wrote", args.out / "examples.png", "and", args.out / "examples.html")


if __name__ == "__main__":
    main()
