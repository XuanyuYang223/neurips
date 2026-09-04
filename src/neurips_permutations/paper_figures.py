"""Generate the compact, paper-ready permutation figure set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Iterable, Sequence
from xml.sax.saxutils import escape


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
BLACK = "#202124"
GRAY = "#687078"
LIGHT_GRAY = "#D9DEE3"
PALE_GRAY = "#F4F6F8"
WHITE = "#FFFFFF"

TASK_COUNTS = (1, 2, 4, 8, 16)
FIGURE_FILES = (
    "figure1_generalization_signals.svg",
    "figure1_generalization_signals.png",
    "figure2_task_geometry.svg",
    "figure2_task_geometry.png",
    "figureS1_scaling_diagnostics.svg",
    "figureS1_scaling_diagnostics.png",
    "figureS2_category_linear_probes.svg",
    "figureS2_category_linear_probes.png",
    "figureS3_representation_transfer.svg",
    "figureS3_representation_transfer.png",
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class SvgCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.items: list[str] = []

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = BLACK,
        width: float = 1.0,
        dash: str | None = None,
        opacity: float = 1.0,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{width:.2f}" opacity="{opacity:.3f}"{dash_attr}/>'
        )

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float = 0.0,
    ) -> None:
        self.items.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}"/>'
        )

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        *,
        fill: str,
        stroke: str = WHITE,
        stroke_width: float = 1.0,
        opacity: float = 1.0,
    ) -> None:
        self.items.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: float = 11.0,
        color: str = BLACK,
        anchor: str = "start",
        bold: bool = False,
    ) -> None:
        weight = "600" if bold else "400"
        self.items.append(
            f'<text x="{x:.2f}" y="{y:.2f}" fill="{color}" font-size="{size:.2f}" '
            f'font-family="DejaVu Sans,Arial,sans-serif" font-weight="{weight}" '
            f'text-anchor="{anchor}">{escape(value)}</text>'
        )

    def finish(self) -> bytes:
        body = "\n".join(self.items)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img">\n'
            f'<rect width="100%" height="100%" fill="{WHITE}"/>\n{body}\n</svg>\n'
        ).encode("utf-8")


class PillowCanvas:
    SCALE = 2

    def __init__(self, width: int, height: int) -> None:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("PNG generation requires `pip install -e '.[figures]'`") from error
        self.Image = Image
        self.ImageDraw = ImageDraw
        self.ImageFont = ImageFont
        self.width = width
        self.height = height
        self.image = Image.new("RGB", (width * self.SCALE, height * self.SCALE), WHITE)
        self.draw = ImageDraw.Draw(self.image)
        self._fonts: dict[tuple[int, bool], Any] = {}

    def _p(self, value: float) -> int:
        return int(round(value * self.SCALE))

    def _font(self, size: float, bold: bool) -> Any:
        key = (self._p(size), bold)
        if key not in self._fonts:
            filename = (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                if bold
                else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            )
            self._fonts[key] = self.ImageFont.truetype(filename, key[0])
        return self._fonts[key]

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = BLACK,
        width: float = 1.0,
        dash: str | None = None,
        opacity: float = 1.0,
    ) -> None:
        del opacity
        points = (self._p(x1), self._p(y1), self._p(x2), self._p(y2))
        line_width = max(1, self._p(width))
        if not dash:
            self.draw.line(points, fill=color, width=line_width)
            return
        dash_length, gap_length = (float(value) for value in dash.split()[:2])
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return
        offset = 0.0
        while offset < length:
            end = min(offset + dash_length, length)
            ax, ay = x1 + dx * offset / length, y1 + dy * offset / length
            bx, by = x1 + dx * end / length, y1 + dy * end / length
            self.draw.line(
                (self._p(ax), self._p(ay), self._p(bx), self._p(by)),
                fill=color,
                width=line_width,
            )
            offset = end + gap_length

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float = 0.0,
    ) -> None:
        box = (self._p(x), self._p(y), self._p(x + width), self._p(y + height))
        self.draw.rectangle(
            box,
            fill=None if fill == "none" else fill,
            outline=None if stroke == "none" else stroke,
            width=max(1, self._p(stroke_width)) if stroke != "none" else 1,
        )

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        *,
        fill: str,
        stroke: str = WHITE,
        stroke_width: float = 1.0,
        opacity: float = 1.0,
    ) -> None:
        del opacity
        box = (
            self._p(x - radius),
            self._p(y - radius),
            self._p(x + radius),
            self._p(y + radius),
        )
        self.draw.ellipse(
            box,
            fill=fill,
            outline=stroke,
            width=max(1, self._p(stroke_width)),
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: float = 11.0,
        color: str = BLACK,
        anchor: str = "start",
        bold: bool = False,
    ) -> None:
        font = self._font(size, bold)
        px, py = self._p(x), self._p(y)
        box = self.draw.textbbox((0, 0), value, font=font)
        width = box[2] - box[0]
        if anchor == "middle":
            px -= width // 2
        elif anchor == "end":
            px -= width
        self.draw.text((px, py - (box[3] - box[1])), value, fill=color, font=font)

    def finish(self) -> bytes:
        payload = io.BytesIO()
        self.image.save(payload, format="PNG", optimize=True, dpi=(300, 300))
        return payload.getvalue()


def _map_y(value: float, y: float, height: float, minimum: float, maximum: float) -> float:
    return y + height - (value - minimum) * height / (maximum - minimum)


def _x_positions(x: float, width: float, count: int) -> list[float]:
    if count == 1:
        return [x + width / 2]
    return [x + width * index / (count - 1) for index in range(count)]


def _axes(
    canvas: Any,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    xlabels: Sequence[str],
    minimum: float,
    maximum: float,
    ticks: Sequence[float],
    tick_format: Any,
    metric_label: str,
    x_padding: float = 0.0,
) -> list[float]:
    canvas.text(x, y - 10, metric_label, size=10, color=GRAY)
    for tick in ticks:
        py = _map_y(tick, y, height, minimum, maximum)
        canvas.line(x, py, x + width, py, color=LIGHT_GRAY, width=0.8)
        canvas.text(x - 8, py + 4, tick_format(tick), size=9, color=GRAY, anchor="end")
    canvas.line(x, y, x, y + height, color=BLACK, width=1.0)
    canvas.line(x, y + height, x + width, y + height, color=BLACK, width=1.0)
    positions = _x_positions(x + x_padding, width - 2 * x_padding, len(xlabels))
    for px, label in zip(positions, xlabels, strict=True):
        canvas.text(px, y + height + 18, label, size=9, color=GRAY, anchor="middle")
    return positions


def _error_bar(
    canvas: Any,
    x: float,
    mean: float,
    sd: float,
    *,
    y: float,
    height: float,
    minimum: float,
    maximum: float,
    color: str,
) -> None:
    low = max(minimum, mean - sd)
    high = min(maximum, mean + sd)
    py_low = _map_y(low, y, height, minimum, maximum)
    py_high = _map_y(high, y, height, minimum, maximum)
    canvas.line(x, py_low, x, py_high, color=color, width=1.4)
    canvas.line(x - 4, py_low, x + 4, py_low, color=color, width=1.4)
    canvas.line(x - 4, py_high, x + 4, py_high, color=color, width=1.4)


def _series(
    canvas: Any,
    positions: Sequence[float],
    means: Sequence[float],
    sds: Sequence[float],
    *,
    y: float,
    height: float,
    minimum: float,
    maximum: float,
    color: str,
    width: float = 2.4,
    radius: float = 4.0,
) -> None:
    mapped = [_map_y(value, y, height, minimum, maximum) for value in means]
    for left, right in zip(range(len(positions) - 1), range(1, len(positions)), strict=True):
        canvas.line(
            positions[left],
            mapped[left],
            positions[right],
            mapped[right],
            color=color,
            width=width,
        )
    for px, mean, sd, py in zip(positions, means, sds, mapped, strict=True):
        _error_bar(
            canvas,
            px,
            mean,
            sd,
            y=y,
            height=height,
            minimum=minimum,
            maximum=maximum,
            color=color,
        )
        canvas.circle(px, py, radius, fill=color)


def _legend_line(canvas: Any, x: float, y: float, label: str, color: str) -> None:
    canvas.line(x, y, x + 18, y, color=color, width=2.4)
    canvas.circle(x + 9, y, 3, fill=color)
    canvas.text(x + 24, y + 4, label, size=9, color=GRAY)


def _blend_hex(left: str, right: str, fraction: float) -> str:
    """Linearly blend two hexadecimal colors for deterministic heatmaps."""

    fraction = min(1.0, max(0.0, fraction))
    left_rgb = tuple(int(left[index : index + 2], 16) for index in (1, 3, 5))
    right_rgb = tuple(int(right[index : index + 2], 16) for index in (1, 3, 5))
    values = tuple(
        round(a + fraction * (b - a)) for a, b in zip(left_rgb, right_rgb, strict=True)
    )
    return "#" + "".join(f"{value:02X}" for value in values)


def _transfer_color(value: float) -> str:
    if value >= 0.0:
        return _blend_hex(PALE_GRAY, BLUE, value / 0.8)
    return _blend_hex(PALE_GRAY, ORANGE, -value / 0.1)


def _figure1(canvas: Any, repository: Path) -> None:
    behavior = {
        int(row["trained_task_count"]): row
        for row in _rows(
            repository / "results/property32-zero-overlap/replicates/behavior_summary.csv"
        )
    }
    probe = {
        int(row["trained_task_count"]): row
        for row in _rows(
            repository
            / "results/property32-zero-overlap/linear-probing/opposite_pool_summary.csv"
        )
        if row["layer"] == "final_norm"
    }
    random_probe = next(
        row
        for row in _rows(
            repository
            / "results/property32-zero-overlap/linear-probing/random_baseline_summary.csv"
        )
        if row["layer"] == "final_norm"
    )
    lr_rows = _rows(
        repository
        / "results/property32-zero-overlap/fewshot/lr-sensitivity/matched_lr_summary.csv"
    )
    lr = {
        (float(row["learning_rate"]), int(row["base_trained_task_count"])): row
        for row in lr_rows
    }

    panel_xs = (18, 412, 806)
    plot_y, plot_h, plot_w = 76, 244, 292
    labels = tuple(str(value) for value in TASK_COUNTS)

    canvas.text(panel_xs[0], 25, "(a) Hard zero-shot execution", size=13, bold=True)
    x0 = panel_xs[0] + 55
    positions = _axes(
        canvas,
        x=x0,
        y=plot_y,
        width=plot_w,
        height=plot_h,
        xlabels=labels,
        minimum=0.0,
        maximum=0.4,
        ticks=(0.0, 0.1, 0.2, 0.3, 0.4),
        tick_format=lambda value: f"{100 * value:.0f}",
        metric_label="Opposite-pool exact accuracy (%)",
    )
    means = [float(behavior[k]["macro_sequence_accuracy_mean"]) for k in TASK_COUNTS]
    sds = [float(behavior[k]["macro_sequence_accuracy_sample_sd"]) for k in TASK_COUNTS]
    _series(
        canvas,
        positions,
        means,
        sds,
        y=plot_y,
        height=plot_h,
        minimum=0.0,
        maximum=0.4,
        color=BLUE,
    )
    majority = float(behavior[1]["macro_majority_baseline_sequence_accuracy_mean"])
    majority_y = _map_y(majority, plot_y, plot_h, 0.0, 0.4)
    canvas.line(x0, majority_y, x0 + plot_w, majority_y, color=ORANGE, width=1.5, dash="6 4")
    _legend_line(canvas, x0 + 3, 48, "model mean +/- SD", BLUE)
    canvas.line(x0 + 145, 48, x0 + 163, 48, color=ORANGE, width=1.5, dash="6 4")
    canvas.text(x0 + 169, 52, "majority baseline", size=9, color=GRAY)

    canvas.text(panel_xs[1], 25, "(b) Linear decodability", size=13, bold=True)
    x1 = panel_xs[1] + 55
    positions = _axes(
        canvas,
        x=x1,
        y=plot_y,
        width=plot_w,
        height=plot_h,
        xlabels=labels,
        minimum=0.0,
        maximum=0.4,
        ticks=(0.0, 0.1, 0.2, 0.3, 0.4),
        tick_format=lambda value: f"{value:.1f}",
        metric_label="Final-layer length-conditioned R²",
    )
    means = [float(probe[k]["length_conditioned_r2_mean"]) for k in TASK_COUNTS]
    sds = [float(probe[k]["length_conditioned_r2_sample_sd"]) for k in TASK_COUNTS]
    _series(
        canvas,
        positions,
        means,
        sds,
        y=plot_y,
        height=plot_h,
        minimum=0.0,
        maximum=0.4,
        color=GREEN,
    )
    random_value = float(random_probe["length_conditioned_r2_mean"])
    random_y = _map_y(random_value, plot_y, plot_h, 0.0, 0.4)
    canvas.line(x1, random_y, x1 + plot_w, random_y, color=GRAY, width=1.5, dash="6 4")
    _legend_line(canvas, x1 + 3, 48, "trained", GREEN)
    canvas.line(x1 + 105, 48, x1 + 123, 48, color=GRAY, width=1.5, dash="6 4")
    canvas.text(x1 + 129, 52, "random init", size=9, color=GRAY)

    canvas.text(panel_xs[2], 25, "(c) Twenty-shot transfer", size=13, bold=True)
    x2 = panel_xs[2] + 55
    positions = _axes(
        canvas,
        x=x2,
        y=plot_y,
        width=plot_w,
        height=plot_h,
        xlabels=labels,
        minimum=-0.16,
        maximum=0.22,
        ticks=(-0.1, 0.0, 0.1, 0.2),
        tick_format=lambda value: f"{100 * value:+.0f}",
        metric_label="Pretrained - random exact accuracy (pp)",
    )
    zero_y = _map_y(0.0, plot_y, plot_h, -0.16, 0.22)
    canvas.line(x2, zero_y, x2 + plot_w, zero_y, color=GRAY, width=1.2, dash="5 4")
    for rate, color in ((1e-5, PURPLE), (3e-4, ORANGE)):
        means = [float(lr[(rate, k)]["pretrained_minus_random_mean"]) for k in TASK_COUNTS]
        sds = [float(lr[(rate, k)]["pretrained_minus_random_sample_sd"]) for k in TASK_COUNTS]
        _series(
            canvas,
            positions,
            means,
            sds,
            y=plot_y,
            height=plot_h,
            minimum=-0.16,
            maximum=0.22,
            color=color,
        )
    _legend_line(canvas, x2 + 3, 48, "LR 1e-5", PURPLE)
    _legend_line(canvas, x2 + 115, 48, "LR 3e-4", ORANGE)

    for x in panel_xs:
        canvas.text(x + 200, 382, "Number of base-training tasks (k)", size=10, color=GRAY, anchor="middle")
    canvas.text(
        600,
        405,
        "Error bars are sample SD over three joint task-split/model-seed replicates.",
        size=9,
        color=GRAY,
        anchor="middle",
    )


def _figure2(canvas: Any, repository: Path) -> None:
    cka_replicates = _rows(
        repository
        / "results/property32-zero-overlap/subset-replicates/cka_replicates.csv"
    )
    cka_summary = {
        int(row["trained_task_count"]): row
        for row in _rows(
            repository
            / "results/property32-zero-overlap/subset-replicates/cka_summary.csv"
        )
    }
    specialist = {
        row["comparison"]: row
        for row in _rows(
            repository / "results/property-task-geometry/cka/specialist_group_summary.csv"
        )
    }
    symmetry_rows = _rows(
        repository / "results/property-task-geometry/cka/symmetry_summary.csv"
    )
    symmetry: dict[str, dict[str, float]] = {}
    for row in symmetry_rows:
        symmetry.setdefault(row["pair_id"], {})[row["condition"]] = float(
            row["final_layer_cka_mean"]
        )

    panel_xs = (18, 412, 806)
    plot_y, plot_h, plot_w = 76, 244, 292

    canvas.text(panel_xs[0], 25, "(a) Zero-overlap CKA", size=13, bold=True)
    x0 = panel_xs[0] + 55
    positions = _axes(
        canvas,
        x=x0,
        y=plot_y,
        width=plot_w,
        height=plot_h,
        xlabels=tuple(str(k) for k in TASK_COUNTS),
        minimum=0.0,
        maximum=1.0,
        ticks=(0.0, 0.25, 0.5, 0.75, 1.0),
        tick_format=lambda value: f"{value:.2g}",
        metric_label="Final-layer linear CKA",
        x_padding=35,
    )
    replicate_ids = sorted({row["replicate_id"] for row in cka_replicates})
    for replicate in replicate_ids:
        values = {
            int(row["trained_task_count"]): float(row["final_layer_linear_cka"])
            for row in cka_replicates
            if row["replicate_id"] == replicate
        }
        mapped = [_map_y(values[k], plot_y, plot_h, 0.0, 1.0) for k in TASK_COUNTS]
        for left in range(len(positions) - 1):
            canvas.line(
                positions[left],
                mapped[left],
                positions[left + 1],
                mapped[left + 1],
                color=LIGHT_GRAY,
                width=1.2,
            )
        for px, py in zip(positions, mapped, strict=True):
            canvas.circle(px, py, 2.8, fill=LIGHT_GRAY, stroke=LIGHT_GRAY)
    means = [float(cka_summary[k]["final_layer_linear_cka_mean"]) for k in TASK_COUNTS]
    sds = [float(cka_summary[k]["final_layer_linear_cka_sample_sd"]) for k in TASK_COUNTS]
    _series(
        canvas,
        positions,
        means,
        sds,
        y=plot_y,
        height=plot_h,
        minimum=0.0,
        maximum=1.0,
        color=BLUE,
    )
    _legend_line(canvas, x0 + 3, 48, "mean +/- SD", BLUE)
    canvas.line(x0 + 132, 48, x0 + 150, 48, color=LIGHT_GRAY, width=1.5)
    canvas.text(x0 + 156, 52, "replicates", size=9, color=GRAY)

    canvas.text(panel_xs[1], 25, "(b) Specialist task geometry", size=13, bold=True)
    x1 = panel_xs[1] + 55
    bar_labels = ("Same task", "Direct relation", "Other pairs")
    positions = _axes(
        canvas,
        x=x1,
        y=plot_y,
        width=plot_w,
        height=plot_h,
        xlabels=bar_labels,
        minimum=0.0,
        maximum=1.0,
        ticks=(0.0, 0.25, 0.5, 0.75, 1.0),
        tick_format=lambda value: f"{value:.2g}",
        metric_label="Final-layer linear CKA",
        x_padding=35,
    )
    keys = ("same_task", "direct_relation", "no_direct_relation")
    colors = (GRAY, GREEN, SKY)
    for px, key, color in zip(positions, keys, colors, strict=True):
        mean = float(specialist[key]["final_layer_cka_mean"])
        sd = float(specialist[key]["final_layer_cka_sample_sd"])
        py = _map_y(mean, plot_y, plot_h, 0.0, 1.0)
        canvas.rect(px - 25, py, 50, plot_y + plot_h - py, fill=color)
        _error_bar(
            canvas,
            px,
            mean,
            sd,
            y=plot_y,
            height=plot_h,
            minimum=0.0,
            maximum=1.0,
            color=BLACK,
        )
    canvas.text(x1 + plot_w / 2, 365, "Task-pair class", size=10, color=GRAY, anchor="middle")
    canvas.text(x1 + plot_w / 2, 386, "Direct vs other: permutation p = 0.015", size=9, color=GRAY, anchor="middle")

    canvas.text(panel_xs[2], 25, "(c) Symmetry-aligned inputs", size=13, bold=True)
    x2 = panel_xs[2] + 55
    conditions = ("identity", "wrong", "correct")
    positions = _axes(
        canvas,
        x=x2,
        y=plot_y,
        width=plot_w,
        height=plot_h,
        xlabels=("Identity", "Wrong", "Correct"),
        minimum=0.0,
        maximum=1.0,
        ticks=(0.0, 0.25, 0.5, 0.75, 1.0),
        tick_format=lambda value: f"{value:.2g}",
        metric_label="Relation-level final-layer CKA",
        x_padding=20,
    )
    for values in symmetry.values():
        mapped = [_map_y(values[c], plot_y, plot_h, 0.0, 1.0) for c in conditions]
        for index in range(2):
            canvas.line(
                positions[index],
                mapped[index],
                positions[index + 1],
                mapped[index + 1],
                color=LIGHT_GRAY,
                width=1.2,
            )
        for px, py in zip(positions, mapped, strict=True):
            canvas.circle(px, py, 2.6, fill=LIGHT_GRAY, stroke=LIGHT_GRAY)
    means = [statistics.fmean(values[c] for values in symmetry.values()) for c in conditions]
    mapped = [_map_y(value, plot_y, plot_h, 0.0, 1.0) for value in means]
    for index in range(2):
        canvas.line(
            positions[index],
            mapped[index],
            positions[index + 1],
            mapped[index + 1],
            color=PURPLE,
            width=3.0,
        )
    for px, py in zip(positions, mapped, strict=True):
        canvas.circle(px, py, 4.2, fill=PURPLE)
    canvas.text(x2 + plot_w / 2, 365, "Input alignment condition", size=10, color=GRAY, anchor="middle")
    canvas.text(x2 + plot_w / 2, 386, "Correct > both controls in 8/8 relations", size=9, color=GRAY, anchor="middle")
    canvas.text(
        600,
        435,
        "CKA measures representation alignment; it is not behavioral accuracy.",
        size=9,
        color=GRAY,
        anchor="middle",
    )


def _figure_s1(canvas: Any, repository: Path) -> None:
    rows = _rows(repository / "results/v3/scaling/k16/summary.csv")
    condition_order = ("baseline", "data10x_model1x", "data1x_model2x", "data10x_model2x")
    condition_labels = ("1x / 1x", "10x / 1x", "1x / 2x", "10x / 2x")
    indexed = {(row["condition"], row["architecture"]): row for row in rows}
    panel_xs = (30, 465)
    plot_y, plot_h, plot_w = 82, 238, 350
    canvas.text(450, 25, "Structured-holdout exact accuracy = 0% in all 24 endpoints", size=14, bold=True, anchor="middle")
    for panel, (metric, sd_metric, maximum, ticks, title) in enumerate(
        (
            (
                "structured_holdout_token_accuracy_mean",
                "structured_holdout_token_accuracy_sample_sd",
                0.4,
                (0.0, 0.1, 0.2, 0.3, 0.4),
                "(a) Teacher-forced token accuracy",
            ),
            (
                "structured_holdout_loss_mean",
                "structured_holdout_loss_sample_sd",
                16.0,
                (0.0, 4.0, 8.0, 12.0, 16.0),
                "(b) Answer-token loss (lower is better)",
            ),
        )
    ):
        x = panel_xs[panel]
        canvas.text(x, 52, title, size=13, bold=True)
        positions = _axes(
            canvas,
            x=x + 54,
            y=plot_y,
            width=plot_w,
            height=plot_h,
            xlabels=condition_labels,
            minimum=0.0,
            maximum=maximum,
            ticks=ticks,
            tick_format=(lambda value: f"{100 * value:.0f}" if maximum == 0.4 else f"{value:.0f}"),
            metric_label="Accuracy (%)" if maximum == 0.4 else "Task-macro NLL",
            x_padding=25,
        )
        bar_width = 27
        for px, condition in zip(positions, condition_order, strict=True):
            for offset, architecture, color in (
                (-bar_width / 2, "transformer", BLUE),
                (bar_width / 2, "mlp", ORANGE),
            ):
                row = indexed[(condition, architecture)]
                mean, sd = float(row[metric]), float(row[sd_metric])
                py = _map_y(mean, plot_y, plot_h, 0.0, maximum)
                canvas.rect(px + offset - bar_width / 2, py, bar_width, plot_y + plot_h - py, fill=color)
                _error_bar(
                    canvas,
                    px + offset,
                    mean,
                    sd,
                    y=plot_y,
                    height=plot_h,
                    minimum=0.0,
                    maximum=maximum,
                    color=BLACK,
                )
        canvas.text(x + 54 + plot_w / 2, 365, "Training exposure / model depth", size=10, color=GRAY, anchor="middle")
    canvas.rect(338, 386, 12, 12, fill=BLUE)
    canvas.text(356, 397, "Transformer", size=9, color=GRAY)
    canvas.rect(442, 386, 12, 12, fill=ORANGE)
    canvas.text(460, 397, "MLP", size=9, color=GRAY)


def _figure_s2(canvas: Any, repository: Path) -> None:
    rows = _rows(
        repository
        / "results/v3/linear-probing/category/paired_random_contrasts.csv"
    )
    indexed = {
        (row["architecture"], row["condition"], row["probe_task_family"]): row
        for row in rows
        if row["layer"] == "final_norm"
    }
    families = ("local", "positional", "cycle", "global_run")
    family_labels = ("Local", "Positional", "Cycle", "Global/run")
    conditions = (
        ("encoding_e4", "Encoding E4", BLUE),
        ("statistics_s4", "Statistics S4", GREEN),
        ("algebra_a4", "Algebra A4", ORANGE),
    )
    panel_xs = (30, 465)
    plot_y, plot_h, plot_w = 82, 238, 350
    minimum, maximum = -0.05, 0.35
    canvas.text(
        450,
        25,
        "Frozen category probes: trained minus random initialization",
        size=14,
        bold=True,
        anchor="middle",
    )
    for panel, architecture in enumerate(("transformer", "mlp")):
        x = panel_xs[panel]
        title = "Transformer" if architecture == "transformer" else "MLP"
        canvas.text(x, 52, f"({chr(ord('a') + panel)}) {title}", size=13, bold=True)
        positions = _axes(
            canvas,
            x=x + 54,
            y=plot_y,
            width=plot_w,
            height=plot_h,
            xlabels=family_labels,
            minimum=minimum,
            maximum=maximum,
            ticks=(-0.05, 0.0, 0.1, 0.2, 0.3),
            tick_format=lambda value: f"{value:+.2f}",
            metric_label="Paired delta length-conditioned R²",
            x_padding=25,
        )
        zero_y = _map_y(0.0, plot_y, plot_h, minimum, maximum)
        canvas.line(x + 54, zero_y, x + 54 + plot_w, zero_y, color=GRAY, width=1.1)
        bar_width = 18
        for px, family in zip(positions, families, strict=True):
            for offset, (condition, _label, color) in zip(
                (-22, 0, 22), conditions, strict=True
            ):
                row = indexed[(architecture, condition, family)]
                mean = float(row["length_conditioned_r2_delta_mean"])
                sd = float(row["length_conditioned_r2_delta_sample_sd"])
                py = _map_y(mean, plot_y, plot_h, minimum, maximum)
                canvas.rect(
                    px + offset - bar_width / 2,
                    min(py, zero_y),
                    bar_width,
                    abs(zero_y - py),
                    fill=color,
                )
                _error_bar(
                    canvas,
                    px + offset,
                    mean,
                    sd,
                    y=plot_y,
                    height=plot_h,
                    minimum=minimum,
                    maximum=maximum,
                    color=BLACK,
                )
        canvas.text(
            x + 54 + plot_w / 2,
            365,
            "Scalar probe target family",
            size=10,
            color=GRAY,
            anchor="middle",
        )
    legend_x = 270
    for index, (_condition, label, color) in enumerate(conditions):
        canvas.rect(legend_x + index * 145, 386, 12, 12, fill=color)
        canvas.text(legend_x + 18 + index * 145, 397, label, size=9, color=GRAY)


def _figure_s3(canvas: Any, repository: Path) -> None:
    rows = _rows(repository / "results/representation-transfer/CELL_SUMMARY.csv")
    indexed = {(row["representation"], row["task"]): row for row in rows}
    representations = ("one_line", "cycle", "lehmer", "inversion_vector")
    representation_labels = ("One-line", "Cycle", "Lehmer", "Inv. vector")
    tasks = (
        "length",
        "parity",
        "peaks",
        "exceedances",
        "fixed_points",
        "descents",
        "recoils",
        "lis_length",
    )
    task_labels = (
        "Length",
        "Parity",
        "Peaks",
        "Exceed.",
        "Fixed pts",
        "Descents",
        "Recoils",
        "LIS",
    )
    if set(indexed) != {(representation, task) for representation in representations for task in tasks}:
        raise ValueError("representation-transfer heatmap requires the complete 4 x 8 grid")

    canvas.text(
        450,
        28,
        "Cross-representation/task transfer above the majority baseline",
        size=14,
        bold=True,
        anchor="middle",
    )
    left, top = 142.0, 92.0
    cell_width, cell_height = 88.0, 54.0
    for column, label in enumerate(task_labels):
        canvas.text(
            left + (column + 0.5) * cell_width,
            top - 14,
            label,
            size=9,
            color=GRAY,
            anchor="middle",
        )
    for row_index, (representation, representation_label) in enumerate(
        zip(representations, representation_labels, strict=True)
    ):
        y = top + row_index * cell_height
        canvas.text(left - 12, y + 33, representation_label, size=10, color=GRAY, anchor="end")
        for column, task in enumerate(tasks):
            value = float(indexed[(representation, task)]["sequence_accuracy_minus_majority_mean"])
            trained = indexed[(representation, task)]["task_status"] == "seen"
            x = left + column * cell_width
            color = _transfer_color(value)
            canvas.rect(
                x,
                y,
                cell_width - 2,
                cell_height - 2,
                fill=color,
                stroke=WHITE,
                stroke_width=1.0,
            )
            text_color = WHITE if value >= 0.45 else BLACK
            suffix = "*" if trained else ""
            canvas.text(
                x + (cell_width - 2) / 2,
                y + 32,
                f"{100 * value:+.1f}{suffix}",
                size=10,
                color=text_color,
                anchor="middle",
                bold=trained,
            )

    legend_y = 338.0
    legend_values = (-0.1, 0.0, 0.2, 0.4, 0.6, 0.8)
    legend_x = 250.0
    for index, value in enumerate(legend_values):
        x = legend_x + index * 64
        canvas.rect(x, legend_y, 62, 14, fill=_transfer_color(value))
        canvas.text(x + 31, legend_y + 30, f"{100 * value:+.0f}", size=8, color=GRAY, anchor="middle")
    canvas.text(450, legend_y - 8, "Exact accuracy minus majority baseline (percentage points)", size=9, color=GRAY, anchor="middle")
    canvas.text(
        450,
        405,
        "* Trained cell. Values are means over three jointly trained Transformer seeds; the other 21 cells were held out from gradients.",
        size=9,
        color=GRAY,
        anchor="middle",
    )


def _render_pair(
    output: Path,
    stem: str,
    width: int,
    height: int,
    draw_figure: Any,
    repository: Path,
) -> None:
    svg = SvgCanvas(width, height)
    draw_figure(svg, repository)
    _atomic_bytes(output / f"{stem}.svg", svg.finish())
    png = PillowCanvas(width, height)
    draw_figure(png, repository)
    _atomic_bytes(output / f"{stem}.png", png.finish())


def generate(
    repository: Path,
    output: Path,
) -> dict[str, Any]:
    repository = repository.resolve()
    output = output.resolve()
    inputs = (
        "results/property32-zero-overlap/replicates/behavior_summary.csv",
        "results/property32-zero-overlap/subset-replicates/cka_replicates.csv",
        "results/property32-zero-overlap/subset-replicates/cka_summary.csv",
        "results/property32-zero-overlap/linear-probing/opposite_pool_summary.csv",
        "results/property32-zero-overlap/linear-probing/random_baseline_summary.csv",
        "results/property32-zero-overlap/fewshot/lr-sensitivity/matched_lr_summary.csv",
        "results/property-task-geometry/cka/specialist_group_summary.csv",
        "results/property-task-geometry/cka/symmetry_summary.csv",
        "results/v3/scaling/k16/summary.csv",
        "results/v3/linear-probing/category/paired_random_contrasts.csv",
        "results/representation-transfer/CELL_SUMMARY.csv",
    )
    for relative in inputs:
        if not (repository / relative).is_file():
            raise FileNotFoundError(repository / relative)
    output.mkdir(parents=True, exist_ok=True)
    _render_pair(output, "figure1_generalization_signals", 1200, 420, _figure1, repository)
    _render_pair(output, "figure2_task_geometry", 1200, 450, _figure2, repository)
    _render_pair(output, "figureS1_scaling_diagnostics", 900, 420, _figure_s1, repository)
    _render_pair(output, "figureS2_category_linear_probes", 900, 420, _figure_s2, repository)
    _render_pair(output, "figureS3_representation_transfer", 900, 420, _figure_s3, repository)
    manifest = {
        "format_version": "permutation-paper-figures/v1",
        "status": "completed",
        "main_text_figures": [
            "figure1_generalization_signals",
            "figure2_task_geometry",
        ],
        "supplementary_figures": [
            "figureS1_scaling_diagnostics",
            "figureS2_category_linear_probes",
            "figureS3_representation_transfer",
        ],
        "inputs": {relative: _sha256(repository / relative) for relative in inputs},
        "outputs": {
            name: {
                "sha256": _sha256(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in FIGURE_FILES
        },
    }
    _atomic_bytes(
        output / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = generate(args.repository, args.output_dir)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
