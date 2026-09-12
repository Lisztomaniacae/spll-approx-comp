"""Shared color palette for MNIST SPLL thesis visualizations.

The palette follows the Okabe-Ito / Color Universal Design family used in
scientific figures: hues stay distinguishable under common color-vision
variants, and plot code should still keep redundant markers/line styles where
possible.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import matplotlib as mpl
from matplotlib.legend import Legend


# Thesis export settings.  Figures are raster PNGs intended to be embedded at
# approximately A4 page width, so 300 dpi gives print-safe resolution.
FIGURE_DPI = 300
A4_PAGE_WIDTH_IN = 8.27
A4_PAGE_WIDTH_PX = int(round(FIGURE_DPI * A4_PAGE_WIDTH_IN))

# Thesis figures are embedded inside the TU Darmstadt report text block, not at
# full A4 paper width.  Generating directly for that narrower target keeps text
# and markers readable after LaTeX scales the PNG to \textwidth.
THESIS_TEXT_WIDTH_IN = 5.80
THESIS_SINGLE_PANEL_HEIGHT_IN = 3.30
THESIS_TWO_ROW_HEIGHT_IN = 5.30

THESIS_FONT_SIZE = 10.5
THESIS_LABEL_SIZE = 11.0
THESIS_TICK_SIZE = 9.5
THESIS_LEGEND_SIZE = 9.0
THESIS_PANEL_LABEL_SIZE = 11.5
THESIS_GRID_ALPHA = 0.20
THESIS_LINE_WIDTH = 1.65
THESIS_MARKER_SIZE = 4.5

LEGEND_BOX_FACE = "#F3F6FA"
LEGEND_BOX_EDGE = "#CBD6E2"
LEGEND_BOX_ALPHA = 0.96
LEGEND_BOX_LINEWIDTH = 0.9


# Core categorical hues.
BLUE = "#0072B2"
SKY_BLUE = "#56B4E9"
BLUISH_GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
REDDISH_PURPLE = "#CC79A7"
YELLOW = "#F0E442"
BLACK = "#000000"

# Neutral colors for guides and non-data encodings.
DARK_GREY = "#4D4D4D"
MID_GREY = "#7A7A7A"
LIGHT_GREY = "#E4E8EE"

# Stable model-accuracy constellation used in Pipeline I.
MODEL_ACCURACY_COLORS: Sequence[str] = (
    BLUE,          # 50%
    ORANGE,        # 70%
    BLUISH_GREEN,  # 90%
    REDDISH_PURPLE,
    SKY_BLUE,
    VERMILION,
    YELLOW,
    DARK_GREY,
)

# Training-trace constellation used in Pipeline II.
TRAINING_TRACE_COLORS: Dict[str, str] = {
    "exact": BLUE,
    "pure": VERMILION,
    "transfer": BLUISH_GREEN,
    "checkpoint": REDDISH_PURPLE,
}

# Fixed inference-mode constellation.  These are keyed by the numeric cutoff
# after formatting with ``:g`` so both 0.10 and 0.1 map to the same color.
CUTOFF_COLORS: Dict[str, str] = {
    "0.01": REDDISH_PURPLE,
    "0.05": BLUISH_GREEN,
    "0.1": VERMILION,
    "0.25": SKY_BLUE,
}
INFERENCE_MODE_COLORS: Dict[str, str] = {
    "exact": BLUE,
    **{f"cutoff_{key}": value for key, value in CUTOFF_COLORS.items()},
}

# Dual-axis bars use the same hue as the inference mode; the outer bar is a
# light fill with colored edge, and the inner bar is a darker fill.
DUAL_AXIS_BAR_COLORS: Dict[str, Dict[str, str]] = {
    "exact": {"base": BLUE, "outer": "#B8D8E9", "inner": "#005D92"},
    "cutoff_0.01": {"base": REDDISH_PURPLE, "outer": "#F1D9E6", "inner": "#A76389"},
    "cutoff_0.05": {"base": BLUISH_GREEN, "outer": "#B8E4D8", "inner": "#00825E"},
    "cutoff_0.1": {"base": VERMILION, "outer": "#F3D2B8", "inner": "#AF4D00"},
    "cutoff_0.25": {"base": SKY_BLUE, "outer": "#D0EAF9", "inner": "#4794BF"},
}

# Runtime-accuracy constellation.
TRADEOFF_COLORS: Dict[str, str] = {
    "speedup": BLUE,
    "accuracy": VERMILION,
    "score": REDDISH_PURPLE,
    "baseline": MID_GREY,
    "positive_zone": BLUISH_GREEN,
    "mass_marker": DARK_GREY,
    "ci_band": DARK_GREY,
}

# CVD-safer diverging palettes for sign-sensitive matrix plots.
SPEEDUP_DIVERGING = (VERMILION, "#F4EAD6", BLUISH_GREEN)
ACCURACY_DIVERGING = (VERMILION, "#F4EAD6", BLUISH_GREEN)


def apply_thesis_plot_style() -> None:
    """Apply one reader-facing Matplotlib style across both thesis pipelines."""

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.labelsize": THESIS_LABEL_SIZE,
            "axes.titlesize": THESIS_PANEL_LABEL_SIZE,
            "axes.titleweight": "regular",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "xtick.labelsize": THESIS_TICK_SIZE,
            "ytick.labelsize": THESIS_TICK_SIZE,
            "grid.color": "#d9d9d9",
            "grid.linestyle": "-",
            "grid.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "legend.fancybox": True,
            "legend.facecolor": LEGEND_BOX_FACE,
            "legend.edgecolor": LEGEND_BOX_EDGE,
            "legend.framealpha": LEGEND_BOX_ALPHA,
            "legend.fontsize": THESIS_LEGEND_SIZE,
            "font.size": THESIS_FONT_SIZE,
            "lines.linewidth": THESIS_LINE_WIDTH,
            "lines.markersize": THESIS_MARKER_SIZE,
        }
    )


def thesis_panel_figure_size(nrows: int) -> tuple[float, float]:
    """Return a compact figure size intended for thesis-width embedding."""

    return (
        THESIS_TEXT_WIDTH_IN,
        THESIS_SINGLE_PANEL_HEIGHT_IN if int(nrows) <= 1 else THESIS_TWO_ROW_HEIGHT_IN,
    )


apply_thesis_plot_style()


def _fallback_color(index: int) -> str:
    return MODEL_ACCURACY_COLORS[int(index) % len(MODEL_ACCURACY_COLORS)]


def _cutoff_key(cutoff: Any) -> str | None:
    if cutoff is None:
        return None
    try:
        value = float(cutoff)
    except (TypeError, ValueError):
        return None
    return f"{value:g}"


def inference_mode_key(*, mode_name: str | None = None, cutoff: Any = None) -> str:
    if mode_name is not None and str(mode_name) == "exact":
        return "exact"
    key = _cutoff_key(cutoff)
    if key is not None:
        return f"cutoff_{key}"
    return str(mode_name or "")


def inference_mode_color(*, mode_name: str | None = None, cutoff: Any = None, fallback_index: int = 0) -> str:
    key = inference_mode_key(mode_name=mode_name, cutoff=cutoff)
    return INFERENCE_MODE_COLORS.get(key, _fallback_color(fallback_index))


def dual_axis_bar_colors(*, mode_name: str | None = None, cutoff: Any = None, fallback_index: int = 0) -> Dict[str, str]:
    key = inference_mode_key(mode_name=mode_name, cutoff=cutoff)
    if key in DUAL_AXIS_BAR_COLORS:
        return DUAL_AXIS_BAR_COLORS[key]
    base = inference_mode_color(mode_name=mode_name, cutoff=cutoff, fallback_index=fallback_index)
    return {"base": base, "outer": base, "inner": base}


def style_legend_frame(legend: Legend | None) -> Legend | None:
    """Apply one thesis legend frame style and left-align its contents."""

    if legend is None:
        return None
    frame = legend.get_frame()
    frame.set_facecolor(LEGEND_BOX_FACE)
    frame.set_edgecolor(LEGEND_BOX_EDGE)
    frame.set_linewidth(LEGEND_BOX_LINEWIDTH)
    frame.set_alpha(LEGEND_BOX_ALPHA)
    try:
        frame.set_boxstyle("round,pad=0.35,rounding_size=0.18")
    except AttributeError:
        pass
    legend._legend_box.align = "left"
    return legend
