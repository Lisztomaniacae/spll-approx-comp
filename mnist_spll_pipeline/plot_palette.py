"""Shared color palette for MNIST SPLL thesis visualizations.

The palette follows the Okabe-Ito / Color Universal Design family used in
scientific figures: hues stay distinguishable under common color-vision
variants, and plot code should still keep redundant markers/line styles where
possible.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence


# Thesis export settings.  Figures are raster PNGs intended to be embedded at
# approximately A4 page width, so 300 dpi gives print-safe resolution.
FIGURE_DPI = 300
A4_PAGE_WIDTH_IN = 8.27
A4_PAGE_WIDTH_PX = int(round(FIGURE_DPI * A4_PAGE_WIDTH_IN))


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
