"""Dependency-free SVG rendering for a canonical ``SpectrumResult``."""

from __future__ import annotations

import html
import math
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..artifacts import relative_workspace_path
from .common import (
    output_directory,
    request_parts,
    success,
    unsupported,
    unwrap_artifact,
)

BACKEND_ID = "internal_plotting"
BACKEND_VERSION = "1.0.0"
ACTION_ID = "render_spectrum_plot"
MAX_PLOT_POINTS = 200_000
MAX_PLOT_STICKS = 100_000


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _number_list(value: Any, *, name: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array of finite numbers")
    result = [
        _finite_number(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    ]
    if len(result) < 2:
        raise ValueError(f"{name} must contain at least two values")
    return result


def _bool_setting(settings: dict[str, Any], name: str, default: bool) -> bool:
    value = settings.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be an explicit boolean")
    return value


def _integer_setting(
    settings: dict[str, Any], name: str, minimum: int, maximum: int
) -> int:
    if name not in settings:
        raise ValueError(f"{name} must be supplied explicitly")
    value = settings[name]
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if number != value or number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number


def _positive_float_setting(
    settings: dict[str, Any], name: str, minimum: float, maximum: float
) -> float:
    if name not in settings:
        raise ValueError(f"{name} must be supplied explicitly")
    value = _finite_number(settings[name], name=name)
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _spectrum(inputs: dict[str, Any]) -> dict[str, Any]:
    if "spectrum" not in inputs:
        raise ValueError("render_spectrum_plot requires inputs.spectrum")
    value = unwrap_artifact(inputs["spectrum"])
    while (
        isinstance(value, Mapping)
        and "result" in value
        and ("status" in value or set(value) == {"result"})
    ):
        value = value["result"]
    if not isinstance(value, Mapping):
        raise ValueError("inputs.spectrum must resolve to a SpectrumResult object")
    result = dict(value)
    schema_name = result.get("schema_name")
    if schema_name != "SpectrumResult":
        raise ValueError(f"Expected SpectrumResult, received {schema_name!r}")
    if result.get("schema_version") != "1.0":
        raise ValueError(
            "render_spectrum_plot supports SpectrumResult schema_version='1.0' only"
        )
    return result


def _write_unique_svg(directory: Path, content: str) -> Path:
    for _ in range(10):
        candidate = directory / f"spectrum_{uuid.uuid4().hex}.svg"
        if candidate.resolve().parent != directory.resolve():
            raise ValueError("Plot output path escaped the requested output directory")
        try:
            with candidate.open("x", encoding="utf-8") as handle:
                handle.write(content)
        except FileExistsError:
            continue
        except Exception:
            candidate.unlink(missing_ok=True)
            raise
        return candidate
    raise RuntimeError("Could not allocate a unique plot filename")


def _range(
    values: list[float],
    settings: dict[str, Any],
    lower_name: str,
    upper_name: str,
    *,
    include_zero: bool,
) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    if lower_name in settings:
        lower = _finite_number(settings[lower_name], name=lower_name)
    if upper_name in settings:
        upper = _finite_number(settings[upper_name], name=upper_name)
    if lower > upper:
        raise ValueError(f"{lower_name} must be smaller than {upper_name}")
    if lower == upper and lower_name in settings and upper_name in settings:
        raise ValueError(f"{lower_name} must be smaller than {upper_name}")
    if lower == upper:
        padding = max(abs(lower) * 0.05, 1.0)
        lower -= padding
        upper += padding
    return lower, upper


def _ticks(lower: float, upper: float, count: int = 5) -> list[float]:
    return [lower + index * (upper - lower) / (count - 1) for index in range(count)]


def _format_number(value: float) -> str:
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 10000.0 or magnitude < 0.001:
        return f"{value:.3e}"
    return f"{value:.6g}"


def _text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _svg(
    spectrum: dict[str, Any], settings: dict[str, Any]
) -> tuple[str, int, str, str, str]:
    axis = spectrum.get("axis")
    intensity = spectrum.get("intensity")
    if not isinstance(axis, Mapping) or not isinstance(intensity, Mapping):
        raise ValueError("SpectrumResult requires axis and intensity objects")
    x_values = _number_list(axis.get("values"), name="spectrum.axis.values")
    y_values = _number_list(intensity.get("values"), name="spectrum.intensity.values")
    if len(x_values) != len(y_values):
        raise ValueError(
            "SpectrumResult axis and intensity arrays must have the same length"
        )
    if len(x_values) > MAX_PLOT_POINTS:
        raise ValueError(
            f"SpectrumResult exceeds the plotting limit of {MAX_PLOT_POINTS} points"
        )
    reported_point_count = spectrum.get("point_count")
    if reported_point_count is not None and reported_point_count != len(x_values):
        raise ValueError(
            "SpectrumResult point_count does not match its axis/intensity arrays"
        )

    width_inches = _positive_float_setting(settings, "width_inches", 1.0, 40.0)
    height_inches = _positive_float_setting(settings, "height_inches", 1.0, 40.0)
    dpi = _integer_setting(settings, "dpi", 36, 1200)
    width = round(width_inches * dpi)
    height = round(height_inches * dpi)
    if width < 320 or height < 240 or width > 12000 or height > 12000:
        raise ValueError(
            "width_inches and height_inches at the selected dpi must produce an "
            "image between 320x240 and 12000x12000 pixels"
        )
    if "reverse_x_axis" not in settings:
        raise ValueError("reverse_x_axis must be supplied explicitly")
    reverse_x = _bool_setting(settings, "reverse_x_axis", False)
    reverse_y = _bool_setting(settings, "reverse_y_axis", False)
    show_grid = _bool_setting(settings, "show_grid", True)
    show_sticks = _bool_setting(settings, "show_sticks", True)

    x_min, x_max = _range(x_values, settings, "x_min", "x_max", include_zero=False)
    y_min, y_max = _range(y_values, settings, "y_min", "y_max", include_zero=True)
    x_padding = 0.03 * (x_max - x_min)
    y_padding = 0.05 * (y_max - y_min)
    if "x_min" not in settings:
        x_min -= x_padding
    if "x_max" not in settings:
        x_max += x_padding
    if "y_min" not in settings:
        y_min -= y_padding
    if "y_max" not in settings:
        y_max += y_padding

    left = 92.0
    right = 28.0
    top = 62.0
    bottom = 78.0
    plot_width = width - left - right
    plot_height = height - top - bottom

    def map_x(value: float) -> float:
        fraction = (value - x_min) / (x_max - x_min)
        if reverse_x:
            fraction = 1.0 - fraction
        return left + fraction * plot_width

    def map_y(value: float) -> float:
        fraction = (value - y_min) / (y_max - y_min)
        if reverse_y:
            fraction = 1.0 - fraction
        return top + (1.0 - fraction) * plot_height

    spectrum_type = str(spectrum.get("spectrum_type") or "Spectrum")
    axis_name = str(axis.get("name") or "spectral position")
    axis_unit = str(axis.get("unit") or "").strip()
    intensity_name = str(intensity.get("name") or "spectral intensity")
    intensity_unit = str(intensity.get("unit") or "").strip()
    if not axis_unit or not intensity_unit:
        raise ValueError(
            "SpectrumResult axis.unit and intensity.unit must be explicit non-empty strings"
        )
    title = str(settings.get("title") or f"{spectrum_type} Spectrum")
    x_label = str(settings.get("x_label") or f"{axis_name} ({axis_unit})")
    y_label = str(settings.get("y_label") or f"{intensity_name} ({intensity_unit})")

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{_text(title)}">'
        ),
        "<defs>",
        (
            f'<clipPath id="plot-area"><rect x="{left:.3f}" y="{top:.3f}" '
            f'width="{plot_width:.3f}" height="{plot_height:.3f}" /></clipPath>'
        ),
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff" />',
        (
            f'<text x="{width / 2:.3f}" y="30" text-anchor="middle" '
            f'font-family="sans-serif" font-size="18" fill="#111111">'
            f"{_text(title)}</text>"
        ),
    ]

    x_ticks = _ticks(x_min, x_max)
    y_ticks = _ticks(y_min, y_max)
    for value in x_ticks:
        x_position = map_x(value)
        if show_grid:
            parts.append(
                f'<line x1="{x_position:.3f}" y1="{top:.3f}" x2="{x_position:.3f}" '
                f'y2="{top + plot_height:.3f}" stroke="#e2e2e2" stroke-width="1" />'
            )
        parts.append(
            f'<line x1="{x_position:.3f}" y1="{top + plot_height:.3f}" '
            f'x2="{x_position:.3f}" y2="{top + plot_height + 5:.3f}" '
            'stroke="#222222" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{x_position:.3f}" y="{top + plot_height + 22:.3f}" '
            'text-anchor="middle" font-family="sans-serif" font-size="11" '
            f'fill="#333333">{_text(_format_number(value))}</text>'
        )
    for value in y_ticks:
        y_position = map_y(value)
        if show_grid:
            parts.append(
                f'<line x1="{left:.3f}" y1="{y_position:.3f}" '
                f'x2="{left + plot_width:.3f}" y2="{y_position:.3f}" '
                'stroke="#e2e2e2" stroke-width="1" />'
            )
        parts.append(
            f'<line x1="{left - 5:.3f}" y1="{y_position:.3f}" x2="{left:.3f}" '
            f'y2="{y_position:.3f}" stroke="#222222" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{left - 10:.3f}" y="{y_position + 4:.3f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11" fill="#333333">'
            f"{_text(_format_number(value))}</text>"
        )

    parts.extend(
        [
            (
                f'<rect x="{left:.3f}" y="{top:.3f}" width="{plot_width:.3f}" '
                f'height="{plot_height:.3f}" fill="none" stroke="#222222" '
                'stroke-width="1.2" />'
            ),
            (
                f'<text x="{left + plot_width / 2:.3f}" y="{height - 20:.3f}" '
                'text-anchor="middle" font-family="sans-serif" font-size="13" '
                f'fill="#111111">{_text(x_label)}</text>'
            ),
            (
                f'<text x="22" y="{top + plot_height / 2:.3f}" text-anchor="middle" '
                f'transform="rotate(-90 22 {top + plot_height / 2:.3f})" '
                f'font-family="sans-serif" font-size="13" fill="#111111">'
                f"{_text(y_label)}</text>"
            ),
            '<g clip-path="url(#plot-area)">',
        ]
    )
    if y_min <= 0.0 <= y_max:
        zero = map_y(0.0)
        parts.append(
            f'<line x1="{left:.3f}" y1="{zero:.3f}" x2="{left + plot_width:.3f}" '
            f'y2="{zero:.3f}" stroke="#707070" stroke-width="1" />'
        )

    if show_sticks:
        sticks = spectrum.get("sticks") or []
        if not isinstance(sticks, Sequence) or isinstance(sticks, (str, bytes)):
            raise ValueError("SpectrumResult sticks must be an array")
        if len(sticks) > MAX_PLOT_STICKS:
            raise ValueError(
                f"SpectrumResult exceeds the plotting limit of {MAX_PLOT_STICKS} sticks"
            )
        baseline = map_y(0.0)
        for index, stick in enumerate(sticks):
            if not isinstance(stick, Mapping):
                raise ValueError(f"SpectrumResult sticks[{index}] must be an object")
            position = _finite_number(
                stick.get("position"), name=f"sticks[{index}].position"
            )
            strength = _finite_number(
                stick.get("intensity"), name=f"sticks[{index}].intensity"
            )
            parts.append(
                f'<line x1="{map_x(position):.3f}" y1="{baseline:.3f}" '
                f'x2="{map_x(position):.3f}" y2="{map_y(strength):.3f}" '
                'stroke="#202020" stroke-width="1" opacity="0.75" />'
            )

    points = " ".join(
        f"{map_x(x_value):.3f},{map_y(y_value):.3f}"
        for x_value, y_value in zip(x_values, y_values, strict=True)
    )
    parts.extend(
        [
            f'<polyline points="{points}" fill="none" stroke="#b3212d" '
            'stroke-width="2" />',
            "</g>",
            "</svg>",
        ]
    )
    return (
        "\n".join(parts) + "\n",
        len(x_values),
        spectrum_type,
        axis_unit,
        intensity_unit,
    )


def _render(request: dict[str, Any]) -> dict[str, Any]:
    inputs, _method, settings = request_parts(request)
    if "output_format" not in settings:
        raise ValueError("output_format must be supplied explicitly")
    output_format = str(settings["output_format"]).strip().lower()
    if output_format != "svg":
        raise ValueError(
            "The dependency-free plotting backend supports format='svg' only"
        )
    spectrum = _spectrum(inputs)
    svg, point_count, spectrum_type, axis_unit, intensity_unit = _svg(
        spectrum, settings
    )
    directory = output_directory(ACTION_ID, BACKEND_ID)
    path = _write_unique_svg(directory, svg)
    relative_path = relative_workspace_path(path)
    result = {
        "schema_name": "PlotArtifact",
        "schema_version": "1.0",
        "path": relative_path,
        "format": "svg",
        "media_type": "image/svg+xml",
        "point_count": point_count,
        "spectrum_type": spectrum_type,
        "axis_unit": axis_unit,
        "intensity_unit": intensity_unit,
    }
    return success(
        result,
        artifact_files=[
            {
                "path": relative_path,
                "semantic_type": "SpectrumPlotFile",
                "media_type": "image/svg+xml",
            }
        ],
        backend_version=BACKEND_VERSION,
        provenance={
            "implementation": "minichem_toolbox.backends.plotting",
            "renderer": "dependency_free_svg",
        },
    )


def execute(action_id: str, backend_id: str, request: dict[str, Any]) -> dict[str, Any]:
    if backend_id != BACKEND_ID:
        return unsupported(f"Unsupported plotting backend: {backend_id}")
    if action_id != ACTION_ID:
        return unsupported(f"Unsupported plotting action: {action_id}")
    return _render(request)
