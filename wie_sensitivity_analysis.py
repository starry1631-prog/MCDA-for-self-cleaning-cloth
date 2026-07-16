#!/usr/bin/env python3
"""
WIE sensitivity analysis for nanoparticle-functionalized textile formulations.

The script reads the first worksheet of the supplied .xlsx file without pandas or
openpyxl, recalculates WIE-standardized washing cycles across a Q10/alpha grid,
recalculates durability and antibacterial scores, reruns the MCDA weight-space
analysis, and exports audit-ready CSV files and figures.

Run:
    python wie_sensitivity_analysis.py "Original Data Extracted From Sources(1).xlsx"

Optional:
    python wie_sensitivity_analysis.py input.xlsx --output-dir results \
        --q10 1.5,2.0,2.5,3.0 --alpha 0.3,0.4,0.5,0.6,0.7

Dependencies:
    numpy, matplotlib
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Editable study settings
# -----------------------------------------------------------------------------

T_REF_C = 40.0
T_REF_MIN = 45.0
TARGET_REFERENCE_WASHES = 10.0
BASELINE_Q10 = 2.0
BASELINE_ALPHA = 0.5

# Full precision is preferable for sensitivity analysis. Set to 2 only if you
# need to reproduce calculations that rounded WIE before multiplying cycles.
ROUND_WIE_DECIMALS: Optional[int] = None

# Antibacterial reduction is physically bounded between 0% and 100%.
CLAMP_ANTIBACTERIAL_TO_100 = True

# MCDA resolution. 0.01 gives 101 two-criterion points and 5,151 simplex points.
MCDA_STEP = 0.01


@dataclass(frozen=True)
class FormulationMeta:
    display_name: str
    temperature_c: Optional[float]
    duration_min: Optional[float]
    cost_score: float
    material_class: str


FORMULATION_METADATA: Dict[str, FormulationMeta] = {
    "tio2_two_step": FormulationMeta(
        "TiO2 Two-Step Dipping", 40.0, 45.0, 0.264, "TiO2"
    ),
    "tio2_alkaline": FormulationMeta(
        "TiO2 Alkaline Hydrolysis", 60.0, 20.0, 0.327, "TiO2"
    ),
    "hybrid_pad": FormulationMeta(
        "Hybrid Padding-Squeezing", 60.0, 30.0, 1.000, "Hybrid"
    ),
    "hybrid_insitu": FormulationMeta(
        "Hybrid In-Situ", 40.0, 45.0, 0.496, "Hybrid"
    ),
    "zno_starch": FormulationMeta(
        "ZnO Starch", 22.0, 5.0, 0.000, "ZnO"
    ),
    "zno_sds": FormulationMeta(
        "ZnO SDS", 22.0, 5.0, 0.377, "ZnO"
    ),
    # The Ag studies do not report temperature/duration. WIE is therefore fixed
    # at 1.0 in this analysis, exactly as in the manuscript baseline.
    "ag_two_step": FormulationMeta(
        "Ag Two-Step", None, None, 0.663, "Ag"
    ),
    "ag_one_step": FormulationMeta(
        "Ag One-Step", None, None, 0.203, "Ag"
    ),
}


ALIASES = {
    "tio2twosteapdipping": "tio2_two_step",
    "tio2twostepdipping": "tio2_two_step",
    "tio2alkalinehydrolysis": "tio2_alkaline",
    "hybridpadsqueeze": "hybrid_pad",
    "hybridpaddingsqueezing": "hybrid_pad",
    "hybridinsitu": "hybrid_insitu",
    "znostarch": "zno_starch",
    "znosds": "zno_sds",
    "agtwostep": "ag_two_step",
    "agonestep": "ag_one_step",
}


# Current manuscript values, used only for an audit comparison. They do not
# control the calculations.
MANUSCRIPT_BASELINE = {
    "tio2_two_step": {"durability": 0.459, "antibacterial": None},
    "tio2_alkaline": {"durability": 0.772, "antibacterial": None},
    "hybrid_pad": {"durability": 0.623, "antibacterial": 0.999},
    "hybrid_insitu": {"durability": 0.738, "antibacterial": 0.996},
    "zno_starch": {"durability": 0.805, "antibacterial": 0.395},
    "zno_sds": {"durability": 0.540, "antibacterial": 0.793},
    "ag_two_step": {"durability": 0.917, "antibacterial": 0.997},
    "ag_one_step": {"durability": 0.911, "antibacterial": 0.929},
}


# -----------------------------------------------------------------------------
# Minimal XLSX reader (first worksheet, values only)
# -----------------------------------------------------------------------------

_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL_DOC = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_REL_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"


def _column_number(cell_reference: str) -> int:
    letters = re.match(r"[A-Za-z]+", cell_reference)
    if not letters:
        raise ValueError(f"Invalid cell reference: {cell_reference!r}")
    value = 0
    for char in letters.group(0).upper():
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def _normalize_zip_path(base: str, target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    base_dir = str(Path(base).parent)
    combined = Path(base_dir, target)
    parts: List[str] = []
    for part in combined.parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part not in (".", ""):
            parts.append(part)
    return "/".join(parts)


def read_first_sheet_xlsx(path: Path) -> List[List[object]]:
    """Read the first worksheet as a rectangular matrix of values."""
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"Cannot open XLSX file: {path}") from exc

    with archive:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{{{_NS_MAIN}}}si"):
                text = "".join(t.text or "" for t in si.iter(f"{{{_NS_MAIN}}}t"))
                shared_strings.append(text)

        workbook_path = "xl/workbook.xml"
        workbook_root = ET.fromstring(archive.read(workbook_path))
        first_sheet = workbook_root.find(f".//{{{_NS_MAIN}}}sheet")
        if first_sheet is None:
            raise RuntimeError("The workbook contains no worksheets.")
        relationship_id = first_sheet.attrib.get(f"{{{_NS_REL_DOC}}}id")
        if not relationship_id:
            raise RuntimeError("Cannot resolve the first worksheet relationship.")

        rels_path = "xl/_rels/workbook.xml.rels"
        rels_root = ET.fromstring(archive.read(rels_path))
        worksheet_target = None
        for rel in rels_root.findall(f"{{{_NS_REL_PKG}}}Relationship"):
            if rel.attrib.get("Id") == relationship_id:
                worksheet_target = rel.attrib.get("Target")
                break
        if not worksheet_target:
            raise RuntimeError("Cannot locate the first worksheet XML.")
        worksheet_path = _normalize_zip_path(workbook_path, worksheet_target)

        sheet_root = ET.fromstring(archive.read(worksheet_path))
        cell_values: Dict[Tuple[int, int], object] = {}
        max_row = -1
        max_col = -1

        for cell in sheet_root.findall(f".//{{{_NS_MAIN}}}c"):
            reference = cell.attrib.get("r")
            if not reference:
                continue
            row_match = re.search(r"\d+", reference)
            if not row_match:
                continue
            row_index = int(row_match.group(0)) - 1
            col_index = _column_number(reference)
            cell_type = cell.attrib.get("t")

            value: object = None
            if cell_type == "inlineStr":
                inline = cell.find(f"{{{_NS_MAIN}}}is")
                if inline is not None:
                    value = "".join(
                        t.text or "" for t in inline.iter(f"{{{_NS_MAIN}}}t")
                    )
            else:
                value_node = cell.find(f"{{{_NS_MAIN}}}v")
                raw_value = value_node.text if value_node is not None else None
                if raw_value is not None:
                    if cell_type == "s":
                        value = shared_strings[int(raw_value)]
                    elif cell_type == "b":
                        value = raw_value == "1"
                    elif cell_type in ("str", "e"):
                        value = raw_value
                    else:
                        try:
                            number = float(raw_value)
                            value = int(number) if number.is_integer() else number
                        except ValueError:
                            value = raw_value

            cell_values[(row_index, col_index)] = value
            max_row = max(max_row, row_index)
            max_col = max(max_col, col_index)

        return [
            [cell_values.get((r, c)) for c in range(max_col + 1)]
            for r in range(max_row + 1)
        ]


# -----------------------------------------------------------------------------
# Input parsing
# -----------------------------------------------------------------------------

_POINT_PATTERN = re.compile(
    r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)"
)


def canonical_key(name: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "", str(name).lower())
    if text not in ALIASES:
        raise ValueError(
            f"Unrecognized formulation name {name!r}. Add an alias in ALIASES."
        )
    return ALIASES[text]


def parse_point(value: object) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "x":
        return None
    match = _POINT_PATTERN.fullmatch(text)
    if not match:
        raise ValueError(f"Cannot parse data point: {value!r}")
    return float(match.group(1)), float(match.group(2))


def _collect_points(row: Sequence[object], start: int, end: int) -> List[Tuple[float, float]]:
    points = [parse_point(value) for value in row[start:end]]
    clean = sorted(point for point in points if point is not None)
    seen: Dict[float, float] = {}
    for x, y in clean:
        if x in seen and not math.isclose(seen[x], y, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Duplicate cycle {x} with conflicting values {seen[x]} and {y}.")
        seen[x] = y
    return sorted(seen.items())


def load_source_data(path: Path) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    rows = read_first_sheet_xlsx(path)
    data: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}

    for row in rows[2:]:  # first two rows are headers
        if not row or row[0] is None or str(row[0]).strip() == "":
            continue
        row = list(row) + [None] * max(0, 11 - len(row))
        key = canonical_key(row[0])
        durability = _collect_points(row, 1, 5)
        e_coli = _collect_points(row, 5, 8)
        s_aureus = _collect_points(row, 8, 11)
        if not durability:
            raise ValueError(f"No durability data found for {row[0]!r}.")
        data[key] = {
            "durability": durability,
            "e_coli": e_coli,
            "s_aureus": s_aureus,
        }

    missing = set(FORMULATION_METADATA) - set(data)
    extra = set(data) - set(FORMULATION_METADATA)
    if missing:
        raise ValueError(f"Missing expected formulations: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected formulations: {sorted(extra)}")
    return data


# -----------------------------------------------------------------------------
# Score calculations
# -----------------------------------------------------------------------------


def calculate_wie(
    temperature_c: Optional[float],
    duration_min: Optional[float],
    q10: float,
    alpha: float,
) -> float:
    if temperature_c is None or duration_min is None:
        return 1.0
    thermal = q10 ** ((temperature_c - T_REF_C) / 10.0)
    duration = duration_min / T_REF_MIN
    value = thermal**alpha * duration ** (1.0 - alpha)
    if ROUND_WIE_DECIMALS is not None:
        value = round(value, ROUND_WIE_DECIMALS)
    return value


def fit_initial(points: Sequence[Tuple[float, float]]) -> Dict[str, float | str | bool]:
    for x, y in points:
        if math.isclose(x, 0.0, abs_tol=1e-12):
            return {
                "initial": float(y),
                "method": "reported_cycle_0",
                "slope": math.nan,
                "intercept": float(y),
                "r_squared": math.nan,
                "estimated": False,
            }

    if len(points) < 2:
        raise ValueError("At least two points are required to estimate cycle 0.")
    x = np.array([point[0] for point in points], dtype=float)
    y = np.array([point[1] for point in points], dtype=float)
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = slope * x + intercept
    residual_sum = float(np.sum((y - predicted) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 1.0
    if intercept <= 0:
        raise ValueError(f"Estimated cycle-0 value is not positive: {intercept}")
    return {
        "initial": float(intercept),
        "method": "linear_best_fit_intercept",
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "estimated": True,
    }


def standardized_points(
    points: Sequence[Tuple[float, float]], wie: float, initial: Optional[float] = None
) -> List[Tuple[float, float]]:
    converted = [(x * wie, y) for x, y in points]
    if initial is not None and not any(math.isclose(x, 0.0, abs_tol=1e-12) for x, _ in points):
        converted.append((0.0, initial))
    return sorted(converted)


def linear_value_at(
    points: Sequence[Tuple[float, float]], target: float
) -> Tuple[float, str, float]:
    """Return linearly interpolated/extrapolated value, method, and distance."""
    ordered = sorted(points)
    if len(ordered) < 2:
        raise ValueError("At least two points are required for linear interpolation/extrapolation.")

    for x, y in ordered:
        if math.isclose(x, target, rel_tol=0, abs_tol=1e-10):
            return y, "exact", 0.0

    if target < ordered[0][0]:
        left, right = ordered[0], ordered[1]
        method = "backward_extrapolation"
        distance = ordered[0][0] - target
    elif target > ordered[-1][0]:
        left, right = ordered[-2], ordered[-1]
        method = "forward_extrapolation"
        distance = target - ordered[-1][0]
    else:
        method = "interpolation"
        distance = 0.0
        left = right = ordered[0]
        for index in range(len(ordered) - 1):
            if ordered[index][0] < target < ordered[index + 1][0]:
                left, right = ordered[index], ordered[index + 1]
                break

    x1, y1 = left
    x2, y2 = right
    if math.isclose(x1, x2, abs_tol=1e-15):
        raise ValueError("Cannot interpolate between identical x values.")
    value = y1 + (target - x1) / (x2 - x1) * (y2 - y1)
    return float(value), method, float(distance)


def coverage_factor_for_short_horizon(wmax: float) -> Tuple[float, str]:
    """
    Evidence-coverage factor used when Wmax < 10.

    By two washes, 75% of the assumed 30-cycle loss has occurred; by ten washes,
    80% has occurred. Therefore two washes cover 75/80 = 0.9375 of the estimated
    ten-wash loss. Below two washes, coverage is proportional to Wmax/2. From
    two to ten washes, coverage increases linearly from 0.9375 to 1.0.
    """
    if wmax < 0:
        raise ValueError("Wmax cannot be negative.")
    if wmax < 2.0:
        return 0.9375 * (wmax / 2.0), "Wmax_below_2"
    if wmax < TARGET_REFERENCE_WASHES:
        cumulative_loss_fraction = 0.75 + 0.05 * (wmax - 2.0) / 8.0
        return cumulative_loss_fraction / 0.80, "Wmax_2_to_10"
    return 1.0, "Wmax_at_least_10"


def calculate_durability(
    raw_points: Sequence[Tuple[float, float]], wie: float
) -> Dict[str, float | str]:
    fit = fit_initial(raw_points)
    p0 = float(fit["initial"])
    converted = standardized_points(raw_points, wie, initial=p0)
    original_without_synthetic = [(x * wie, y) for x, y in raw_points]
    wmax, pmax = max(original_without_synthetic, key=lambda item: item[0])
    rmax = pmax / p0

    if wmax >= TARGET_REFERENCE_WASHES:
        p10, method, distance = linear_value_at(converted, TARGET_REFERENCE_WASHES)
        score = p10 / p0
        coverage = 1.0
        branch = "R10_interpolated_or_exact"
    else:
        coverage, branch = coverage_factor_for_short_horizon(wmax)
        score = rmax * coverage
        method = branch
        distance = TARGET_REFERENCE_WASHES - wmax

    unclamped = score
    score = min(1.0, max(0.0, score))
    return {
        "initial": p0,
        "initial_method": str(fit["method"]),
        "initial_slope": float(fit["slope"]),
        "initial_r_squared": float(fit["r_squared"]),
        "wmax": float(wmax),
        "rmax": float(rmax),
        "coverage_factor": float(coverage),
        "durability_score": float(score),
        "durability_unclamped": float(unclamped),
        "durability_method": method,
        "durability_branch": branch,
        "target_distance": float(distance),
    }


def calculate_antibacterial_species(
    raw_points: Sequence[Tuple[float, float]], wie: float
) -> Optional[Dict[str, float | str]]:
    if not raw_points:
        return None
    converted = standardized_points(raw_points, wie)
    value, method, distance = linear_value_at(converted, TARGET_REFERENCE_WASHES)
    raw_value = value
    value = max(0.0, value)
    if CLAMP_ANTIBACTERIAL_TO_100:
        value = min(100.0, value)
    return {
        "percent_at_10": float(value),
        "raw_percent_at_10": float(raw_value),
        "score": float(value / 100.0),
        "method": method,
        "target_distance": float(distance),
    }


def calculate_scores_for_scenario(
    source_data: Mapping[str, Mapping[str, Sequence[Tuple[float, float]]]],
    q10: float,
    alpha: float,
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for key, series in source_data.items():
        meta = FORMULATION_METADATA[key]
        wie = calculate_wie(meta.temperature_c, meta.duration_min, q10, alpha)
        durability = calculate_durability(series["durability"], wie)
        e_coli = calculate_antibacterial_species(series["e_coli"], wie)
        s_aureus = calculate_antibacterial_species(series["s_aureus"], wie)
        antibacterial = None
        if e_coli is not None and s_aureus is not None:
            antibacterial = (float(e_coli["score"]) + float(s_aureus["score"])) / 2.0

        results[key] = {
            "key": key,
            "formulation": meta.display_name,
            "material_class": meta.material_class,
            "temperature_c": meta.temperature_c,
            "duration_min": meta.duration_min,
            "q10": q10,
            "alpha": alpha,
            "wie": wie,
            "cost_score": meta.cost_score,
            **durability,
            "e_coli_score": None if e_coli is None else e_coli["score"],
            "e_coli_percent_at_10": None if e_coli is None else e_coli["percent_at_10"],
            "e_coli_method": None if e_coli is None else e_coli["method"],
            "e_coli_target_distance": None if e_coli is None else e_coli["target_distance"],
            "s_aureus_score": None if s_aureus is None else s_aureus["score"],
            "s_aureus_percent_at_10": None if s_aureus is None else s_aureus["percent_at_10"],
            "s_aureus_method": None if s_aureus is None else s_aureus["method"],
            "s_aureus_target_distance": None if s_aureus is None else s_aureus["target_distance"],
            "antibacterial_score": antibacterial,
        }
    return results


# -----------------------------------------------------------------------------
# MCDA and Pareto analysis
# -----------------------------------------------------------------------------


def _winner_shares(
    alternatives: Mapping[str, Sequence[float]], weight_vectors: Iterable[Sequence[float]]
) -> Dict[str, float]:
    shares = {key: 0.0 for key in alternatives}
    total = 0
    for weights in weight_vectors:
        total += 1
        scores = {
            key: float(np.dot(np.asarray(values, dtype=float), np.asarray(weights, dtype=float)))
            for key, values in alternatives.items()
        }
        top = max(scores.values())
        winners = [key for key, score in scores.items() if math.isclose(score, top, abs_tol=1e-12)]
        increment = 1.0 / len(winners)
        for key in winners:
            shares[key] += increment
    if total == 0:
        raise ValueError("Weight grid is empty.")
    return {key: 100.0 * value / total for key, value in shares.items()}


def two_criterion_weight_grid(step: float) -> List[Tuple[float, float]]:
    count = int(round(1.0 / step))
    return [(i / count, 1.0 - i / count) for i in range(count + 1)]


def three_criterion_weight_grid(step: float) -> List[Tuple[float, float, float]]:
    count = int(round(1.0 / step))
    grid: List[Tuple[float, float, float]] = []
    for i in range(count + 1):
        for j in range(count - i + 1):
            k = count - i - j
            grid.append((i / count, j / count, k / count))
    return grid


def pareto_front(alternatives: Mapping[str, Sequence[float]]) -> List[str]:
    keys = list(alternatives)
    front: List[str] = []
    for key in keys:
        candidate = np.asarray(alternatives[key], dtype=float)
        dominated = False
        for other_key in keys:
            if other_key == key:
                continue
            other = np.asarray(alternatives[other_key], dtype=float)
            if np.all(other >= candidate) and np.any(other > candidate):
                dominated = True
                break
        if not dominated:
            front.append(key)
    return front


def pair_crossover_weight(
    first: Sequence[float], second: Sequence[float]
) -> Optional[float]:
    """Durability weight where two D-C lines cross."""
    d1, c1 = first
    d2, c2 = second
    denominator = (d1 - c1) - (d2 - c2)
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return None
    weight = (c2 - c1) / denominator
    return float(weight) if 0.0 <= weight <= 1.0 else None


def run_mcda(scores: Mapping[str, Mapping[str, object]], step: float) -> Dict[str, object]:
    dc = {
        key: (float(row["durability_score"]), float(row["cost_score"]))
        for key, row in scores.items()
    }
    dac = {
        key: (
            float(row["durability_score"]),
            float(row["antibacterial_score"]),
            float(row["cost_score"]),
        )
        for key, row in scores.items()
        if row["antibacterial_score"] is not None
    }
    dac_no_ag = {key: value for key, value in dac.items() if not key.startswith("ag_")}

    dc_shares = _winner_shares(dc, two_criterion_weight_grid(step))
    dac_shares = _winner_shares(dac, three_criterion_weight_grid(step))
    dac_no_ag_shares = _winner_shares(dac_no_ag, three_criterion_weight_grid(step))

    return {
        "dc_shares": dc_shares,
        "dac_shares": dac_shares,
        "dac_no_ag_shares": dac_no_ag_shares,
        "dc_pareto": pareto_front(dc),
        "dac_pareto": pareto_front(dac),
        "dac_no_ag_pareto": pareto_front(dac_no_ag),
        "hybrid_ag_crossover": pair_crossover_weight(dc["hybrid_pad"], dc["ag_two_step"]),
        "dc_cost_only_winner": max(dc, key=lambda key: dc[key][1]),
        "dc_balanced_winner": max(dc, key=lambda key: 0.5 * dc[key][0] + 0.5 * dc[key][1]),
        "dc_durability_only_winner": max(dc, key=lambda key: dc[key][0]),
    }


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_front(keys: Sequence[str]) -> str:
    return " | ".join(FORMULATION_METADATA[key].display_name for key in keys)


def create_initial_fit_outputs(
    source_data: Mapping[str, Mapping[str, Sequence[Tuple[float, float]]]], output_dir: Path
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    fit_dir = output_dir / "figures" / "initial_fits"
    fit_dir.mkdir(parents=True, exist_ok=True)

    for key, series_map in source_data.items():
        for series_name in ("durability", "e_coli", "s_aureus"):
            points = series_map[series_name]
            if not points or any(math.isclose(x, 0.0, abs_tol=1e-12) for x, _ in points):
                continue
            fit = fit_initial(points)
            used_in_score = series_name == "durability"
            rows.append(
                {
                    "formulation": FORMULATION_METADATA[key].display_name,
                    "series": series_name,
                    "estimated_cycle_0": fit["initial"],
                    "slope": fit["slope"],
                    "r_squared": fit["r_squared"],
                    "used_to_normalize_score": used_in_score,
                }
            )

            x = np.array([point[0] for point in points], dtype=float)
            y = np.array([point[1] for point in points], dtype=float)
            x_line = np.linspace(0.0, max(x) * 1.05, 100)
            y_line = float(fit["slope"]) * x_line + float(fit["intercept"])
            plt.figure(figsize=(7.0, 4.5))
            plt.scatter(x, y, label="Reported data")
            plt.plot(x_line, y_line, label="Linear best fit")
            plt.scatter([0], [fit["initial"]], marker="x", s=90, label="Estimated cycle 0")
            plt.xlabel("Unstandardized washing cycles")
            plt.ylabel("Reported value")
            plt.title(f"{FORMULATION_METADATA[key].display_name}: {series_name} cycle-0 fit")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fit_dir / f"{slug(FORMULATION_METADATA[key].display_name)}_{series_name}.png", dpi=220)
            plt.close()
    return rows


def plot_heatmap(
    matrix: np.ndarray,
    q10_values: Sequence[float],
    alpha_values: Sequence[float],
    title: str,
    label: str,
    output_path: Path,
    decimals: int = 3,
) -> None:
    plt.figure(figsize=(7.2, 5.0))
    image = plt.imshow(matrix, aspect="auto", origin="lower")
    plt.colorbar(image, label=label)
    plt.xticks(range(len(q10_values)), [f"{value:g}" for value in q10_values])
    plt.yticks(range(len(alpha_values)), [f"{value:g}" for value in alpha_values])
    plt.xlabel("Q10")
    plt.ylabel("Temperature weighting alpha")
    plt.title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, f"{matrix[row, col]:.{decimals}f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def create_heatmaps(
    scenario_scores: Mapping[Tuple[float, float], Mapping[str, Mapping[str, object]]],
    scenario_mcda: Mapping[Tuple[float, float], Mapping[str, object]],
    q10_values: Sequence[float],
    alpha_values: Sequence[float],
    output_dir: Path,
) -> None:
    score_dir = output_dir / "figures" / "score_heatmaps"
    mcda_dir = output_dir / "figures" / "mcda_heatmaps"

    for key, meta in FORMULATION_METADATA.items():
        durability_matrix = np.array(
            [
                [scenario_scores[(q10, alpha)][key]["durability_score"] for q10 in q10_values]
                for alpha in alpha_values
            ],
            dtype=float,
        )
        if float(np.ptp(durability_matrix)) > 1e-12:
            plot_heatmap(
                durability_matrix,
                q10_values,
                alpha_values,
                f"Durability sensitivity: {meta.display_name}",
                "Durability score",
                score_dir / f"durability_{slug(meta.display_name)}.png",
            )

        values = [
            scenario_scores[(q10, alpha)][key]["antibacterial_score"]
            for alpha in alpha_values
            for q10 in q10_values
        ]
        if all(value is not None for value in values):
            antibacterial_matrix = np.array(
                [
                    [scenario_scores[(q10, alpha)][key]["antibacterial_score"] for q10 in q10_values]
                    for alpha in alpha_values
                ],
                dtype=float,
            )
            if float(np.ptp(antibacterial_matrix)) > 1e-12:
                plot_heatmap(
                    antibacterial_matrix,
                    q10_values,
                    alpha_values,
                    f"Antibacterial sensitivity: {meta.display_name}",
                    "Antibacterial score",
                    score_dir / f"antibacterial_{slug(meta.display_name)}.png",
                )

    crossover_matrix = np.array(
        [
            [
                scenario_mcda[(q10, alpha)]["hybrid_ag_crossover"]
                if scenario_mcda[(q10, alpha)]["hybrid_ag_crossover"] is not None
                else np.nan
                for q10 in q10_values
            ]
            for alpha in alpha_values
        ],
        dtype=float,
    )
    plot_heatmap(
        crossover_matrix,
        q10_values,
        alpha_values,
        "Hybrid Padding-Squeezing / Ag Two-Step crossover",
        "Durability weight at crossover",
        mcda_dir / "dc_hybrid_ag_crossover.png",
    )

    for analysis_name, share_key in (
        ("Durability-Cost", "dc_shares"),
        ("Durability-Antibacterial-Cost", "dac_shares"),
        ("DAC excluding Ag", "dac_no_ag_shares"),
    ):
        all_keys = sorted(
            {
                key
                for result in scenario_mcda.values()
                for key, share in result[share_key].items()
                if share > 1e-12
            }
        )
        for key in all_keys:
            matrix = np.array(
                [
                    [scenario_mcda[(q10, alpha)][share_key].get(key, 0.0) for q10 in q10_values]
                    for alpha in alpha_values
                ],
                dtype=float,
            )
            plot_heatmap(
                matrix,
                q10_values,
                alpha_values,
                f"{analysis_name} winning share: {FORMULATION_METADATA[key].display_name}",
                "Weight-space share (%)",
                mcda_dir / f"{slug(analysis_name)}_{slug(FORMULATION_METADATA[key].display_name)}.png",
                decimals=1,
            )


def write_readme(
    output_dir: Path,
    input_path: Path,
    q10_values: Sequence[float],
    alpha_values: Sequence[float],
) -> None:
    text = f"""WIE sensitivity analysis outputs
================================

Input workbook: {input_path.name}
Q10 values: {', '.join(map(str, q10_values))}
Alpha values: {', '.join(map(str, alpha_values))}
Reference condition: {T_REF_C:g} C, {T_REF_MIN:g} min
Target horizon: {TARGET_REFERENCE_WASHES:g} reference-equivalent washes
WIE rounding: {ROUND_WIE_DECIMALS if ROUND_WIE_DECIMALS is not None else 'none (full precision)'}
MCDA grid step: {MCDA_STEP:g}

Main files
----------
all_scenario_scores.csv
    One row per formulation and Q10-alpha combination. Includes WIE, Wmax,
    calculation branch, durability score, species-specific antibacterial scores,
    and interpolation/extrapolation audit fields.

scenario_mcda_summary.csv
    One row per Q10-alpha combination. Includes Pareto fronts, Hybrid/Ag
    crossover, and weight-space shares for the D-C, D-A-C, and Ag-excluded
    D-A-C analyses.

score_ranges.csv
    Baseline, minimum, maximum, and range of WIE, Wmax, durability, and
    antibacterial scores for each formulation.

baseline_scores.csv
    Scores at Q10={BASELINE_Q10:g}, alpha={BASELINE_ALPHA:g} using the exact
    calculation implemented here.

baseline_vs_manuscript.csv
    Audit comparison between recalculated baseline values and the values currently
    written in the manuscript. Differences should be checked against the raw data,
    rounding choices, and previous versions of the score calculation.

initial_fit_report.csv
    Cycle-0 intercepts estimated by linear best fit where no initial value was
    reported. Durability intercepts are used for retention normalization;
    antibacterial intercepts are reported for completeness but are not used in the
    antibacterial score.

Figures
-------
figures/initial_fits/
    Best-fit plots for series lacking cycle 0.
figures/score_heatmaps/
    Parameter heatmaps for scores that vary across the WIE grid.
figures/mcda_heatmaps/
    Crossover and weight-space-share heatmaps.

Interpretation note
-------------------
For 2 <= Wmax < 10, the evidence-coverage factor rises linearly from 0.9375
at two washes to 1.0 at ten washes, consistent with the stated assumption that
75% of 30-cycle loss occurs by wash 2 and 80% occurs by wash 10.
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------


def parse_float_list(text: str, option_name: str) -> List[float]:
    try:
        values = [float(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid {option_name} list: {text!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{option_name} list cannot be empty.")
    return values


def nearest_scenario(
    q10_values: Sequence[float], alpha_values: Sequence[float], q10: float, alpha: float
) -> Tuple[float, float]:
    return min(q10_values, key=lambda value: abs(value - q10)), min(
        alpha_values, key=lambda value: abs(value - alpha)
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_xlsx", type=Path, help="Corrected source-data workbook")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("WIE_sensitivity_results"),
        help="Directory for CSV files and figures",
    )
    parser.add_argument(
        "--q10",
        default="1.5,2.0,2.5,3.0",
        help="Comma-separated Q10 values",
    )
    parser.add_argument(
        "--alpha",
        default="0.3,0.4,0.5,0.6,0.7",
        help="Comma-separated temperature-weighting values",
    )
    parser.add_argument(
        "--mcda-step",
        type=float,
        default=MCDA_STEP,
        help="Weight-grid step, normally 0.01",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG figure generation")
    args = parser.parse_args(argv)

    if not args.input_xlsx.exists():
        parser.error(f"Input file does not exist: {args.input_xlsx}")
    if args.mcda_step <= 0 or args.mcda_step > 1:
        parser.error("--mcda-step must be between 0 and 1.")

    q10_values = sorted(set(parse_float_list(args.q10, "Q10")))
    alpha_values = sorted(set(parse_float_list(args.alpha, "alpha")))
    if any(value <= 0 for value in q10_values):
        parser.error("All Q10 values must be positive.")
    if any(value < 0 or value > 1 for value in alpha_values):
        parser.error("All alpha values must lie between 0 and 1.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    source_data = load_source_data(args.input_xlsx)

    initial_fit_rows = create_initial_fit_outputs(source_data, output_dir)
    if initial_fit_rows:
        write_csv(
            output_dir / "initial_fit_report.csv",
            initial_fit_rows,
            [
                "formulation",
                "series",
                "estimated_cycle_0",
                "slope",
                "r_squared",
                "used_to_normalize_score",
            ],
        )

    scenario_scores: Dict[Tuple[float, float], Dict[str, Dict[str, object]]] = {}
    scenario_mcda: Dict[Tuple[float, float], Dict[str, object]] = {}
    all_score_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for q10 in q10_values:
        for alpha in alpha_values:
            scores = calculate_scores_for_scenario(source_data, q10, alpha)
            mcda = run_mcda(scores, args.mcda_step)
            scenario_scores[(q10, alpha)] = scores
            scenario_mcda[(q10, alpha)] = mcda
            all_score_rows.extend(scores.values())

            row: Dict[str, object] = {
                "q10": q10,
                "alpha": alpha,
                "dc_cost_only_winner": FORMULATION_METADATA[mcda["dc_cost_only_winner"]].display_name,
                "dc_balanced_winner": FORMULATION_METADATA[mcda["dc_balanced_winner"]].display_name,
                "dc_durability_only_winner": FORMULATION_METADATA[mcda["dc_durability_only_winner"]].display_name,
                "hybrid_ag_crossover_durability_weight": mcda["hybrid_ag_crossover"],
                "dc_pareto": format_front(mcda["dc_pareto"]),
                "dac_pareto": format_front(mcda["dac_pareto"]),
                "dac_no_ag_pareto": format_front(mcda["dac_no_ag_pareto"]),
            }
            for key, share in mcda["dc_shares"].items():
                row[f"dc_share__{slug(FORMULATION_METADATA[key].display_name)}"] = share
            for key, share in mcda["dac_shares"].items():
                row[f"dac_share__{slug(FORMULATION_METADATA[key].display_name)}"] = share
            for key, share in mcda["dac_no_ag_shares"].items():
                row[f"dac_no_ag_share__{slug(FORMULATION_METADATA[key].display_name)}"] = share
            summary_rows.append(row)

    score_fields = [
        "q10",
        "alpha",
        "key",
        "formulation",
        "material_class",
        "temperature_c",
        "duration_min",
        "wie",
        "wmax",
        "initial",
        "initial_method",
        "initial_slope",
        "initial_r_squared",
        "rmax",
        "coverage_factor",
        "durability_score",
        "durability_unclamped",
        "durability_method",
        "durability_branch",
        "target_distance",
        "e_coli_percent_at_10",
        "e_coli_score",
        "e_coli_method",
        "e_coli_target_distance",
        "s_aureus_percent_at_10",
        "s_aureus_score",
        "s_aureus_method",
        "s_aureus_target_distance",
        "antibacterial_score",
        "cost_score",
    ]
    write_csv(output_dir / "all_scenario_scores.csv", all_score_rows, score_fields)

    summary_fields = list(dict.fromkeys(key for row in summary_rows for key in row.keys()))
    write_csv(output_dir / "scenario_mcda_summary.csv", summary_rows, summary_fields)

    baseline_key = nearest_scenario(q10_values, alpha_values, BASELINE_Q10, BASELINE_ALPHA)
    baseline = scenario_scores[baseline_key]
    baseline_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    range_rows: List[Dict[str, object]] = []

    for key, meta in FORMULATION_METADATA.items():
        baseline_row = baseline[key]
        baseline_rows.append(
            {
                "formulation": meta.display_name,
                "q10": baseline_key[0],
                "alpha": baseline_key[1],
                "wie": baseline_row["wie"],
                "wmax": baseline_row["wmax"],
                "durability_score": baseline_row["durability_score"],
                "antibacterial_score": baseline_row["antibacterial_score"],
                "cost_score": baseline_row["cost_score"],
                "durability_branch": baseline_row["durability_branch"],
            }
        )

        manuscript = MANUSCRIPT_BASELINE[key]
        calculated_d = float(baseline_row["durability_score"])
        calculated_a = baseline_row["antibacterial_score"]
        audit_rows.append(
            {
                "formulation": meta.display_name,
                "calculated_durability": calculated_d,
                "manuscript_durability": manuscript["durability"],
                "durability_difference": calculated_d - float(manuscript["durability"]),
                "calculated_antibacterial": calculated_a,
                "manuscript_antibacterial": manuscript["antibacterial"],
                "antibacterial_difference": (
                    None
                    if calculated_a is None or manuscript["antibacterial"] is None
                    else float(calculated_a) - float(manuscript["antibacterial"])
                ),
            }
        )

        formulation_rows = [row for row in all_score_rows if row["key"] == key]
        durability_values = [float(row["durability_score"]) for row in formulation_rows]
        wie_values = [float(row["wie"]) for row in formulation_rows]
        wmax_values = [float(row["wmax"]) for row in formulation_rows]
        antibacterial_values = [
            float(row["antibacterial_score"])
            for row in formulation_rows
            if row["antibacterial_score"] is not None
        ]
        branches = sorted({str(row["durability_branch"]) for row in formulation_rows})
        range_rows.append(
            {
                "formulation": meta.display_name,
                "baseline_wie": baseline_row["wie"],
                "min_wie": min(wie_values),
                "max_wie": max(wie_values),
                "baseline_wmax": baseline_row["wmax"],
                "min_wmax": min(wmax_values),
                "max_wmax": max(wmax_values),
                "baseline_durability": baseline_row["durability_score"],
                "min_durability": min(durability_values),
                "max_durability": max(durability_values),
                "durability_range": max(durability_values) - min(durability_values),
                "baseline_antibacterial": baseline_row["antibacterial_score"],
                "min_antibacterial": min(antibacterial_values) if antibacterial_values else None,
                "max_antibacterial": max(antibacterial_values) if antibacterial_values else None,
                "antibacterial_range": (
                    max(antibacterial_values) - min(antibacterial_values)
                    if antibacterial_values
                    else None
                ),
                "durability_branches_encountered": " | ".join(branches),
            }
        )

    write_csv(
        output_dir / "baseline_scores.csv",
        baseline_rows,
        list(baseline_rows[0].keys()),
    )
    write_csv(
        output_dir / "baseline_vs_manuscript.csv",
        audit_rows,
        list(audit_rows[0].keys()),
    )
    write_csv(
        output_dir / "score_ranges.csv",
        range_rows,
        list(range_rows[0].keys()),
    )

    if not args.no_plots:
        create_heatmaps(
            scenario_scores,
            scenario_mcda,
            q10_values,
            alpha_values,
            output_dir,
        )

    write_readme(output_dir, args.input_xlsx, q10_values, alpha_values)

    print(f"Completed {len(q10_values) * len(alpha_values)} WIE scenarios.")
    print(f"Baseline used for audit: Q10={baseline_key[0]}, alpha={baseline_key[1]}")
    print(f"Results written to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
