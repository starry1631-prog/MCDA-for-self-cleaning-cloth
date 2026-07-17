
"""
Targeted Ag WIE sensitivity analysis

Both Ag formulations are assigned the same assumed WIE, varied from 1.00 down
to 0.18 in increments of 0.01. At each WIE, the script recalculates:

1. Ag durability scores from the raw retained-quantity observations;
2. Ag antibacterial scores from the raw species-specific observations;
3. Durability-Cost (DC) winning shares across 101 durability weights;
4. Durability-Antibacterial-Cost (DAC) winning shares across the 5,151-point
   three-criterion simplex;
5. Pareto-front membership.

The 2 <= Wmax < 10 durability branch is constructed to be continuous with the
existing Wmax < 2 evidence-coverage rule:

- At Wmax = 2, the observed evidence covers 15/16 of the estimated ten-wash
  cumulative loss.
- At Wmax = 10, the observed evidence covers the full ten-wash cumulative loss.
- Between these limits, the loss-coverage fraction is interpolated linearly.
- The interpolation applies to evidence coverage, not to the physical retention
  trajectory.

Raw Ag observations are copied from:
"Original Data Extracted From Sources(5).xlsx"

The other formulation-level normalized scores are copied from the current
manuscript Table 4.

Outputs:
- ag_wie_sensitivity_results.csv
- ag_wie_sensitivity_summary.txt
- ag_wie_sensitivity_dc.png
- ag_wie_sensitivity_dac.png
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent / "ag_wie_sensitivity_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_WASHES = 10.0
MIN_AG_WIE = 0.18
MAX_AG_WIE = 1.00
WIE_STEP = 0.01

DC_WEIGHT_STEP = 0.01
DAC_WEIGHT_STEP = 0.01

# The current manuscript dataset uses formulation-level normalized scores
# reported to three decimal places. Keeping this as 3 reproduces the current
# baseline winner shares (approximately 53.6% vs 46.4% in DAC).
# Set to None to use the full-precision recalculated Ag scores in the MCDA.
MCDA_SCORE_DECIMALS: int | None = 3


# ---------------------------------------------------------------------------
# 2. Raw Ag data from the uploaded Excel file
# ---------------------------------------------------------------------------

# Durability observations:
# (reported, unstandardized washing cycles, retained quantity)
AG_DURABILITY_RAW: Mapping[str, Sequence[Tuple[float, float]]] = {
    "Ag Two-Step": [
        (0.0, 180.0),
        (10.0, 165.0),
        (30.0, 148.0),
    ],
    "Ag One-Step": [
        (0.0, 1059.0),
        (20.0, 871.0),
    ],
}

# Antibacterial observations:
# (reported, unstandardized washing cycles, bacterial reduction %)
AG_ANTIBACTERIAL_RAW: Mapping[
    str, Mapping[str, Sequence[Tuple[float, float]]]
] = {
    "Ag Two-Step": {
        "E. coli": [
            (0.0, 99.99),
            (30.0, 98.92),
        ],
        "S. aureus": [
            (0.0, 99.99),
            (30.0, 99.08),
        ],
    },
    "Ag One-Step": {
        "E. coli": [
            (5.0, 95.87),
            (10.0, 93.59),
        ],
        "S. aureus": [
            (5.0, 94.59),
            (10.0, 92.23),
        ],
    },
}


# ---------------------------------------------------------------------------
# 3. Fixed formulation-level scores from current manuscript Table 4
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DCFormulation:
    name: str
    durability: float
    cost: float


@dataclass(frozen=True)
class DACFormulation:
    name: str
    durability: float
    antibacterial: float
    cost: float


FIXED_DC: Mapping[str, Tuple[float, float]] = {
    "TiO2 Two-Step Dipping": (0.459, 0.264),
    "TiO2 Alkaline Hydrolysis": (0.772, 0.327),
    "Hybrid Padding-Squeezing": (0.623, 1.000),
    "Hybrid In-Situ": (0.738, 0.496),
    "ZnO Starch": (0.842, 0.000),
    "ZnO SDS": (0.544, 0.377),
}

FIXED_DAC: Mapping[str, Tuple[float, float, float]] = {
    "Hybrid Padding-Squeezing": (0.623, 0.999, 1.000),
    "Hybrid In-Situ": (0.738, 0.993, 0.496),
    "ZnO Starch": (0.842, 0.395, 0.000),
    "ZnO SDS": (0.544, 0.802, 0.377),
}

AG_COST_SCORES: Mapping[str, float] = {
    "Ag Two-Step": 0.663,
    "Ag One-Step": 0.203,
}


# ---------------------------------------------------------------------------
# 4. Score-recalculation functions
# ---------------------------------------------------------------------------

def clip_score(value: float) -> float:
    """Restrict a normalized score to the interval [0, 1]."""
    return float(np.clip(value, 0.0, 1.0))


def maybe_round_score(value: float) -> float:
    """Optionally round a score before it enters the MCDA."""
    if MCDA_SCORE_DECIMALS is None:
        return value
    return round(value, MCDA_SCORE_DECIMALS)


def linear_interpolate(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    target_x: float,
) -> float:
    """Linear interpolation or extrapolation through two points."""
    if math.isclose(x1, x2):
        raise ValueError("Cannot interpolate using identical x values.")
    return y1 + (y2 - y1) * (target_x - x1) / (x2 - x1)


def ten_wash_loss_coverage(
    wmax: float,
) -> float:
    """
    Evidence-coverage fraction for 2 <= Wmax < 10.

    At Wmax = 2:
        coverage = 15/16,
    because the first two washes represent 75/80 of estimated ten-wash loss.

    At Wmax = 10:
        coverage = 1.

    The interpolation is applied to evidence coverage, not to the physical
    retention curve.
    """
    if not 2.0 <= wmax < 10.0:
        raise ValueError("This function requires 2 <= Wmax < 10.")

    coverage_at_2 = 15.0 / 16.0
    fraction_between_2_and_10 = (wmax - 2.0) / 8.0

    return (
        coverage_at_2
        + (1.0 - coverage_at_2) * fraction_between_2_and_10
    )


def calculate_durability_score(
    observations: Sequence[Tuple[float, float]],
    assumed_wie: float,
) -> Tuple[float, float, str]:
    """
    Recalculate durability at the ten-reference-wash horizon.

    Branches:
    1. Wmax >= 10:
       obtain the retained quantity at 10 by direct observation or interpolation.

    2. 2 <= Wmax < 10:
       convert the observed retained fraction into an evidence-adjusted ten-wash
       score using the interpolated ten-wash loss-coverage fraction:

           coverage(W) = 15/16 + (1 - 15/16) * (W - 2) / 8

           S_R = 1 - (1 - Rmax) / coverage(W)

       This is continuous with both adjacent branches:
       - at W = 2, S_R = 1 - (16/15)(1 - Rmax);
       - at W = 10, S_R = Rmax.

    3. Wmax < 2:
       retain the existing near-threshold evidence-penalty formula:

           S_R = 1 - (16/15) *
                 [1 - Rmax * min(Wmax / 2, 1)].
    """
    ordered = sorted(observations)
    standardized_cycles = np.asarray(
        [cycle * assumed_wie for cycle, _ in ordered],
        dtype=float,
    )
    retained_quantities = np.asarray(
        [value for _, value in ordered],
        dtype=float,
    )

    initial_quantity = retained_quantities[0]
    if initial_quantity <= 0:
        raise ValueError("Initial retained quantity must be positive.")

    wmax = float(standardized_cycles[-1])
    rmax = float(retained_quantities[-1] / initial_quantity)

    if wmax >= TARGET_WASHES:
        retained_at_10 = float(
            np.interp(
                TARGET_WASHES,
                standardized_cycles,
                retained_quantities,
            )
        )
        score = retained_at_10 / initial_quantity
        branch = "direct/interpolated at 10"

    elif wmax >= 2.0:
        coverage = ten_wash_loss_coverage(wmax)
        score = 1.0 - (1.0 - rmax) / coverage
        branch = "2-10 evidence coverage"

    else:
        initial_threshold_coverage = min(wmax / 2.0, 1.0)
        score = 1.0 - (16.0 / 15.0) * (
            1.0 - rmax * initial_threshold_coverage
        )
        branch = "<2 evidence penalty"

    return clip_score(score), wmax, branch


def calculate_species_antibacterial_value(
    observations: Sequence[Tuple[float, float]],
    assumed_wie: float,
) -> Tuple[float, str]:
    """
    Recalculate a species-specific bacterial-reduction value at ten washes.

    - Direct observation or linear interpolation is used when 10 lies within
      the standardized reported range.
    - Linear extrapolation from the final two points is used when the reported
      range ends before 10.
    - Values are restricted to 0-100%.
    """
    ordered = sorted(observations)
    standardized_cycles = np.asarray(
        [cycle * assumed_wie for cycle, _ in ordered],
        dtype=float,
    )
    reduction_percentages = np.asarray(
        [value for _, value in ordered],
        dtype=float,
    )

    if len(standardized_cycles) < 2:
        raise ValueError(
            "At least two antibacterial observations are required."
        )

    exact_indices = np.where(
        np.isclose(
            standardized_cycles,
            TARGET_WASHES,
            atol=1e-12,
        )
    )[0]

    if exact_indices.size:
        estimate = float(reduction_percentages[exact_indices[0]])
        branch = "direct at 10"

    elif (
        standardized_cycles[0]
        < TARGET_WASHES
        < standardized_cycles[-1]
    ):
        estimate = float(
            np.interp(
                TARGET_WASHES,
                standardized_cycles,
                reduction_percentages,
            )
        )
        branch = "interpolated at 10"

    elif TARGET_WASHES > standardized_cycles[-1]:
        estimate = linear_interpolate(
            standardized_cycles[-2],
            reduction_percentages[-2],
            standardized_cycles[-1],
            reduction_percentages[-1],
            TARGET_WASHES,
        )
        branch = "extrapolated to 10"

    else:
        estimate = linear_interpolate(
            standardized_cycles[0],
            reduction_percentages[0],
            standardized_cycles[1],
            reduction_percentages[1],
            TARGET_WASHES,
        )
        branch = "back-extrapolated to 10"

    return float(np.clip(estimate, 0.0, 100.0)), branch


def calculate_antibacterial_score(
    species_data: Mapping[str, Sequence[Tuple[float, float]]],
    assumed_wie: float,
) -> Tuple[float, Dict[str, float], Dict[str, str]]:
    """Calculate the mean normalized antibacterial score across both species."""
    species_values: Dict[str, float] = {}
    species_branches: Dict[str, str] = {}

    for species, observations in species_data.items():
        value_percent, branch = calculate_species_antibacterial_value(
            observations,
            assumed_wie,
        )
        species_values[species] = value_percent / 100.0
        species_branches[species] = branch

    overall = float(np.mean(list(species_values.values())))
    return clip_score(overall), species_values, species_branches


# ---------------------------------------------------------------------------
# 5. Weight spaces
# ---------------------------------------------------------------------------

def generate_dc_weights(step: float = DC_WEIGHT_STEP) -> np.ndarray:
    divisions = round(1.0 / step)
    if not math.isclose(divisions * step, 1.0, abs_tol=1e-12):
        raise ValueError("DC weight step must divide 1 exactly.")
    return np.arange(divisions + 1, dtype=float) / divisions


def generate_dac_weights(step: float = DAC_WEIGHT_STEP) -> np.ndarray:
    divisions = round(1.0 / step)
    if not math.isclose(divisions * step, 1.0, abs_tol=1e-12):
        raise ValueError("DAC weight step must divide 1 exactly.")

    rows: List[Tuple[float, float, float]] = []
    for durability_integer in range(divisions + 1):
        for antibacterial_integer in range(
            divisions + 1 - durability_integer
        ):
            w_d = durability_integer / divisions
            w_a = antibacterial_integer / divisions
            w_c = 1.0 - w_d - w_a
            rows.append((w_d, w_a, w_c))

    return np.asarray(rows, dtype=float)


def calculate_winning_shares(
    criteria_matrix: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Calculate winning shares and split exact ties equally."""
    scores = weights @ criteria_matrix.T
    maxima = scores.max(axis=1, keepdims=True)
    tied = np.isclose(scores, maxima, atol=1e-12, rtol=0.0)
    fractional_wins = tied / tied.sum(axis=1, keepdims=True)
    return fractional_wins.sum(axis=0) / len(weights) * 100.0


def pareto_optimal_names(
    names: Sequence[str],
    criteria_matrix: np.ndarray,
) -> List[str]:
    """Return non-dominated formulations; all criteria are higher-is-better."""
    optimal: List[str] = []

    for candidate_index, candidate_name in enumerate(names):
        candidate = criteria_matrix[candidate_index]
        dominated = False

        for comparison_index in range(len(names)):
            if comparison_index == candidate_index:
                continue

            comparison = criteria_matrix[comparison_index]
            at_least_as_good = np.all(
                comparison >= candidate - 1e-12
            )
            strictly_better = np.any(
                comparison > candidate + 1e-12
            )

            if at_least_as_good and strictly_better:
                dominated = True
                break

        if not dominated:
            optimal.append(candidate_name)

    return optimal


# ---------------------------------------------------------------------------
# 6. Build each WIE scenario
# ---------------------------------------------------------------------------

def build_wie_values() -> List[float]:
    """Return 1.00, 0.99, ..., 0.18."""
    maximum_integer = round(MAX_AG_WIE * 100)
    minimum_integer = round(MIN_AG_WIE * 100)
    return [
        integer_value / 100.0
        for integer_value in range(
            maximum_integer,
            minimum_integer - 1,
            -1,
        )
    ]


def analyse_scenario(
    assumed_wie: float,
    dc_weights: np.ndarray,
    dac_weights: np.ndarray,
) -> Dict[str, object]:
    ag_durability_scores: Dict[str, float] = {}
    ag_antibacterial_scores: Dict[str, float] = {}
    ag_wmax: Dict[str, float] = {}
    ag_durability_branches: Dict[str, str] = {}

    for formulation in ("Ag Two-Step", "Ag One-Step"):
        durability, wmax, durability_branch = (
            calculate_durability_score(
                AG_DURABILITY_RAW[formulation],
                assumed_wie,
            )
        )
        antibacterial, _, _ = calculate_antibacterial_score(
            AG_ANTIBACTERIAL_RAW[formulation],
            assumed_wie,
        )

        ag_durability_scores[formulation] = maybe_round_score(
            durability
        )
        ag_antibacterial_scores[formulation] = maybe_round_score(
            antibacterial
        )
        ag_wmax[formulation] = wmax
        ag_durability_branches[formulation] = durability_branch

    # DC dataset
    dc_names = list(FIXED_DC.keys()) + [
        "Ag Two-Step",
        "Ag One-Step",
    ]
    dc_rows = [
        [FIXED_DC[name][0], FIXED_DC[name][1]]
        for name in FIXED_DC
    ]
    dc_rows.extend(
        [
            [
                ag_durability_scores["Ag Two-Step"],
                AG_COST_SCORES["Ag Two-Step"],
            ],
            [
                ag_durability_scores["Ag One-Step"],
                AG_COST_SCORES["Ag One-Step"],
            ],
        ]
    )
    dc_criteria = np.asarray(dc_rows, dtype=float)

    # Convert durability weights into [w_D, w_C].
    dc_weight_matrix = np.column_stack(
        [dc_weights, 1.0 - dc_weights]
    )
    dc_shares_array = calculate_winning_shares(
        dc_criteria,
        dc_weight_matrix,
    )
    dc_shares = {
        name: float(dc_shares_array[index])
        for index, name in enumerate(dc_names)
    }
    dc_pareto = pareto_optimal_names(
        dc_names,
        dc_criteria,
    )

    # DAC dataset
    dac_names = list(FIXED_DAC.keys()) + [
        "Ag Two-Step",
        "Ag One-Step",
    ]
    dac_rows = [
        [
            FIXED_DAC[name][0],
            FIXED_DAC[name][1],
            FIXED_DAC[name][2],
        ]
        for name in FIXED_DAC
    ]
    dac_rows.extend(
        [
            [
                ag_durability_scores["Ag Two-Step"],
                ag_antibacterial_scores["Ag Two-Step"],
                AG_COST_SCORES["Ag Two-Step"],
            ],
            [
                ag_durability_scores["Ag One-Step"],
                ag_antibacterial_scores["Ag One-Step"],
                AG_COST_SCORES["Ag One-Step"],
            ],
        ]
    )
    dac_criteria = np.asarray(dac_rows, dtype=float)

    dac_shares_array = calculate_winning_shares(
        dac_criteria,
        dac_weights,
    )
    dac_shares = {
        name: float(dac_shares_array[index])
        for index, name in enumerate(dac_names)
    }
    dac_pareto = pareto_optimal_names(
        dac_names,
        dac_criteria,
    )

    return {
        "WIE": assumed_wie,
        "Ag Two-Step Wmax": ag_wmax["Ag Two-Step"],
        "Ag One-Step Wmax": ag_wmax["Ag One-Step"],
        "Ag Two-Step durability": (
            ag_durability_scores["Ag Two-Step"]
        ),
        "Ag One-Step durability": (
            ag_durability_scores["Ag One-Step"]
        ),
        "Ag Two-Step antibacterial": (
            ag_antibacterial_scores["Ag Two-Step"]
        ),
        "Ag One-Step antibacterial": (
            ag_antibacterial_scores["Ag One-Step"]
        ),
        "Ag Two-Step durability branch": (
            ag_durability_branches["Ag Two-Step"]
        ),
        "Ag One-Step durability branch": (
            ag_durability_branches["Ag One-Step"]
        ),
        "DC shares": dc_shares,
        "DAC shares": dac_shares,
        "DC Pareto": dc_pareto,
        "DAC Pareto": dac_pareto,
    }


# ---------------------------------------------------------------------------
# 7. Threshold extraction
# ---------------------------------------------------------------------------

def highest_wie_with_positive_share(
    results: Sequence[Mapping[str, object]],
    model_key: str,
    formulation: str,
) -> float | None:
    """Highest tested WIE at which a formulation has a positive winning share."""
    matching = [
        float(result["WIE"])
        for result in results
        if float(
            result[model_key][formulation]  # type: ignore[index]
        ) > 1e-12
    ]
    return max(matching) if matching else None


def first_pareto_change(
    results: Sequence[Mapping[str, object]],
    pareto_key: str,
) -> Tuple[float | None, List[str] | None]:
    """First descending-WIE scenario whose Pareto front differs from baseline."""
    baseline = list(results[0][pareto_key])  # type: ignore[arg-type]

    for result in results[1:]:
        current = list(result[pareto_key])  # type: ignore[arg-type]
        if current != baseline:
            return float(result["WIE"]), current

    return None, None


# ---------------------------------------------------------------------------
# 8. CSV, summary and figures
# ---------------------------------------------------------------------------

def write_results_csv(
    results: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    formulation_order_dc = list(FIXED_DC.keys()) + [
        "Ag Two-Step",
        "Ag One-Step",
    ]
    formulation_order_dac = list(FIXED_DAC.keys()) + [
        "Ag Two-Step",
        "Ag One-Step",
    ]

    fieldnames = [
        "Assumed Ag WIE",
        "Ag Two-Step Wmax",
        "Ag One-Step Wmax",
        "Ag Two-Step durability",
        "Ag One-Step durability",
        "Ag Two-Step antibacterial",
        "Ag One-Step antibacterial",
        "Ag Two-Step durability branch",
        "Ag One-Step durability branch",
        "DC Pareto front",
        "DAC Pareto front",
    ]
    fieldnames.extend(
        [f"DC share - {name}" for name in formulation_order_dc]
    )
    fieldnames.extend(
        [f"DAC share - {name}" for name in formulation_order_dac]
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row: Dict[str, object] = {
                "Assumed Ag WIE": f"{float(result['WIE']):.2f}",
                "Ag Two-Step Wmax": (
                    f"{float(result['Ag Two-Step Wmax']):.6f}"
                ),
                "Ag One-Step Wmax": (
                    f"{float(result['Ag One-Step Wmax']):.6f}"
                ),
                "Ag Two-Step durability": (
                    f"{float(result['Ag Two-Step durability']):.6f}"
                ),
                "Ag One-Step durability": (
                    f"{float(result['Ag One-Step durability']):.6f}"
                ),
                "Ag Two-Step antibacterial": (
                    f"{float(result['Ag Two-Step antibacterial']):.6f}"
                ),
                "Ag One-Step antibacterial": (
                    f"{float(result['Ag One-Step antibacterial']):.6f}"
                ),
                "Ag Two-Step durability branch": result[
                    "Ag Two-Step durability branch"
                ],
                "Ag One-Step durability branch": result[
                    "Ag One-Step durability branch"
                ],
                "DC Pareto front": "; ".join(
                    result["DC Pareto"]  # type: ignore[arg-type]
                ),
                "DAC Pareto front": "; ".join(
                    result["DAC Pareto"]  # type: ignore[arg-type]
                ),
            }

            dc_shares = result["DC shares"]  # type: ignore[assignment]
            dac_shares = result["DAC shares"]  # type: ignore[assignment]

            for name in formulation_order_dc:
                row[f"DC share - {name}"] = (
                    f"{float(dc_shares[name]):.6f}"
                )

            for name in formulation_order_dac:
                row[f"DAC share - {name}"] = (
                    f"{float(dac_shares[name]):.6f}"
                )

            writer.writerow(row)


def plot_winning_shares(
    results: Sequence[Mapping[str, object]],
    model_key: str,
    formulation_names: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    wie_values = np.asarray(
        [float(result["WIE"]) for result in results],
        dtype=float,
    )

    figure = plt.figure(figsize=(9.0, 6.0))
    axis = figure.add_subplot(111)

    for formulation in formulation_names:
        shares = np.asarray(
            [
                float(
                    result[model_key][formulation]  # type: ignore[index]
                )
                for result in results
            ],
            dtype=float,
        )
        axis.plot(
            wie_values,
            shares,
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=formulation,
        )

    axis.set_title(title)
    axis.set_xlabel("Assumed WIE for both Ag formulations")
    axis.set_ylabel("Winning weight-space share (%)")
    axis.set_ylim(bottom=0)
    axis.invert_xaxis()
    axis.grid(True, linewidth=0.5)
    axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_summary(
    results: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    baseline = results[0]
    minimum = results[-1]

    dc_zno_entry = highest_wie_with_positive_share(
        results,
        "DC shares",
        "ZnO Starch",
    )
    dac_zno_entry = highest_wie_with_positive_share(
        results,
        "DAC shares",
        "ZnO Starch",
    )
    dac_hybrid_in_situ_entry = highest_wie_with_positive_share(
        results,
        "DAC shares",
        "Hybrid In-Situ",
    )

    dc_pareto_wie, dc_pareto_front = first_pareto_change(
        results,
        "DC Pareto",
    )
    dac_pareto_wie, dac_pareto_front = first_pareto_change(
        results,
        "DAC Pareto",
    )

    minimum_dc = minimum["DC shares"]  # type: ignore[assignment]
    minimum_dac = minimum["DAC shares"]  # type: ignore[assignment]

    ag_two_step_positive_dc = all(
        float(
            result["DC shares"]["Ag Two-Step"]  # type: ignore[index]
        ) > 1e-12
        for result in results
    )
    ag_two_step_positive_dac = all(
        float(
            result["DAC shares"]["Ag Two-Step"]  # type: ignore[index]
        ) > 1e-12
        for result in results
    )

    lines = [
        "Targeted Ag WIE sensitivity analysis",
        "",
        (
            f"Tested simultaneous Ag WIE range: "
            f"{MAX_AG_WIE:.2f} to {MIN_AG_WIE:.2f}"
        ),
        (
            f"Number of WIE scenarios: {len(results)}"
        ),
        "",
        "Baseline WIE = 1.00:",
        (
            "  DC shares: "
            f"Hybrid Padding-Squeezing "
            f"{baseline['DC shares']['Hybrid Padding-Squeezing']:.3f}%, "
            f"Ag Two-Step "
            f"{baseline['DC shares']['Ag Two-Step']:.3f}%."
        ),
        (
            "  DAC shares: "
            f"Hybrid Padding-Squeezing "
            f"{baseline['DAC shares']['Hybrid Padding-Squeezing']:.3f}%, "
            f"Ag Two-Step "
            f"{baseline['DAC shares']['Ag Two-Step']:.3f}%."
        ),
        "",
        f"At WIE = {MIN_AG_WIE:.2f}:",
        (
            "  Ag Two-Step scores: "
            f"durability "
            f"{minimum['Ag Two-Step durability']:.3f}, "
            f"antibacterial "
            f"{minimum['Ag Two-Step antibacterial']:.3f}."
        ),
        (
            "  Ag One-Step scores: "
            f"durability "
            f"{minimum['Ag One-Step durability']:.3f}, "
            f"antibacterial "
            f"{minimum['Ag One-Step antibacterial']:.3f}."
        ),
        (
            "  DC shares: "
            f"Hybrid Padding-Squeezing "
            f"{minimum_dc['Hybrid Padding-Squeezing']:.3f}%, "
            f"Ag Two-Step "
            f"{minimum_dc['Ag Two-Step']:.3f}%, "
            f"ZnO Starch "
            f"{minimum_dc['ZnO Starch']:.3f}%."
        ),
        (
            "  DAC shares: "
            f"Hybrid Padding-Squeezing "
            f"{minimum_dac['Hybrid Padding-Squeezing']:.3f}%, "
            f"Ag Two-Step "
            f"{minimum_dac['Ag Two-Step']:.3f}%, "
            f"Hybrid In-Situ "
            f"{minimum_dac['Hybrid In-Situ']:.3f}%, "
            f"ZnO Starch "
            f"{minimum_dac['ZnO Starch']:.3f}%."
        ),
        "",
        (
            f"Highest tested WIE at which ZnO Starch first has "
            f"a positive DC winning region: {dc_zno_entry}"
        ),
        (
            f"Highest tested WIE at which ZnO Starch first has "
            f"a positive DAC winning region: {dac_zno_entry}"
        ),
        (
            f"Highest tested WIE at which Hybrid In-Situ first has "
            f"a positive DAC winning region: "
            f"{dac_hybrid_in_situ_entry}"
        ),
        (
            f"First DC Pareto-front change: WIE = {dc_pareto_wie}; "
            f"{dc_pareto_front}"
        ),
        (
            f"First DAC Pareto-front change: WIE = {dac_pareto_wie}; "
            f"{dac_pareto_front}"
        ),
        "",
        (
            "Ag Two-Step retains a positive DC winning region at "
            f"every tested WIE: {ag_two_step_positive_dc}"
        ),
        (
            "Ag Two-Step retains a positive DAC winning region at "
            f"every tested WIE: {ag_two_step_positive_dac}"
        ),
    ]

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main() -> None:
    dc_weights = generate_dc_weights()
    dac_weights = generate_dac_weights()

    results = [
        analyse_scenario(
            assumed_wie,
            dc_weights,
            dac_weights,
        )
        for assumed_wie in build_wie_values()
    ]

    csv_path = OUTPUT_DIR / "ag_wie_sensitivity_results.csv"
    summary_path = OUTPUT_DIR / "ag_wie_sensitivity_summary.txt"
    dc_figure_path = OUTPUT_DIR / "ag_wie_sensitivity_dc.png"
    dac_figure_path = OUTPUT_DIR / "ag_wie_sensitivity_dac.png"

    write_results_csv(results, csv_path)
    write_summary(results, summary_path)

    plot_winning_shares(
        results=results,
        model_key="DC shares",
        formulation_names=[
            "Hybrid Padding-Squeezing",
            "Ag Two-Step",
            "ZnO Starch",
            "TiO2 Alkaline Hydrolysis",
            "Hybrid In-Situ",
            "Ag One-Step",
        ],
        title=(
            "Durability-Cost winning shares under assumed Ag WIE"
        ),
        output_path=dc_figure_path,
    )

    plot_winning_shares(
        results=results,
        model_key="DAC shares",
        formulation_names=[
            "Hybrid Padding-Squeezing",
            "Ag Two-Step",
            "Hybrid In-Situ",
            "ZnO Starch",
            "ZnO SDS",
            "Ag One-Step",
        ],
        title=(
            "Durability-Antibacterial-Cost winning shares "
            "under assumed Ag WIE"
        ),
        output_path=dac_figure_path,
    )

    print(summary_path.read_text(encoding="utf-8"))
    print()
    print(f"CSV: {csv_path.resolve()}")
    print(f"DC figure: {dc_figure_path.resolve()}")
    print(f"DAC figure: {dac_figure_path.resolve()}")


if __name__ == "__main__":
    main()
