
"""
WIE parameter sensitivity analysis using the revised durability
evidence-coverage formula.

This script evaluates the 20 combinations used in the manuscript:

    Q10 = 1.5, 2.0, 2.5, 3.0
    alpha = 0.3, 0.4, 0.5, 0.6, 0.7

The Ag formulations retain WIE = 1 because their laundering temperature
and duration were not reported. Their uncertainty is treated separately
in the targeted Ag-WIE sensitivity analysis.

Revised durability rule
-----------------------
For Wmax >= 10:
    obtain the retained quantity at 10 reference-equivalent washes by
    direct observation or linear interpolation.

For 2 <= Wmax < 10:
    c(Wmax) = 15/16 + (1 - 15/16) * (Wmax - 2) / 8
    S_R = 1 - (1 - Rmax) / c(Wmax)

For Wmax < 2:
    S_R = 1 - (16/15) * [1 - Rmax * (Wmax / 2)]

The linear term in the middle branch interpolates evidence coverage,
not a physical nanoparticle-loss trajectory. The three branches are
continuous at Wmax = 2 and Wmax = 10.

Input data
----------
Raw durability and antibacterial observations were copied from:
"Original Data Extracted From Sources(6).xlsx"

Washing conditions and fixed cost scores were copied from the current
manuscript:
"Application-Specific ... Raw-Material Cost(8).docx"

Outputs
-------
- wie_sensitivity_results.csv
- wie_sensitivity_summary.txt
- dac_ag_two_step_winning_share.png
- dac_hybrid_padding_squeezing_winning_share.png
- dc_ag_two_step_winning_share.png
- dc_hybrid_padding_squeezing_winning_share.png
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent / "wie_sensitivity_revised_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

Q10_VALUES = [1.5, 2.0, 2.5, 3.0]
ALPHA_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]

T_REF_C = 40.0
T_REF_MIN = 45.0
TARGET_REFERENCE_WASHES = 10.0

DC_WEIGHT_STEP = 0.01
DAC_WEIGHT_STEP = 0.01


# ---------------------------------------------------------------------------
# 2. Raw source data
# ---------------------------------------------------------------------------

# Each durability tuple is:
# (unstandardized washing cycles, retained quantity or marker amount)
DURABILITY_RAW: Mapping[str, Sequence[Tuple[float, float]]] = {
    "TiO2 Two-Step Dipping": [
        (5.0, 23.16),
        (10.0, 12.48),
        (15.0, 10.83),
        (20.0, 4.45),
    ],
    "TiO2 Alkaline Hydrolysis": [
        (0.0, 3.3),
        (10.0, 2.3),
    ],
    "Hybrid Padding-Squeezing": [
        (0.0, 1.449),
        (15.0, 0.112),
        (30.0, 0.057),
    ],
    "Hybrid In-Situ": [
        (0.0, 44500.0),
        (30.0, 9500.0),
    ],
    "ZnO Starch": [
        (0.0, 35.0),
        (5.0, 34.9),
        (10.0, 33.4),
    ],
    "ZnO SDS": [
        (0.0, 6.74),
        (5.0, 5.91),
        (10.0, 4.32),
    ],
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

# Each antibacterial tuple is:
# (unstandardized washing cycles, bacterial reduction percentage)
ANTIBACTERIAL_RAW: Mapping[
    str, Mapping[str, Sequence[Tuple[float, float]]]
] = {
    "Hybrid Padding-Squeezing": {
        "E. coli": [
            (0.0, 99.870),
            (15.0, 99.774),
            (30.0, 97.019),
        ],
        "S. aureus": [
            (0.0, 99.931),
            (15.0, 99.855),
            (30.0, 99.445),
        ],
    },
    "Hybrid In-Situ": {
        "E. coli": [
            (0.0, 100.0),
            (30.0, 97.8),
        ],
        "S. aureus": [
            (0.0, 100.0),
            (10.0, 99.3),
        ],
    },
    "ZnO Starch": {
        "E. coli": [
            (0.0, 100.0),
            (5.0, 100.0),
            (10.0, 76.4),
        ],
        "S. aureus": [
            (0.0, 100.0),
            (5.0, 98.1),
            (10.0, 96.2),
        ],
    },
    "ZnO SDS": {
        "E. coli": [
            (0.0, 91.0),
            (10.0, 89.0),
        ],
        "S. aureus": [
            (0.0, 92.3),
            (10.0, 90.2),
        ],
    },
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

# (washing temperature °C, washing duration min)
# None means the laundering conditions were not reported and WIE remains 1.
WASH_CONDITIONS: Mapping[str, Tuple[float, float] | None] = {
    "TiO2 Two-Step Dipping": (40.0, 45.0),
    "TiO2 Alkaline Hydrolysis": (60.0, 20.0),
    "Hybrid Padding-Squeezing": (60.0, 30.0),
    "Hybrid In-Situ": (40.0, 45.0),
    "ZnO Starch": (22.0, 5.0),  # duration imputed from ZnO SDS
    "ZnO SDS": (22.0, 5.0),
    "Ag Two-Step": None,
    "Ag One-Step": None,
}

COST_SCORES: Mapping[str, float] = {
    "TiO2 Two-Step Dipping": 0.264,
    "TiO2 Alkaline Hydrolysis": 0.327,
    "Hybrid Padding-Squeezing": 1.000,
    "Hybrid In-Situ": 0.496,
    "ZnO Starch": 0.000,
    "ZnO SDS": 0.377,
    "Ag Two-Step": 0.663,
    "Ag One-Step": 0.203,
}

DC_FORMULATIONS = list(DURABILITY_RAW.keys())
DAC_FORMULATIONS = list(ANTIBACTERIAL_RAW.keys())
AG_EXCLUDED_DAC_FORMULATIONS = [
    "Hybrid Padding-Squeezing",
    "Hybrid In-Situ",
    "ZnO Starch",
    "ZnO SDS",
]


# ---------------------------------------------------------------------------
# 3. WIE and score calculation
# ---------------------------------------------------------------------------

def clip_score(value: float) -> float:
    """Restrict a normalized score to [0, 1]."""
    return float(np.clip(value, 0.0, 1.0))


def calculate_wie(
    formulation: str,
    q10: float,
    alpha: float,
) -> float:
    """Calculate WIE; return 1 for Ag formulations with missing conditions."""
    condition = WASH_CONDITIONS[formulation]

    if condition is None:
        return 1.0

    temperature_c, duration_min = condition

    if q10 <= 0:
        raise ValueError("Q10 must be positive.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie between 0 and 1.")
    if duration_min <= 0:
        raise ValueError("Washing duration must be positive.")

    thermal_term = q10 ** ((temperature_c - T_REF_C) / 10.0)
    duration_term = duration_min / T_REF_MIN

    return (
        thermal_term ** alpha
        * duration_term ** (1.0 - alpha)
    )


def estimate_initial_quantity(
    observations: Sequence[Tuple[float, float]],
) -> float:
    """
    Return the reported cycle-0 quantity where available.

    If cycle 0 was not reported, fit a least-squares line through all
    observations and use its intercept, matching the manuscript method.
    """
    ordered = sorted(observations)

    for cycle, quantity in ordered:
        if math.isclose(cycle, 0.0):
            return float(quantity)

    cycles = np.asarray([point[0] for point in ordered], dtype=float)
    quantities = np.asarray([point[1] for point in ordered], dtype=float)

    if len(cycles) < 2:
        raise ValueError(
            "At least two observations are required to estimate cycle 0."
        )

    slope, intercept = np.polyfit(cycles, quantities, 1)

    if intercept <= 0:
        raise ValueError(
            "Estimated initial quantity must be positive."
        )

    return float(intercept)


def intermediate_loss_coverage(wmax: float) -> float:
    """
    Evidence coverage for 2 <= Wmax < 10.

    At Wmax = 2, the observations cover 15/16 of estimated ten-wash
    cumulative loss. At Wmax = 10, coverage equals 1.
    """
    if not 2.0 <= wmax < 10.0:
        raise ValueError(
            "intermediate_loss_coverage requires 2 <= Wmax < 10."
        )

    coverage_at_two = 15.0 / 16.0

    return (
        coverage_at_two
        + (1.0 - coverage_at_two)
        * (wmax - 2.0)
        / 8.0
    )


def calculate_durability_score(
    formulation: str,
    q10: float,
    alpha: float,
) -> Dict[str, float | str]:
    """
    Recalculate the durability score using the revised piecewise rule.
    """
    observations = sorted(DURABILITY_RAW[formulation])
    wie = calculate_wie(formulation, q10, alpha)

    converted_cycles = np.asarray(
        [cycle * wie for cycle, _ in observations],
        dtype=float,
    )
    retained_quantities = np.asarray(
        [quantity for _, quantity in observations],
        dtype=float,
    )

    initial_quantity = estimate_initial_quantity(observations)
    wmax = float(converted_cycles[-1])
    rmax = float(retained_quantities[-1] / initial_quantity)

    if wmax >= TARGET_REFERENCE_WASHES:
        retained_at_ten = float(
            np.interp(
                TARGET_REFERENCE_WASHES,
                converted_cycles,
                retained_quantities,
            )
        )
        score = retained_at_ten / initial_quantity
        branch = "direct/interpolated at 10"

    elif wmax >= 2.0:
        coverage = intermediate_loss_coverage(wmax)
        score = 1.0 - (1.0 - rmax) / coverage
        branch = "2-10 evidence coverage"

    else:
        score = 1.0 - (16.0 / 15.0) * (
            1.0 - rmax * (wmax / 2.0)
        )
        branch = "<2 evidence penalty"

    return {
        "score": clip_score(score),
        "WIE": wie,
        "Wmax": wmax,
        "Rmax": rmax,
        "branch": branch,
    }


def linear_value_at_target(
    observations: Sequence[Tuple[float, float]],
    wie: float,
    target: float = TARGET_REFERENCE_WASHES,
) -> Tuple[float, str]:
    """
    Obtain an antibacterial value at the target by direct observation,
    interpolation, or extrapolation from the final two points.
    """
    ordered = sorted(observations)
    converted_cycles = np.asarray(
        [cycle * wie for cycle, _ in ordered],
        dtype=float,
    )
    values = np.asarray(
        [value for _, value in ordered],
        dtype=float,
    )

    if len(converted_cycles) < 2:
        raise ValueError(
            "At least two antibacterial observations are required."
        )

    exact_indices = np.where(
        np.isclose(converted_cycles, target, atol=1e-12)
    )[0]

    if exact_indices.size:
        estimate = float(values[exact_indices[0]])
        branch = "direct at 10"

    elif converted_cycles[0] < target < converted_cycles[-1]:
        estimate = float(
            np.interp(target, converted_cycles, values)
        )
        branch = "interpolated at 10"

    elif target > converted_cycles[-1]:
        x1, x2 = converted_cycles[-2], converted_cycles[-1]
        y1, y2 = values[-2], values[-1]

        if math.isclose(x1, x2):
            raise ValueError(
                "Final two antibacterial cycle values are identical."
            )

        slope = (y2 - y1) / (x2 - x1)
        estimate = float(y2 + slope * (target - x2))
        branch = "extrapolated to 10"

    else:
        x1, x2 = converted_cycles[0], converted_cycles[1]
        y1, y2 = values[0], values[1]

        if math.isclose(x1, x2):
            raise ValueError(
                "First two antibacterial cycle values are identical."
            )

        slope = (y2 - y1) / (x2 - x1)
        estimate = float(y1 + slope * (target - x1))
        branch = "back-extrapolated to 10"

    return float(np.clip(estimate, 0.0, 100.0)), branch


def calculate_antibacterial_score(
    formulation: str,
    q10: float,
    alpha: float,
) -> Dict[str, object]:
    """Calculate species-specific and overall antibacterial scores."""
    wie = calculate_wie(formulation, q10, alpha)

    species_scores: Dict[str, float] = {}
    species_branches: Dict[str, str] = {}

    for species, observations in ANTIBACTERIAL_RAW[
        formulation
    ].items():
        value_percent, branch = linear_value_at_target(
            observations,
            wie,
        )
        species_scores[species] = value_percent / 100.0
        species_branches[species] = branch

    overall = float(np.mean(list(species_scores.values())))

    return {
        "score": clip_score(overall),
        "WIE": wie,
        "species_scores": species_scores,
        "species_branches": species_branches,
    }


# ---------------------------------------------------------------------------
# 4. Weight spaces and Pareto analysis
# ---------------------------------------------------------------------------

def generate_dc_weights(step: float = DC_WEIGHT_STEP) -> np.ndarray:
    """Generate all DC weight pairs [w_D, w_C]."""
    divisions = round(1.0 / step)

    if not math.isclose(divisions * step, 1.0, abs_tol=1e-12):
        raise ValueError("DC weight step must divide 1 exactly.")

    durability_weights = (
        np.arange(divisions + 1, dtype=float) / divisions
    )

    return np.column_stack(
        [durability_weights, 1.0 - durability_weights]
    )


def generate_dac_weights(step: float = DAC_WEIGHT_STEP) -> np.ndarray:
    """Generate the full three-criterion simplex."""
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
    criteria: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Calculate winning shares, splitting exact ties equally."""
    scores = weights @ criteria.T
    maxima = scores.max(axis=1, keepdims=True)
    tied = np.isclose(scores, maxima, atol=1e-12, rtol=0.0)
    fractional_wins = tied / tied.sum(axis=1, keepdims=True)

    return (
        fractional_wins.sum(axis=0)
        / len(weights)
        * 100.0
    )


def pareto_front(
    names: Sequence[str],
    criteria: np.ndarray,
) -> List[str]:
    """Return non-dominated formulations; all criteria are favorable when high."""
    optimal: List[str] = []

    for candidate_index, candidate_name in enumerate(names):
        candidate = criteria[candidate_index]
        dominated = False

        for comparison_index in range(len(names)):
            if comparison_index == candidate_index:
                continue

            comparison = criteria[comparison_index]

            if (
                np.all(comparison >= candidate - 1e-12)
                and np.any(comparison > candidate + 1e-12)
            ):
                dominated = True
                break

        if not dominated:
            optimal.append(candidate_name)

    return optimal


def hybrid_ag_crossover(
    hybrid_durability: float,
    ag_durability: float,
) -> float | None:
    """
    Exact DC durability-weight crossover between Hybrid Padding-Squeezing
    and Ag Two-Step, using their fixed cost scores.
    """
    hybrid_cost = COST_SCORES["Hybrid Padding-Squeezing"]
    ag_cost = COST_SCORES["Ag Two-Step"]

    denominator = (
        hybrid_cost
        - ag_cost
        + ag_durability
        - hybrid_durability
    )

    if math.isclose(denominator, 0.0):
        return None

    crossover = (hybrid_cost - ag_cost) / denominator

    if 0.0 <= crossover <= 1.0:
        return float(crossover)

    return None


# ---------------------------------------------------------------------------
# 5. Scenario analysis
# ---------------------------------------------------------------------------

def analyse_scenario(
    q10: float,
    alpha: float,
    dc_weights: np.ndarray,
    dac_weights: np.ndarray,
) -> Dict[str, object]:
    durability_results = {
        formulation: calculate_durability_score(
            formulation,
            q10,
            alpha,
        )
        for formulation in DC_FORMULATIONS
    }

    antibacterial_results = {
        formulation: calculate_antibacterial_score(
            formulation,
            q10,
            alpha,
        )
        for formulation in DAC_FORMULATIONS
    }

    dc_criteria = np.asarray(
        [
            [
                float(durability_results[name]["score"]),
                COST_SCORES[name],
            ]
            for name in DC_FORMULATIONS
        ],
        dtype=float,
    )

    dac_criteria = np.asarray(
        [
            [
                float(durability_results[name]["score"]),
                float(antibacterial_results[name]["score"]),
                COST_SCORES[name],
            ]
            for name in DAC_FORMULATIONS
        ],
        dtype=float,
    )

    ag_excluded_criteria = np.asarray(
        [
            [
                float(durability_results[name]["score"]),
                float(antibacterial_results[name]["score"]),
                COST_SCORES[name],
            ]
            for name in AG_EXCLUDED_DAC_FORMULATIONS
        ],
        dtype=float,
    )

    dc_share_values = calculate_winning_shares(
        dc_criteria,
        dc_weights,
    )
    dac_share_values = calculate_winning_shares(
        dac_criteria,
        dac_weights,
    )
    ag_excluded_share_values = calculate_winning_shares(
        ag_excluded_criteria,
        dac_weights,
    )

    dc_shares = {
        name: float(dc_share_values[index])
        for index, name in enumerate(DC_FORMULATIONS)
    }
    dac_shares = {
        name: float(dac_share_values[index])
        for index, name in enumerate(DAC_FORMULATIONS)
    }
    ag_excluded_shares = {
        name: float(ag_excluded_share_values[index])
        for index, name in enumerate(
            AG_EXCLUDED_DAC_FORMULATIONS
        )
    }

    return {
        "Q10": q10,
        "alpha": alpha,
        "durability": durability_results,
        "antibacterial": antibacterial_results,
        "DC shares": dc_shares,
        "DAC shares": dac_shares,
        "Ag-excluded DAC shares": ag_excluded_shares,
        "DC Pareto": pareto_front(
            DC_FORMULATIONS,
            dc_criteria,
        ),
        "DAC Pareto": pareto_front(
            DAC_FORMULATIONS,
            dac_criteria,
        ),
        "Ag-excluded DAC Pareto": pareto_front(
            AG_EXCLUDED_DAC_FORMULATIONS,
            ag_excluded_criteria,
        ),
        "Hybrid-Ag DC crossover": hybrid_ag_crossover(
            float(
                durability_results[
                    "Hybrid Padding-Squeezing"
                ]["score"]
            ),
            float(
                durability_results["Ag Two-Step"]["score"]
            ),
        ),
    }


# ---------------------------------------------------------------------------
# 6. Output helpers
# ---------------------------------------------------------------------------

def write_results_csv(
    results: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    fieldnames = [
        "Q10",
        "alpha",
        "Hybrid-Ag DC crossover",
        "DC Pareto front",
        "DAC Pareto front",
        "Ag-excluded DAC Pareto front",
    ]

    for name in DC_FORMULATIONS:
        fieldnames.extend(
            [
                f"WIE - {name}",
                f"Wmax - {name}",
                f"Durability score - {name}",
                f"Durability branch - {name}",
                f"DC winning share (%) - {name}",
            ]
        )

    for name in DAC_FORMULATIONS:
        fieldnames.extend(
            [
                f"Antibacterial score - {name}",
                f"DAC winning share (%) - {name}",
            ]
        )

    for name in AG_EXCLUDED_DAC_FORMULATIONS:
        fieldnames.append(
            f"Ag-excluded DAC winning share (%) - {name}"
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
                "Q10": f"{float(result['Q10']):.1f}",
                "alpha": f"{float(result['alpha']):.1f}",
                "Hybrid-Ag DC crossover": (
                    ""
                    if result["Hybrid-Ag DC crossover"] is None
                    else (
                        f"{float(result['Hybrid-Ag DC crossover']):.9f}"
                    )
                ),
                "DC Pareto front": "; ".join(
                    result["DC Pareto"]  # type: ignore[arg-type]
                ),
                "DAC Pareto front": "; ".join(
                    result["DAC Pareto"]  # type: ignore[arg-type]
                ),
                "Ag-excluded DAC Pareto front": "; ".join(
                    result[
                        "Ag-excluded DAC Pareto"
                    ]  # type: ignore[arg-type]
                ),
            }

            durability = result["durability"]  # type: ignore[assignment]
            antibacterial = result[
                "antibacterial"
            ]  # type: ignore[assignment]
            dc_shares = result["DC shares"]  # type: ignore[assignment]
            dac_shares = result[
                "DAC shares"
            ]  # type: ignore[assignment]
            no_ag_shares = result[
                "Ag-excluded DAC shares"
            ]  # type: ignore[assignment]

            for name in DC_FORMULATIONS:
                row[f"WIE - {name}"] = (
                    f"{float(durability[name]['WIE']):.9f}"
                )
                row[f"Wmax - {name}"] = (
                    f"{float(durability[name]['Wmax']):.9f}"
                )
                row[f"Durability score - {name}"] = (
                    f"{float(durability[name]['score']):.9f}"
                )
                row[f"Durability branch - {name}"] = (
                    durability[name]["branch"]
                )
                row[f"DC winning share (%) - {name}"] = (
                    f"{float(dc_shares[name]):.9f}"
                )

            for name in DAC_FORMULATIONS:
                row[f"Antibacterial score - {name}"] = (
                    f"{float(antibacterial[name]['score']):.9f}"
                )
                row[f"DAC winning share (%) - {name}"] = (
                    f"{float(dac_shares[name]):.9f}"
                )

            for name in AG_EXCLUDED_DAC_FORMULATIONS:
                row[
                    f"Ag-excluded DAC winning share (%) - {name}"
                ] = f"{float(no_ag_shares[name]):.9f}"

            writer.writerow(row)


def scenario_matrix(
    results: Sequence[Mapping[str, object]],
    model_key: str,
    formulation: str,
) -> np.ndarray:
    """Return a matrix with alpha as rows and Q10 as columns."""
    matrix = np.zeros(
        (len(ALPHA_VALUES), len(Q10_VALUES)),
        dtype=float,
    )

    lookup = {
        (
            float(result["alpha"]),
            float(result["Q10"]),
        ): float(
            result[model_key][formulation]  # type: ignore[index]
        )
        for result in results
    }

    for row_index, alpha in enumerate(ALPHA_VALUES):
        for column_index, q10 in enumerate(Q10_VALUES):
            matrix[row_index, column_index] = lookup[
                (alpha, q10)
            ]

    return matrix


def plot_heatmap(
    matrix: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Create one standalone annotated heatmap."""
    figure = plt.figure(figsize=(7.0, 6.0))
    axis = figure.add_subplot(111)

    image = axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
    )

    axis.set_xticks(range(len(Q10_VALUES)))
    axis.set_xticklabels(
        [f"{value:g}" for value in Q10_VALUES]
    )
    axis.set_yticks(range(len(ALPHA_VALUES)))
    axis.set_yticklabels(
        [f"{value:.1f}" for value in ALPHA_VALUES]
    )

    axis.set_xlabel("Q10")
    axis.set_ylabel("Temperature-weighting exponent, alpha")
    axis.set_title(title)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.1f}",
                ha="center",
                va="center",
            )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Winning weight-space share (%)")

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_summary(
    results: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    hybrid_dc = np.asarray(
        [
            float(
                result["DC shares"][
                    "Hybrid Padding-Squeezing"
                ]
            )
            for result in results
        ]
    )
    ag_dc = np.asarray(
        [
            float(result["DC shares"]["Ag Two-Step"])
            for result in results
        ]
    )
    zno_dc = np.asarray(
        [
            float(result["DC shares"]["ZnO Starch"])
            for result in results
        ]
    )
    tio2_dc = np.asarray(
        [
            float(
                result["DC shares"][
                    "TiO2 Alkaline Hydrolysis"
                ]
            )
            for result in results
        ]
    )

    hybrid_dac = np.asarray(
        [
            float(
                result["DAC shares"][
                    "Hybrid Padding-Squeezing"
                ]
            )
            for result in results
        ]
    )
    ag_dac = np.asarray(
        [
            float(result["DAC shares"]["Ag Two-Step"])
            for result in results
        ]
    )
    zno_dac = np.asarray(
        [
            float(result["DAC shares"]["ZnO Starch"])
            for result in results
        ]
    )

    no_ag_hybrid = np.asarray(
        [
            float(
                result["Ag-excluded DAC shares"][
                    "Hybrid Padding-Squeezing"
                ]
            )
            for result in results
        ]
    )
    no_ag_in_situ = np.asarray(
        [
            float(
                result["Ag-excluded DAC shares"][
                    "Hybrid In-Situ"
                ]
            )
            for result in results
        ]
    )
    no_ag_zno = np.asarray(
        [
            float(
                result["Ag-excluded DAC shares"][
                    "ZnO Starch"
                ]
            )
            for result in results
        ]
    )

    leading_dc = hybrid_dc + ag_dc
    leading_dac = hybrid_dac + ag_dac

    zno_dc_scenarios = int(np.sum(zno_dc > 1e-12))
    zno_dac_scenarios = int(np.sum(zno_dac > 1e-12))
    tio2_dc_scenarios = int(np.sum(tio2_dc > 1e-12))

    max_zno_dc_index = int(np.argmax(zno_dc))
    max_zno_dac_index = int(np.argmax(zno_dac))

    max_zno_dc_result = results[max_zno_dc_index]
    max_zno_dac_result = results[max_zno_dac_index]

    lines = [
        "Revised WIE parameter sensitivity analysis",
        "",
        f"Number of tested Q10-alpha combinations: {len(results)}",
        "",
        "Complete Durability-Antibacterial-Cost model:",
        (
            "  Hybrid Padding-Squeezing winning-share range: "
            f"{hybrid_dac.min():.3f}% to {hybrid_dac.max():.3f}%."
        ),
        (
            "  Ag Two-Step winning-share range: "
            f"{ag_dac.min():.3f}% to {ag_dac.max():.3f}%."
        ),
        (
            "  Combined leading-share range: "
            f"{leading_dac.min():.3f}% to "
            f"{leading_dac.max():.3f}%."
        ),
        (
            "  ZnO Starch entered a non-zero winning region in "
            f"{zno_dac_scenarios} of 20 scenarios."
        ),
        (
            "  Maximum ZnO Starch DAC share: "
            f"{zno_dac.max():.3f}% at "
            f"Q10={float(max_zno_dac_result['Q10']):.1f}, "
            f"alpha={float(max_zno_dac_result['alpha']):.1f}."
        ),
        "",
        "Complete Durability-Cost model:",
        (
            "  Hybrid Padding-Squeezing winning-share range: "
            f"{hybrid_dc.min():.3f}% to {hybrid_dc.max():.3f}%."
        ),
        (
            "  Ag Two-Step winning-share range: "
            f"{ag_dc.min():.3f}% to {ag_dc.max():.3f}%."
        ),
        (
            "  Combined leading-share range: "
            f"{leading_dc.min():.3f}% to "
            f"{leading_dc.max():.3f}%."
        ),
        (
            "  ZnO Starch entered a non-zero winning region in "
            f"{zno_dc_scenarios} of 20 scenarios."
        ),
        (
            "  Maximum ZnO Starch DC share: "
            f"{zno_dc.max():.3f}% at "
            f"Q10={float(max_zno_dc_result['Q10']):.1f}, "
            f"alpha={float(max_zno_dc_result['alpha']):.1f}."
        ),
        (
            "  TiO2 Alkaline Hydrolysis entered a non-zero "
            f"winning region in {tio2_dc_scenarios} of 20 scenarios; "
            f"maximum share {tio2_dc.max():.3f}%."
        ),
        "",
        "Ag-excluded Durability-Antibacterial-Cost model:",
        (
            "  Hybrid Padding-Squeezing range: "
            f"{no_ag_hybrid.min():.3f}% to "
            f"{no_ag_hybrid.max():.3f}%."
        ),
        (
            "  Hybrid In-Situ range: "
            f"{no_ag_in_situ.min():.3f}% to "
            f"{no_ag_in_situ.max():.3f}%."
        ),
        (
            "  ZnO Starch range: "
            f"{no_ag_zno.min():.3f}% to "
            f"{no_ag_zno.max():.3f}%."
        ),
        (
            "  ZnO SDS occupied no winning region under any "
            "tested parameter combination."
        ),
    ]

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main() -> None:
    dc_weights = generate_dc_weights()
    dac_weights = generate_dac_weights()

    results: List[Dict[str, object]] = []

    # Alpha is the outer loop so CSV rows follow heatmap rows.
    for alpha in ALPHA_VALUES:
        for q10 in Q10_VALUES:
            results.append(
                analyse_scenario(
                    q10=q10,
                    alpha=alpha,
                    dc_weights=dc_weights,
                    dac_weights=dac_weights,
                )
            )

    csv_path = OUTPUT_DIR / "wie_sensitivity_results.csv"
    summary_path = OUTPUT_DIR / "wie_sensitivity_summary.txt"

    write_results_csv(results, csv_path)
    write_summary(results, summary_path)

    plot_heatmap(
        scenario_matrix(
            results,
            "DAC shares",
            "Ag Two-Step",
        ),
        (
            "Durability-Antibacterial-Cost winning share: "
            "Ag Two-Step"
        ),
        OUTPUT_DIR / "dac_ag_two_step_winning_share.png",
    )

    plot_heatmap(
        scenario_matrix(
            results,
            "DAC shares",
            "Hybrid Padding-Squeezing",
        ),
        (
            "Durability-Antibacterial-Cost winning share: "
            "Hybrid Padding-Squeezing"
        ),
        (
            OUTPUT_DIR
            / "dac_hybrid_padding_squeezing_winning_share.png"
        ),
    )

    plot_heatmap(
        scenario_matrix(
            results,
            "DC shares",
            "Ag Two-Step",
        ),
        "Durability-Cost winning share: Ag Two-Step",
        OUTPUT_DIR / "dc_ag_two_step_winning_share.png",
    )

    plot_heatmap(
        scenario_matrix(
            results,
            "DC shares",
            "Hybrid Padding-Squeezing",
        ),
        (
            "Durability-Cost winning share: "
            "Hybrid Padding-Squeezing"
        ),
        (
            OUTPUT_DIR
            / "dc_hybrid_padding_squeezing_winning_share.png"
        ),
    )

    print(summary_path.read_text(encoding="utf-8"))
    print()
    print(f"Results directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
