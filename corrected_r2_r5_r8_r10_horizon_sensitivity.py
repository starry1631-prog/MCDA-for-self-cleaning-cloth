
"""
Corrected sensitivity analysis for changing the common laundering horizon
from R10 to R2, R5, and R8.

The script uses the raw data transcribed from:
"Original Data Extracted From Sources(7).xlsx"

Core correction
---------------
The earlier version interpolated durability directly from cycle 0 to later
reported points whenever the target horizon was covered by the raw data.
That contradicted the manuscript's front-loaded-loss assumption.

This corrected version:

1. Calculates each formulation's baseline R10 durability score from the
   original durability data after WIE conversion, using the same baseline
   rules as the manuscript.

2. Keeps that R10 value fixed as the evidence-derived anchor.

3. Converts R10 to R2, R5, and R8 using the assumed cumulative-loss profile:
      75% of cumulative 30-cycle loss by wash 2;
      80% of cumulative 30-cycle loss by wash 10;
      the additional 5 percentage points distributed uniformly from
      wash 2 to wash 10.

For target horizon H:

    F(H) = 0.75 + (0.80 - 0.75) * (H - 2) / (10 - 2)

    R_H = 1 - [F(H) / 0.80] * (1 - R_10)

Thus, R2 already contains 93.75% of the loss estimated at R10.

Antibacterial treatment
-----------------------
The 75%-early-loss assumption concerns nanoparticle retention and is not
imposed on antibacterial decline. Antibacterial scores are independently
recalculated at R2, R5, R8, and R10 from the original species-specific
data after WIE conversion, using the manuscript's existing linear
interpolation/extrapolation rule and clipping to 0-100%.

Two three-criterion outputs are provided:

A. Matched-horizon DAC:
   D_H and A_H are both evaluated at the same horizon H.

B. Durability-only DAC isolation check:
   D_H changes, while antibacterial scores remain fixed at A_10.
   This isolates the effect of changing the durability horizon alone.

Outputs
-------
CSV:
  corrected_horizon_scores.csv
  corrected_horizon_winning_shares.csv

Figures:
  01_durability_scores_by_horizon.png
  02_antibacterial_scores_by_horizon.png
  03_DC_winning_shares_by_horizon.png
  04_DAC_matched_horizon_winning_shares.png
  05_DAC_durability_only_winning_shares.png
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

OUTPUT_DIR = Path("corrected_horizon_sensitivity_outputs")
TARGET_HORIZONS = (2.0, 5.0, 8.0, 10.0)

WEIGHT_STEP = 0.01

Q10 = 2.0
TEMPERATURE_WEIGHT = 0.5
REFERENCE_TEMPERATURE_C = 40.0
REFERENCE_DURATION_MIN = 45.0


# ---------------------------------------------------------------------
# RAW DATA TRANSCRIBED FROM THE UPLOADED EXCEL FILE
# Each tuple is (unstandardized washing cycles, reported value).
# ---------------------------------------------------------------------

DURABILITY_RAW: Dict[str, List[Tuple[float, float]]] = {
    "TiO2 Two-Step Dipping": [
        (5, 23.16),
        (10, 12.48),
        (15, 10.83),
        (20, 4.45),
    ],
    "TiO2 Alkaline Hydrolysis": [
        (0, 3.3),
        (10, 2.3),
    ],
    "Hybrid Padding-Squeezing": [
        (0, 1.449),
        (15, 0.112),
        (30, 0.057),
    ],
    "Hybrid In-Situ": [
        (0, 44500),
        (30, 9500),
    ],
    "ZnO Starch": [
        (0, 35),
        (5, 34.9),
        (10, 33.4),
    ],
    "ZnO SDS": [
        (0, 6.74),
        (5, 5.91),
        (10, 4.32),
    ],
    "Ag Two-Step": [
        (0, 180),
        (10, 165),
        (30, 148),
    ],
    "Ag One-Step": [
        (0, 1059),
        (20, 871),
    ],
}


ANTIBACTERIAL_RAW: Dict[
    str,
    Dict[str, List[Tuple[float, float]]],
] = {
    "Hybrid Padding-Squeezing": {
        "E. coli": [
            (0, 99.870),
            (15, 99.774),
            (30, 97.019),
        ],
        "S. aureus": [
            (0, 99.931),
            (15, 99.855),
            (30, 99.445),
        ],
    },
    "Hybrid In-Situ": {
        "E. coli": [
            (0, 100.0),
            (30, 97.8),
        ],
        "S. aureus": [
            (0, 100.0),
            (10, 99.3),
        ],
    },
    "ZnO Starch": {
        "E. coli": [
            (0, 100.0),
            (5, 100.0),
            (10, 76.4),
        ],
        "S. aureus": [
            (0, 100.0),
            (5, 98.1),
            (10, 96.2),
        ],
    },
    "ZnO SDS": {
        "E. coli": [
            (0, 91.0),
            (10, 89.0),
        ],
        "S. aureus": [
            (0, 92.3),
            (10, 90.2),
        ],
    },
    "Ag Two-Step": {
        "E. coli": [
            (0, 99.99),
            (30, 98.92),
        ],
        "S. aureus": [
            (0, 99.99),
            (30, 99.08),
        ],
    },
    "Ag One-Step": {
        "E. coli": [
            (5, 95.87),
            (10, 93.59),
        ],
        "S. aureus": [
            (5, 94.59),
            (10, 92.23),
        ],
    },
}


# Temperature and duration used for WIE.
# None means that the source did not report these variables; WIE = 1
# is retained as in the manuscript baseline.
WASH_CONDITIONS: Dict[
    str,
    Tuple[float | None, float | None],
] = {
    "TiO2 Two-Step Dipping": (40.0, 45.0),
    "TiO2 Alkaline Hydrolysis": (60.0, 20.0),
    "Hybrid Padding-Squeezing": (60.0, 30.0),
    "Hybrid In-Situ": (40.0, 45.0),
    "ZnO Starch": (22.0, 5.0),
    "ZnO SDS": (22.0, 5.0),
    "Ag Two-Step": (None, None),
    "Ag One-Step": (None, None),
}


RAW_MATERIAL_COST_USD_PER_G_FABRIC: Dict[str, float] = {
    "TiO2 Two-Step Dipping": 2.6950,
    "TiO2 Alkaline Hydrolysis": 1.6562,
    "Hybrid Padding-Squeezing": 0.0040,
    "Hybrid In-Situ": 0.4566,
    "ZnO Starch": 20.0115,
    "ZnO SDS": 1.1374,
    "Ag Two-Step": 0.1284,
    "Ag One-Step": 4.2840,
}


# ---------------------------------------------------------------------
# WIE AND BASELINE R10 DURABILITY
# ---------------------------------------------------------------------

def calculate_wie(
    temperature_c: float | None,
    duration_min: float | None,
) -> float:
    """Calculate WIE using the manuscript baseline equation."""
    if temperature_c is None or duration_min is None:
        return 1.0

    thermal_term = Q10 ** (
        (temperature_c - REFERENCE_TEMPERATURE_C) / 10.0
    )
    duration_term = duration_min / REFERENCE_DURATION_MIN

    return (
        thermal_term ** TEMPERATURE_WEIGHT
        * duration_term ** (1.0 - TEMPERATURE_WEIGHT)
    )


WIE: Dict[str, float] = {
    name: calculate_wie(*WASH_CONDITIONS[name])
    for name in DURABILITY_RAW
}


def estimate_initial_quantity(
    standardized_points: Sequence[Tuple[float, float]],
) -> float:
    """
    Use the reported cycle-0 quantity where available.
    Otherwise estimate the cycle-0 intercept by linear least squares,
    matching the manuscript baseline treatment.
    """
    for cycle, quantity in standardized_points:
        if math.isclose(cycle, 0.0, abs_tol=1e-12):
            return float(quantity)

    x = np.array(
        [cycle for cycle, _ in standardized_points],
        dtype=float,
    )
    y = np.array(
        [quantity for _, quantity in standardized_points],
        dtype=float,
    )

    if len(x) < 2:
        raise ValueError(
            "At least two observations are required "
            "to estimate the initial quantity."
        )

    _, intercept = np.polyfit(x, y, 1)

    if intercept <= 0:
        raise ValueError(
            f"Estimated cycle-0 intercept is non-positive: {intercept}"
        )

    return float(intercept)


def interpolate_quantity(
    standardized_points: Sequence[Tuple[float, float]],
    target_cycle: float,
    initial_quantity: float,
) -> float:
    """Select or linearly interpolate a quantity within observed data."""
    points = sorted(
        (float(cycle), float(quantity))
        for cycle, quantity in standardized_points
    )

    if not any(
        math.isclose(cycle, 0.0, abs_tol=1e-12)
        for cycle, _ in points
    ):
        points = [(0.0, initial_quantity)] + points

    for cycle, quantity in points:
        if math.isclose(
            cycle,
            target_cycle,
            abs_tol=1e-12,
        ):
            return quantity

    for (cycle_a, quantity_a), (cycle_b, quantity_b) in zip(
        points[:-1],
        points[1:],
    ):
        if cycle_a < target_cycle < cycle_b:
            fraction = (
                (target_cycle - cycle_a)
                / (cycle_b - cycle_a)
            )
            return (
                quantity_a
                + fraction * (quantity_b - quantity_a)
            )

    raise ValueError(
        "The requested target is outside the interpolation range."
    )


def baseline_r10_durability(
    formulation: str,
) -> float:
    """
    Recalculate the manuscript baseline R10 durability score from
    the original data after WIE conversion.

    Cases:
      Wmax >= 10:
          direct value or interpolation at 10.

      Wmax < 2:
          D10 = 1 - (0.80 / 0.75)
                    [1 - Rmax (Wmax / 2)]

      2 <= Wmax < 10:
          use the manuscript's continuous evidence-coverage rule.
    """
    standardized_points = sorted(
        (
            reported_cycle * WIE[formulation],
            quantity,
        )
        for reported_cycle, quantity
        in DURABILITY_RAW[formulation]
    )

    initial_quantity = estimate_initial_quantity(
        standardized_points
    )

    max_cycle, max_quantity = standardized_points[-1]
    max_retention = max_quantity / initial_quantity

    if max_cycle >= 10.0:
        quantity_at_ten = interpolate_quantity(
            standardized_points,
            10.0,
            initial_quantity,
        )
        score = quantity_at_ten / initial_quantity

    elif max_cycle < 2.0:
        score = 1.0 - (
            0.80 / 0.75
        ) * (
            1.0
            - max_retention * (max_cycle / 2.0)
        )

    else:
        observed_loss_fraction = (
            0.75
            + (0.80 - 0.75)
            * (max_cycle - 2.0)
            / (10.0 - 2.0)
        )
        evidence_coverage = (
            observed_loss_fraction / 0.80
        )
        score = 1.0 - (
            1.0 - max_retention
        ) / evidence_coverage

    return float(np.clip(score, 0.0, 1.0))


BASELINE_R10_DURABILITY: Dict[str, float] = {
    name: baseline_r10_durability(name)
    for name in DURABILITY_RAW
}


# ---------------------------------------------------------------------
# CORRECTED HORIZON CONVERSION
# ---------------------------------------------------------------------

def cumulative_loss_fraction(
    horizon: float,
) -> float:
    """
    Assumed fraction of cumulative 30-cycle loss occurring by H.

    F(2) = 0.75
    F(10) = 0.80
    The additional 0.05 is distributed uniformly from wash 2 to 10.
    """
    if not 2.0 <= horizon <= 10.0:
        raise ValueError(
            "The present sensitivity model is defined "
            "for horizons from 2 to 10."
        )

    return (
        0.75
        + (0.80 - 0.75)
        * (horizon - 2.0)
        / (10.0 - 2.0)
    )


def durability_from_r10(
    r10: float,
    target_horizon: float,
) -> float:
    """
    Convert the baseline R10 retention score to R_H.

    R_H = 1 - [F(H) / F(10)] (1 - R10)
        = 1 - [F(H) / 0.80] (1 - R10)
    """
    score = (
        1.0
        - cumulative_loss_fraction(target_horizon)
        / 0.80
        * (1.0 - r10)
    )
    return float(np.clip(score, 0.0, 1.0))


DURABILITY_BY_HORIZON: Dict[
    float,
    Dict[str, float],
] = {
    horizon: {
        name: durability_from_r10(
            BASELINE_R10_DURABILITY[name],
            horizon,
        )
        for name in DURABILITY_RAW
    }
    for horizon in TARGET_HORIZONS
}


# ---------------------------------------------------------------------
# ANTIBACTERIAL SCORES AT EACH HORIZON
# ---------------------------------------------------------------------

def interpolate_or_extrapolate(
    points: Sequence[Tuple[float, float]],
    target_horizon: float,
    wie: float,
) -> float:
    """
    Apply WIE to the reported washing cycles, then use:
      - direct value when available,
      - linear interpolation within the observed range,
      - linear extrapolation from the nearest two observations
        outside the range.

    Values are clipped to 0-100%, matching the manuscript.
    """
    standardized = sorted(
        (
            reported_cycle * wie,
            value,
        )
        for reported_cycle, value in points
    )

    for cycle, value in standardized:
        if math.isclose(
            cycle,
            target_horizon,
            abs_tol=1e-12,
        ):
            return float(np.clip(value, 0.0, 100.0))

    if len(standardized) < 2:
        raise ValueError(
            "At least two antibacterial observations are required."
        )

    if target_horizon < standardized[0][0]:
        point_a, point_b = (
            standardized[0],
            standardized[1],
        )

    elif target_horizon > standardized[-1][0]:
        point_a, point_b = (
            standardized[-2],
            standardized[-1],
        )

    else:
        point_a = point_b = None

        for candidate_a, candidate_b in zip(
            standardized[:-1],
            standardized[1:],
        ):
            if (
                candidate_a[0]
                < target_horizon
                < candidate_b[0]
            ):
                point_a, point_b = (
                    candidate_a,
                    candidate_b,
                )
                break

        if point_a is None or point_b is None:
            raise RuntimeError(
                "No antibacterial interpolation interval was found."
            )

    cycle_a, value_a = point_a
    cycle_b, value_b = point_b

    estimated_value = (
        value_a
        + (target_horizon - cycle_a)
        / (cycle_b - cycle_a)
        * (value_b - value_a)
    )

    return float(np.clip(estimated_value, 0.0, 100.0))


def antibacterial_score_at_horizon(
    formulation: str,
    target_horizon: float,
) -> float:
    """Average E. coli and S. aureus reduction ratios."""
    species_scores: List[float] = []

    for species_points in (
        ANTIBACTERIAL_RAW[formulation].values()
    ):
        reduction_percent = interpolate_or_extrapolate(
            species_points,
            target_horizon,
            WIE[formulation],
        )
        species_scores.append(
            reduction_percent / 100.0
        )

    return float(np.mean(species_scores))


ANTIBACTERIAL_BY_HORIZON: Dict[
    float,
    Dict[str, float],
] = {
    horizon: {
        name: antibacterial_score_at_horizon(
            name,
            horizon,
        )
        for name in ANTIBACTERIAL_RAW
    }
    for horizon in TARGET_HORIZONS
}


# ---------------------------------------------------------------------
# COST SCORES
# ---------------------------------------------------------------------

def calculate_cost_scores() -> Dict[str, float]:
    """
    Reverse logarithmic min-max normalization.

    The manuscript's USD 0.01 lower anchor is retained.
    """
    anchored_costs = {
        formulation: max(cost, 0.01)
        for formulation, cost
        in RAW_MATERIAL_COST_USD_PER_G_FABRIC.items()
    }

    minimum = min(anchored_costs.values())
    maximum = max(anchored_costs.values())

    denominator = (
        math.log(maximum) - math.log(minimum)
    )

    return {
        formulation: (
            math.log(maximum) - math.log(cost)
        ) / denominator
        for formulation, cost
        in anchored_costs.items()
    }


COST_SCORE = calculate_cost_scores()


# ---------------------------------------------------------------------
# MCDA FUNCTIONS
# ---------------------------------------------------------------------

def top_winners(
    score_map: Mapping[str, float],
    tolerance: float = 1e-12,
) -> List[str]:
    maximum = max(score_map.values())

    return [
        name
        for name, value in score_map.items()
        if math.isclose(
            value,
            maximum,
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    ]


def winning_shares_dc(
    durability_scores: Mapping[str, float],
) -> Dict[str, float]:
    """
    Durability-Cost model over WD = 0.00, 0.01, ..., 1.00.
    Exact ties are divided equally.
    """
    formulations = list(DURABILITY_RAW)
    counts = {
        name: 0.0
        for name in formulations
    }

    number_of_steps = int(
        round(1.0 / WEIGHT_STEP)
    )
    tested_weights = [
        index * WEIGHT_STEP
        for index in range(number_of_steps + 1)
    ]

    for durability_weight in tested_weights:
        scores = {
            name: (
                durability_weight
                * durability_scores[name]
                + (1.0 - durability_weight)
                * COST_SCORE[name]
            )
            for name in formulations
        }

        winners = top_winners(scores)
        tie_share = 1.0 / len(winners)

        for winner in winners:
            counts[winner] += tie_share

    return {
        name: (
            100.0
            * count
            / len(tested_weights)
        )
        for name, count in counts.items()
    }


def winning_shares_dac(
    durability_scores: Mapping[str, float],
    antibacterial_scores: Mapping[str, float],
) -> Dict[str, float]:
    """
    Durability-Antibacterial-Cost model over the 0.01 simplex.
    Exact ties are divided equally.
    """
    formulations = list(ANTIBACTERIAL_RAW)
    counts = {
        name: 0.0
        for name in formulations
    }

    total_combinations = 0
    number_of_steps = int(
        round(1.0 / WEIGHT_STEP)
    )

    for durability_index in range(
        number_of_steps + 1
    ):
        durability_weight = (
            durability_index * WEIGHT_STEP
        )

        for antibacterial_index in range(
            number_of_steps
            - durability_index
            + 1
        ):
            antibacterial_weight = (
                antibacterial_index
                * WEIGHT_STEP
            )
            cost_weight = (
                1.0
                - durability_weight
                - antibacterial_weight
            )

            scores = {
                name: (
                    durability_weight
                    * durability_scores[name]
                    + antibacterial_weight
                    * antibacterial_scores[name]
                    + cost_weight
                    * COST_SCORE[name]
                )
                for name in formulations
            }

            winners = top_winners(scores)
            tie_share = 1.0 / len(winners)

            for winner in winners:
                counts[winner] += tie_share

            total_combinations += 1

    return {
        name: (
            100.0
            * count
            / total_combinations
        )
        for name, count in counts.items()
    }


# ---------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------

def save_line_chart(
    x_values: Sequence[float],
    series: Mapping[str, Sequence[float]],
    title: str,
    y_label: str,
    output_path: Path,
    y_limits: Tuple[float, float] | None = None,
) -> None:
    """Save one independent line chart."""
    plt.figure(figsize=(10, 6))

    for name, values in series.items():
        plt.plot(
            x_values,
            values,
            marker="o",
            label=name,
        )

    plt.xlabel(
        "Common reference-equivalent wash horizon"
    )
    plt.ylabel(y_label)
    plt.title(title)

    plt.xticks(
        x_values,
        [
            f"R{int(value)}"
            for value in x_values
        ],
    )

    if y_limits is not None:
        plt.ylim(*y_limits)

    plt.grid(True, alpha=0.3)
    plt.legend(
        fontsize=8,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def run_analysis() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    horizons = list(TARGET_HORIZONS)
    durability_formulations = list(DURABILITY_RAW)
    antibacterial_formulations = list(
        ANTIBACTERIAL_RAW
    )

    dc_shares_by_horizon = {
        horizon: winning_shares_dc(
            DURABILITY_BY_HORIZON[horizon]
        )
        for horizon in horizons
    }

    dac_matched_shares_by_horizon = {
        horizon: winning_shares_dac(
            DURABILITY_BY_HORIZON[horizon],
            ANTIBACTERIAL_BY_HORIZON[horizon],
        )
        for horizon in horizons
    }

    antibacterial_at_r10 = (
        ANTIBACTERIAL_BY_HORIZON[10.0]
    )

    dac_durability_only_shares_by_horizon = {
        horizon: winning_shares_dac(
            DURABILITY_BY_HORIZON[horizon],
            antibacterial_at_r10,
        )
        for horizon in horizons
    }

    # -------------------------------------------------------------
    # Scores CSV
    # -------------------------------------------------------------
    score_csv = (
        OUTPUT_DIR
        / "corrected_horizon_scores.csv"
    )

    score_headers = [
        "Formulation",
        "WIE",
        "Raw material cost (USD/g fabric)",
        "Cost score",
        "Baseline R10 durability",
    ]

    for horizon in horizons:
        score_headers.append(
            f"Durability R{int(horizon)}"
        )
        score_headers.append(
            f"Antibacterial R{int(horizon)}"
        )

    with score_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=score_headers,
        )
        writer.writeheader()

        for formulation in durability_formulations:
            row = {
                "Formulation": formulation,
                "WIE": WIE[formulation],
                "Raw material cost (USD/g fabric)":
                    RAW_MATERIAL_COST_USD_PER_G_FABRIC[
                        formulation
                    ],
                "Cost score": COST_SCORE[
                    formulation
                ],
                "Baseline R10 durability":
                    BASELINE_R10_DURABILITY[
                        formulation
                    ],
            }

            for horizon in horizons:
                row[
                    f"Durability R{int(horizon)}"
                ] = DURABILITY_BY_HORIZON[
                    horizon
                ][formulation]

                row[
                    f"Antibacterial R{int(horizon)}"
                ] = ANTIBACTERIAL_BY_HORIZON[
                    horizon
                ].get(
                    formulation,
                    "",
                )

            writer.writerow(row)

    # -------------------------------------------------------------
    # Winning shares CSV
    # -------------------------------------------------------------
    winning_csv = (
        OUTPUT_DIR
        / "corrected_horizon_winning_shares.csv"
    )

    winning_rows: List[Dict[str, object]] = []

    for horizon in horizons:
        for formulation in durability_formulations:
            winning_rows.append({
                "Model": "Durability-Cost",
                "Horizon": f"R{int(horizon)}",
                "Formulation": formulation,
                "Winning share (%)":
                    dc_shares_by_horizon[
                        horizon
                    ][formulation],
            })

        for formulation in antibacterial_formulations:
            winning_rows.append({
                "Model":
                    "Durability-Antibacterial-Cost "
                    "(matched horizon)",
                "Horizon": f"R{int(horizon)}",
                "Formulation": formulation,
                "Winning share (%)":
                    dac_matched_shares_by_horizon[
                        horizon
                    ][formulation],
            })

            winning_rows.append({
                "Model":
                    "Durability-Antibacterial-Cost "
                    "(antibacterial fixed at R10)",
                "Horizon": f"R{int(horizon)}",
                "Formulation": formulation,
                "Winning share (%)":
                    dac_durability_only_shares_by_horizon[
                        horizon
                    ][formulation],
            })

    with winning_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Model",
                "Horizon",
                "Formulation",
                "Winning share (%)",
            ],
        )
        writer.writeheader()
        writer.writerows(winning_rows)

    # -------------------------------------------------------------
    # Figure 1: durability scores
    # -------------------------------------------------------------
    durability_series = {
        formulation: [
            DURABILITY_BY_HORIZON[
                horizon
            ][formulation]
            for horizon in horizons
        ]
        for formulation in durability_formulations
    }

    save_line_chart(
        horizons,
        durability_series,
        (
            "Durability-score sensitivity "
            "to the evaluation horizon"
        ),
        "Durability score",
        (
            OUTPUT_DIR
            / "01_durability_scores_by_horizon.png"
        ),
        (0.0, 1.02),
    )

    # -------------------------------------------------------------
    # Figure 2: antibacterial scores
    # -------------------------------------------------------------
    antibacterial_series = {
        formulation: [
            ANTIBACTERIAL_BY_HORIZON[
                horizon
            ][formulation]
            for horizon in horizons
        ]
        for formulation in antibacterial_formulations
    }

    save_line_chart(
        horizons,
        antibacterial_series,
        (
            "Antibacterial-score sensitivity "
            "to the evaluation horizon"
        ),
        "Antibacterial score",
        (
            OUTPUT_DIR
            / "02_antibacterial_scores_by_horizon.png"
        ),
        (0.0, 1.02),
    )

    # -------------------------------------------------------------
    # Figure 3: DC winning shares
    # -------------------------------------------------------------
    dc_series = {
        formulation: [
            dc_shares_by_horizon[
                horizon
            ][formulation]
            for horizon in horizons
        ]
        for formulation in durability_formulations
        if max(
            dc_shares_by_horizon[
                horizon
            ][formulation]
            for horizon in horizons
        ) > 0.0
    }

    save_line_chart(
        horizons,
        dc_series,
        (
            "Durability-Cost winning shares "
            "across evaluation horizons"
        ),
        "Winning share of tested weight space (%)",
        (
            OUTPUT_DIR
            / "03_DC_winning_shares_by_horizon.png"
        ),
        (0.0, 100.0),
    )

    # -------------------------------------------------------------
    # Figure 4: matched-horizon DAC winning shares
    # -------------------------------------------------------------
    dac_matched_series = {
        formulation: [
            dac_matched_shares_by_horizon[
                horizon
            ][formulation]
            for horizon in horizons
        ]
        for formulation in antibacterial_formulations
        if max(
            dac_matched_shares_by_horizon[
                horizon
            ][formulation]
            for horizon in horizons
        ) > 0.0
    }

    save_line_chart(
        horizons,
        dac_matched_series,
        (
            "Durability-Antibacterial-Cost "
            "winning shares across matched horizons"
        ),
        "Winning share of tested weight space (%)",
        (
            OUTPUT_DIR
            / (
                "04_DAC_matched_horizon_"
                "winning_shares.png"
            )
        ),
        (0.0, 100.0),
    )

    # -------------------------------------------------------------
    # Figure 5: durability-only DAC isolation check
    # -------------------------------------------------------------
    dac_fixed_series = {
        formulation: [
            dac_durability_only_shares_by_horizon[
                horizon
            ][formulation]
            for horizon in horizons
        ]
        for formulation in antibacterial_formulations
        if max(
            dac_durability_only_shares_by_horizon[
                horizon
            ][formulation]
            for horizon in horizons
        ) > 0.0
    }

    save_line_chart(
        horizons,
        dac_fixed_series,
        (
            "DAC isolation check: durability horizon varied, "
            "antibacterial score fixed at R10"
        ),
        "Winning share of tested weight space (%)",
        (
            OUTPUT_DIR
            / (
                "05_DAC_durability_only_"
                "winning_shares.png"
            )
        ),
        (0.0, 100.0),
    )

    # -------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------
    print("\nBaseline R10 durability reproduced from raw data")
    print("-" * 88)

    for formulation in durability_formulations:
        print(
            f"{formulation:32s}"
            f"{BASELINE_R10_DURABILITY[formulation]:10.6f}"
        )

    print("\nCorrected durability scores")
    print("-" * 88)

    header = (
        f"{'Formulation':32s}"
        + "".join(
            f"{f'R{int(horizon)}':>12s}"
            for horizon in horizons
        )
    )
    print(header)

    for formulation in durability_formulations:
        values = "".join(
            f"{DURABILITY_BY_HORIZON[horizon][formulation]:12.3f}"
            for horizon in horizons
        )
        print(
            f"{formulation:32s}{values}"
        )

    print("\nAntibacterial scores")
    print("-" * 88)
    print(header)

    for formulation in antibacterial_formulations:
        values = "".join(
            f"{ANTIBACTERIAL_BY_HORIZON[horizon][formulation]:12.3f}"
            for horizon in horizons
        )
        print(
            f"{formulation:32s}{values}"
        )

    print("\nNon-zero winning shares")
    print("-" * 88)

    for horizon in horizons:
        dc_nonzero = {
            name: round(value, 3)
            for name, value
            in dc_shares_by_horizon[horizon].items()
            if value > 0.0
        }

        matched_nonzero = {
            name: round(value, 3)
            for name, value
            in dac_matched_shares_by_horizon[
                horizon
            ].items()
            if value > 0.0
        }

        fixed_nonzero = {
            name: round(value, 3)
            for name, value
            in dac_durability_only_shares_by_horizon[
                horizon
            ].items()
            if value > 0.0
        }

        print(
            f"R{int(horizon)} DC: "
            f"{dc_nonzero}"
        )
        print(
            f"R{int(horizon)} DAC matched: "
            f"{matched_nonzero}"
        )
        print(
            f"R{int(horizon)} DAC A fixed: "
            f"{fixed_nonzero}"
        )

    print(
        f"\nFiles saved to: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    run_analysis()
