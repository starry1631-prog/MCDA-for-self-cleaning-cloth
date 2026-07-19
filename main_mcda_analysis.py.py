
"""
Recalculation package for the revised nanoparticle-textile MCDA model.

Key revision
------------
The source-derived loss profile is interpreted as:
- 75 percentage points of the initial metal content lost by wash 2;
- a further 5 percentage points lost gradually over washes 3-30.

For 2 <= W <= 30:

    F(W) = 0.75 + 0.05 * (W - 2) / 28

Therefore:

    F(2)  = 105 / 140
    F(10) = 107 / 140

and:

    F(10) / F(2) = 107 / 105

For Wmax < 2, the revised durability equation is:

    R10 = 1 - (107 / 105) * [1 - Rmax * (Wmax / 2)]

For 2 <= Wmax < 10:

    c(Wmax) = F(Wmax) / F(10)

    R10 = 1 - (1 - Rmax) / c(Wmax)

The script recalculates:
1. Baseline WIE, durability, antibacterial, and cost scores.
2. Baseline DC and DAC winning shares.
3. Ag-WIE sensitivity.
4. WIE-parameter sensitivity.
5. ZnO optimistic-persistence scenarios.
6. R2/R5/R8/R10 horizon sensitivity.
7. All figures affected by the revised ZnO durability scores.

Figures are saved without a top title as 600-dpi PNG and vector SVG files.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


# =====================================================================
# Settings
# =====================================================================

OUTPUT_DIR = Path("revised_107_105_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

T_REF = 40.0
TIME_REF = 45.0
BASELINE_Q10 = 2.0
BASELINE_ALPHA = 0.5

WEIGHT_STEP = 0.01
TARGET_HORIZONS = (2.0, 5.0, 8.0, 10.0)

Q10_VALUES = (1.5, 2.0, 2.5, 3.0)
ALPHA_VALUES = (0.3, 0.4, 0.5, 0.6, 0.7)

FORMULATIONS = (
    "TiO2 Two-Step Dipping",
    "TiO2 Alkaline Hydrolysis",
    "Hybrid Padding-Squeezing",
    "Hybrid In-Situ",
    "ZnO Starch",
    "ZnO SDS",
    "Ag Two-Step",
    "Ag One-Step",
)

DAC_FORMULATIONS = (
    "Hybrid Padding-Squeezing",
    "Hybrid In-Situ",
    "ZnO Starch",
    "ZnO SDS",
    "Ag Two-Step",
    "Ag One-Step",
)

AG_EXCLUDED_FORMULATIONS = (
    "Hybrid Padding-Squeezing",
    "Hybrid In-Situ",
    "ZnO Starch",
    "ZnO SDS",
)


# =====================================================================
# Raw data transcribed from the source-data workbook
# =====================================================================

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

RAW_COST_USD_PER_G_FABRIC: Dict[str, float] = {
    "TiO2 Two-Step Dipping": 2.6950,
    "TiO2 Alkaline Hydrolysis": 1.6562,
    "Hybrid Padding-Squeezing": 0.0040,
    "Hybrid In-Situ": 0.4566,
    "ZnO Starch": 20.0115,
    "ZnO SDS": 1.1374,
    "Ag Two-Step": 0.1284,
    "Ag One-Step": 4.2840,
}


# =====================================================================
# Core formulas
# =====================================================================

def loss_profile(washes: float) -> float:
    """
    Source-derived cumulative loss fraction relative to initial content.
    Valid for 2 <= washes <= 30.
    """
    if not 2.0 <= washes <= 30.0:
        raise ValueError("loss_profile is defined for 2 <= washes <= 30.")

    return (
        0.75
        + 0.05
        * (washes - 2.0)
        / (30.0 - 2.0)
    )


F2 = loss_profile(2.0)
F10 = loss_profile(10.0)
LOSS_SCALER_2_TO_10 = F10 / F2


def calculate_wie(
    temperature_c: float | None,
    duration_min: float | None,
    q10: float = BASELINE_Q10,
    alpha: float = BASELINE_ALPHA,
) -> float:
    """Calculate the Washing Intensity Equivalent."""
    if temperature_c is None or duration_min is None:
        return 1.0

    thermal_term = q10 ** (
        (temperature_c - T_REF) / 10.0
    )
    duration_term = duration_min / TIME_REF

    return (
        thermal_term ** alpha
        * duration_term ** (1.0 - alpha)
    )


def estimate_initial_quantity(
    standardized_points: Sequence[Tuple[float, float]],
) -> float:
    """
    Use reported cycle 0 where available. Otherwise estimate cycle 0
    by the linear least-squares intercept used in the manuscript.
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
            "to estimate the cycle-0 quantity."
        )

    _, intercept = np.polyfit(x, y, 1)

    if intercept <= 0:
        raise ValueError(
            f"Estimated cycle-0 quantity is non-positive: {intercept}"
        )

    return float(intercept)


def interpolate_quantity(
    standardized_points: Sequence[Tuple[float, float]],
    target_cycle: float,
    initial_quantity: float,
) -> float:
    """Select or linearly interpolate quantity within the data range."""
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
        "Target cycle lies outside the interpolation range."
    )


def durability_score(
    formulation: str,
    wie_value: float,
    horizon: float = 10.0,
) -> float:
    """
    Calculate retention-based durability at the requested horizon.

    Baseline use is horizon=10.

    For Wmax < 2:
        R10 = 1 - (107/105) *
              [1 - Rmax * (Wmax/2)]

    For 2 <= Wmax < 10:
        c(Wmax) = F(Wmax) / F(10)
        R10 = 1 - (1-Rmax) / c(Wmax)
    """
    if not math.isclose(horizon, 10.0):
        raise ValueError(
            "This function calculates the evidence-derived R10. "
            "Use durability_from_r10 for shorter horizons."
        )

    standardized_points = sorted(
        (
            reported_cycle * wie_value,
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
        score = (
            1.0
            - LOSS_SCALER_2_TO_10
            * (
                1.0
                - max_retention
                * (max_cycle / 2.0)
            )
        )

    else:
        evidence_coverage = (
            loss_profile(max_cycle) / F10
        )
        score = (
            1.0
            - (1.0 - max_retention)
            / evidence_coverage
        )

    return float(np.clip(score, 0.0, 1.0))


def durability_from_r10(
    r10_score: float,
    target_horizon: float,
) -> float:
    """
    Convert formulation-specific R10 to R_H using the same temporal
    loss profile while preserving R10 exactly.
    """
    if not 2.0 <= target_horizon <= 10.0:
        raise ValueError(
            "Target horizon must lie from 2 to 10."
        )

    score = (
        1.0
        - loss_profile(target_horizon)
        / F10
        * (1.0 - r10_score)
    )

    return float(np.clip(score, 0.0, 1.0))


def interpolate_or_extrapolate_percentage(
    points: Sequence[Tuple[float, float]],
    target_horizon: float,
    wie_value: float,
) -> float:
    """
    Apply WIE, then use direct selection, interpolation, or
    nearest-two-point extrapolation for antibacterial percentage.
    """
    standardized = sorted(
        (
            reported_cycle * wie_value,
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

    estimated = (
        value_a
        + (target_horizon - cycle_a)
        / (cycle_b - cycle_a)
        * (value_b - value_a)
    )

    return float(np.clip(estimated, 0.0, 100.0))


def antibacterial_score(
    formulation: str,
    wie_value: float,
    target_horizon: float = 10.0,
) -> float:
    """Average E. coli and S. aureus bacterial-reduction fractions."""
    species_values = []

    for species_points in (
        ANTIBACTERIAL_RAW[formulation].values()
    ):
        percentage = interpolate_or_extrapolate_percentage(
            species_points,
            target_horizon,
            wie_value,
        )
        species_values.append(percentage / 100.0)

    return float(np.mean(species_values))


def cost_scores(
    costs: Mapping[str, float],
) -> Dict[str, float]:
    """Reverse logarithmic min-max normalization."""
    anchored = {
        formulation: max(cost, 0.01)
        for formulation, cost in costs.items()
    }

    minimum = min(anchored.values())
    maximum = max(anchored.values())

    denominator = (
        math.log(maximum)
        - math.log(minimum)
    )

    return {
        formulation: (
            math.log(maximum)
            - math.log(cost)
        ) / denominator
        for formulation, cost in anchored.items()
    }


# =====================================================================
# MCDA helpers
# =====================================================================

def highest_ranked(
    scores: Mapping[str, float],
    tolerance: float = 1e-12,
) -> List[str]:
    maximum = max(scores.values())

    return [
        formulation
        for formulation, score in scores.items()
        if math.isclose(
            score,
            maximum,
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    ]


def winning_shares_dc(
    durability: Mapping[str, float],
    cost: Mapping[str, float],
    subset: Sequence[str] = FORMULATIONS,
) -> Dict[str, float]:
    counts = {
        formulation: 0.0
        for formulation in subset
    }

    total = 0

    for durability_index in range(101):
        durability_weight = (
            durability_index * WEIGHT_STEP
        )
        cost_weight = 1.0 - durability_weight

        scores = {
            formulation: (
                durability_weight
                * durability[formulation]
                + cost_weight
                * cost[formulation]
            )
            for formulation in subset
        }

        winners = highest_ranked(scores)
        tie_share = 1.0 / len(winners)

        for winner in winners:
            counts[winner] += tie_share

        total += 1

    return {
        formulation: 100.0 * count / total
        for formulation, count in counts.items()
    }


def winning_shares_dac(
    durability: Mapping[str, float],
    antibacterial: Mapping[str, float],
    cost: Mapping[str, float],
    subset: Sequence[str] = DAC_FORMULATIONS,
) -> Dict[str, float]:
    counts = {
        formulation: 0.0
        for formulation in subset
    }

    total = 0

    for durability_index in range(101):
        durability_weight = (
            durability_index * WEIGHT_STEP
        )

        for antibacterial_index in range(
            101 - durability_index
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
                formulation: (
                    durability_weight
                    * durability[formulation]
                    + antibacterial_weight
                    * antibacterial[formulation]
                    + cost_weight
                    * cost[formulation]
                )
                for formulation in subset
            }

            winners = highest_ranked(scores)
            tie_share = 1.0 / len(winners)

            for winner in winners:
                counts[winner] += tie_share

            total += 1

    return {
        formulation: 100.0 * count / total
        for formulation, count in counts.items()
    }


def pareto_front(
    criteria: Mapping[str, Mapping[str, float]],
    subset: Sequence[str],
) -> List[str]:
    front = []

    for formulation in subset:
        dominated = False

        for alternative in subset:
            if alternative == formulation:
                continue

            formulation_values = [
                criterion[formulation]
                for criterion in criteria.values()
            ]
            alternative_values = [
                criterion[alternative]
                for criterion in criteria.values()
            ]

            weakly_better = all(
                alternative_value
                >= formulation_value - 1e-12
                for alternative_value, formulation_value
                in zip(
                    alternative_values,
                    formulation_values,
                )
            )

            strictly_better = any(
                alternative_value
                > formulation_value + 1e-12
                for alternative_value, formulation_value
                in zip(
                    alternative_values,
                    formulation_values,
                )
            )

            if weakly_better and strictly_better:
                dominated = True
                break

        if not dominated:
            front.append(formulation)

    return front


# =====================================================================
# Baseline scores
# =====================================================================

BASELINE_WIE = {
    formulation: calculate_wie(
        *WASH_CONDITIONS[formulation],
    )
    for formulation in FORMULATIONS
}

BASELINE_DURABILITY = {
    formulation: durability_score(
        formulation,
        BASELINE_WIE[formulation],
    )
    for formulation in FORMULATIONS
}

BASELINE_ANTIBACTERIAL = {
    formulation: antibacterial_score(
        formulation,
        BASELINE_WIE[formulation],
    )
    for formulation in DAC_FORMULATIONS
}

BASELINE_COST_SCORE = cost_scores(
    RAW_COST_USD_PER_G_FABRIC
)

BASELINE_DC_SHARE = winning_shares_dc(
    BASELINE_DURABILITY,
    BASELINE_COST_SCORE,
)

BASELINE_DAC_SHARE = winning_shares_dac(
    BASELINE_DURABILITY,
    BASELINE_ANTIBACTERIAL,
    BASELINE_COST_SCORE,
)

BASELINE_AG_EXCLUDED_SHARE = winning_shares_dac(
    BASELINE_DURABILITY,
    BASELINE_ANTIBACTERIAL,
    BASELINE_COST_SCORE,
    subset=AG_EXCLUDED_FORMULATIONS,
)


# =====================================================================
# Ag-WIE sensitivity
# =====================================================================

AG_WIE_RESULTS = []

for integer_wie in range(100, 17, -1):
    assumed_ag_wie = integer_wie / 100.0

    durability = dict(BASELINE_DURABILITY)
    antibacterial = dict(BASELINE_ANTIBACTERIAL)

    for formulation in (
        "Ag Two-Step",
        "Ag One-Step",
    ):
        durability[formulation] = durability_score(
            formulation,
            assumed_ag_wie,
        )
        antibacterial[formulation] = antibacterial_score(
            formulation,
            assumed_ag_wie,
        )

    dc_share = winning_shares_dc(
        durability,
        BASELINE_COST_SCORE,
    )

    dac_share = winning_shares_dac(
        durability,
        antibacterial,
        BASELINE_COST_SCORE,
    )

    dc_front = pareto_front(
        {
            "Durability": durability,
            "Cost": BASELINE_COST_SCORE,
        },
        FORMULATIONS,
    )

    dac_front = pareto_front(
        {
            "Durability": durability,
            "Antibacterial": antibacterial,
            "Cost": BASELINE_COST_SCORE,
        },
        DAC_FORMULATIONS,
    )

    AG_WIE_RESULTS.append({
        "Ag WIE": assumed_ag_wie,
        "Durability": durability,
        "Antibacterial": antibacterial,
        "DC share": dc_share,
        "DAC share": dac_share,
        "DC Pareto": dc_front,
        "DAC Pareto": dac_front,
    })


# =====================================================================
# WIE-parameter sensitivity
# =====================================================================

WIE_PARAMETER_RESULTS = []

for q10 in Q10_VALUES:
    for alpha in ALPHA_VALUES:
        wie_values = {}

        for formulation in FORMULATIONS:
            if formulation.startswith("Ag "):
                wie_values[formulation] = 1.0
            else:
                wie_values[formulation] = calculate_wie(
                    *WASH_CONDITIONS[formulation],
                    q10=q10,
                    alpha=alpha,
                )

        durability = {
            formulation: durability_score(
                formulation,
                wie_values[formulation],
            )
            for formulation in FORMULATIONS
        }

        antibacterial = {
            formulation: antibacterial_score(
                formulation,
                wie_values[formulation],
            )
            for formulation in DAC_FORMULATIONS
        }

        dc_share = winning_shares_dc(
            durability,
            BASELINE_COST_SCORE,
        )

        dac_share = winning_shares_dac(
            durability,
            antibacterial,
            BASELINE_COST_SCORE,
        )

        ag_excluded_share = winning_shares_dac(
            durability,
            antibacterial,
            BASELINE_COST_SCORE,
            subset=AG_EXCLUDED_FORMULATIONS,
        )

        WIE_PARAMETER_RESULTS.append({
            "Q10": q10,
            "alpha": alpha,
            "WIE": wie_values,
            "Durability": durability,
            "Antibacterial": antibacterial,
            "DC share": dc_share,
            "DAC share": dac_share,
            "Ag-excluded share": ag_excluded_share,
        })


# =====================================================================
# ZnO persistence scenarios
# =====================================================================

OPTIMISTIC_DURABILITY = {
    "ZnO Starch": 33.4 / 35.0,
    "ZnO SDS": 4.32 / 6.74,
}

OPTIMISTIC_ANTIBACTERIAL = {
    "ZnO Starch": (
        76.4 + 96.2
    ) / 2.0 / 100.0,
    "ZnO SDS": (
        89.0 + 90.2
    ) / 2.0 / 100.0,
}

ZNO_SCENARIOS = {}

for scenario in (
    "Baseline",
    "Antibacterial-only",
    "Durability-only",
    "Joint optimistic",
):
    durability = dict(BASELINE_DURABILITY)
    antibacterial = dict(BASELINE_ANTIBACTERIAL)

    if scenario in (
        "Durability-only",
        "Joint optimistic",
    ):
        durability.update(OPTIMISTIC_DURABILITY)

    if scenario in (
        "Antibacterial-only",
        "Joint optimistic",
    ):
        antibacterial.update(
            OPTIMISTIC_ANTIBACTERIAL
        )

    ZNO_SCENARIOS[scenario] = {
        "Durability": durability,
        "Antibacterial": antibacterial,
        "DC share": winning_shares_dc(
            durability,
            BASELINE_COST_SCORE,
        ),
        "DAC share": winning_shares_dac(
            durability,
            antibacterial,
            BASELINE_COST_SCORE,
        ),
        "Ag-excluded share": winning_shares_dac(
            durability,
            antibacterial,
            BASELINE_COST_SCORE,
            subset=AG_EXCLUDED_FORMULATIONS,
        ),
    }


# =====================================================================
# Horizon sensitivity
# =====================================================================

HORIZON_RESULTS = {}

for horizon in TARGET_HORIZONS:
    durability = {
        formulation: durability_from_r10(
            BASELINE_DURABILITY[formulation],
            horizon,
        )
        for formulation in FORMULATIONS
    }

    antibacterial = {
        formulation: antibacterial_score(
            formulation,
            BASELINE_WIE[formulation],
            target_horizon=horizon,
        )
        for formulation in DAC_FORMULATIONS
    }

    HORIZON_RESULTS[horizon] = {
        "Durability": durability,
        "Antibacterial": antibacterial,
        "DC share": winning_shares_dc(
            durability,
            BASELINE_COST_SCORE,
        ),
        "DAC share": winning_shares_dac(
            durability,
            antibacterial,
            BASELINE_COST_SCORE,
        ),
        "DAC A fixed": winning_shares_dac(
            durability,
            BASELINE_ANTIBACTERIAL,
            BASELINE_COST_SCORE,
        ),
    }


# =====================================================================
# CSV outputs
# =====================================================================

def write_baseline_csv() -> None:
    path = OUTPUT_DIR / "baseline_scores.csv"

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow([
            "Formulation",
            "WIE",
            "Durability score",
            "Antibacterial score",
            "Raw cost (USD/g fabric)",
            "Cost score",
            "DC winning share (%)",
            "DAC winning share (%)",
        ])

        for formulation in FORMULATIONS:
            writer.writerow([
                formulation,
                BASELINE_WIE[formulation],
                BASELINE_DURABILITY[formulation],
                BASELINE_ANTIBACTERIAL.get(
                    formulation,
                    "",
                ),
                RAW_COST_USD_PER_G_FABRIC[
                    formulation
                ],
                BASELINE_COST_SCORE[formulation],
                BASELINE_DC_SHARE.get(
                    formulation,
                    "",
                ),
                BASELINE_DAC_SHARE.get(
                    formulation,
                    "",
                ),
            ])


def write_material_average_csv() -> None:
    classes = {
        "TiO2": (
            "TiO2 Two-Step Dipping",
            "TiO2 Alkaline Hydrolysis",
        ),
        "Hybrid": (
            "Hybrid Padding-Squeezing",
            "Hybrid In-Situ",
        ),
        "ZnO": (
            "ZnO Starch",
            "ZnO SDS",
        ),
        "Ag": (
            "Ag Two-Step",
            "Ag One-Step",
        ),
    }

    path = (
        OUTPUT_DIR
        / "material_average_scores.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow([
            "Material class",
            "Average durability",
            "Average cost score",
        ])

        for material, members in classes.items():
            writer.writerow([
                material,
                np.mean([
                    BASELINE_DURABILITY[member]
                    for member in members
                ]),
                np.mean([
                    BASELINE_COST_SCORE[member]
                    for member in members
                ]),
            ])


def write_ag_wie_csv() -> None:
    path = (
        OUTPUT_DIR
        / "ag_wie_sensitivity.csv"
    )

    headers = [
        "Ag WIE",
        "Ag Two-Step durability",
        "Ag Two-Step antibacterial",
        "Ag One-Step durability",
        "Ag One-Step antibacterial",
    ]

    for model in ("DC", "DAC"):
        for formulation in FORMULATIONS:
            if (
                model == "DAC"
                and formulation
                not in DAC_FORMULATIONS
            ):
                continue

            headers.append(
                f"{model} share: {formulation}"
            )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=headers,
        )
        writer.writeheader()

        for result in AG_WIE_RESULTS:
            row = {
                "Ag WIE": result["Ag WIE"],
                "Ag Two-Step durability":
                    result["Durability"][
                        "Ag Two-Step"
                    ],
                "Ag Two-Step antibacterial":
                    result["Antibacterial"][
                        "Ag Two-Step"
                    ],
                "Ag One-Step durability":
                    result["Durability"][
                        "Ag One-Step"
                    ],
                "Ag One-Step antibacterial":
                    result["Antibacterial"][
                        "Ag One-Step"
                    ],
            }

            for formulation in FORMULATIONS:
                row[
                    f"DC share: {formulation}"
                ] = result["DC share"][
                    formulation
                ]

            for formulation in DAC_FORMULATIONS:
                row[
                    f"DAC share: {formulation}"
                ] = result["DAC share"][
                    formulation
                ]

            writer.writerow(row)


def write_wie_parameter_csv() -> None:
    path = (
        OUTPUT_DIR
        / "wie_parameter_sensitivity.csv"
    )

    headers = [
        "Q10",
        "alpha",
    ]

    for model in (
        "DC",
        "DAC",
        "Ag-excluded DAC",
    ):
        subset = (
            FORMULATIONS
            if model == "DC"
            else (
                DAC_FORMULATIONS
                if model == "DAC"
                else AG_EXCLUDED_FORMULATIONS
            )
        )

        for formulation in subset:
            headers.append(
                f"{model} share: {formulation}"
            )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=headers,
        )
        writer.writeheader()

        for result in WIE_PARAMETER_RESULTS:
            row = {
                "Q10": result["Q10"],
                "alpha": result["alpha"],
            }

            for formulation in FORMULATIONS:
                row[
                    f"DC share: {formulation}"
                ] = result["DC share"][
                    formulation
                ]

            for formulation in DAC_FORMULATIONS:
                row[
                    f"DAC share: {formulation}"
                ] = result["DAC share"][
                    formulation
                ]

            for formulation in (
                AG_EXCLUDED_FORMULATIONS
            ):
                row[
                    (
                        "Ag-excluded DAC share: "
                        f"{formulation}"
                    )
                ] = result[
                    "Ag-excluded share"
                ][formulation]

            writer.writerow(row)


def write_zno_scenarios_csv() -> None:
    path = (
        OUTPUT_DIR
        / "zno_persistence_scenarios.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow([
            "Scenario",
            "Model",
            "Formulation",
            "Winning share (%)",
        ])

        for scenario, result in (
            ZNO_SCENARIOS.items()
        ):
            for model_key, model_name in (
                ("DC share", "DC"),
                ("DAC share", "DAC"),
                (
                    "Ag-excluded share",
                    "Ag-excluded DAC",
                ),
            ):
                for formulation, share in (
                    result[model_key].items()
                ):
                    writer.writerow([
                        scenario,
                        model_name,
                        formulation,
                        share,
                    ])


def write_horizon_csv() -> None:
    path = (
        OUTPUT_DIR
        / "horizon_sensitivity.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow([
            "Horizon",
            "Measure",
            "Formulation",
            "Value",
        ])

        for horizon, result in (
            HORIZON_RESULTS.items()
        ):
            for measure in (
                "Durability",
                "Antibacterial",
                "DC share",
                "DAC share",
                "DAC A fixed",
            ):
                for formulation, value in (
                    result[measure].items()
                ):
                    writer.writerow([
                        f"R{int(horizon)}",
                        measure,
                        formulation,
                        value,
                    ])


# =====================================================================
# Figure helpers
# =====================================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def save_figure(
    figure: plt.Figure,
    filename_stem: str,
) -> None:
    png_path = (
        OUTPUT_DIR
        / f"{filename_stem}.png"
    )
    svg_path = (
        OUTPUT_DIR
        / f"{filename_stem}.svg"
    )

    figure.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )
    figure.savefig(
        svg_path,
        bbox_inches="tight",
    )
    plt.close(figure)


def decision_region_data(
    durability: Mapping[str, float],
    antibacterial: Mapping[str, float],
    cost: Mapping[str, float],
    subset: Sequence[str],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
]:
    x_values = []
    y_values = []
    winners = []

    winner_names = []

    for durability_index in range(101):
        durability_weight = (
            durability_index * WEIGHT_STEP
        )

        for antibacterial_index in range(
            101 - durability_index
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
                formulation: (
                    durability_weight
                    * durability[formulation]
                    + antibacterial_weight
                    * antibacterial[formulation]
                    + cost_weight
                    * cost[formulation]
                )
                for formulation in subset
            }

            winner = highest_ranked(scores)[0]

            if winner not in winner_names:
                winner_names.append(winner)

            x_values.append(durability_weight)
            y_values.append(antibacterial_weight)
            winners.append(
                winner_names.index(winner)
            )

    return (
        np.array(x_values),
        np.array(y_values),
        np.array(winners),
        winner_names,
    )


def make_decision_region_figure(
    durability: Mapping[str, float],
    antibacterial: Mapping[str, float],
    cost: Mapping[str, float],
    subset: Sequence[str],
    filename_stem: str,
) -> None:
    x_values, y_values, winner_codes, winner_names = (
        decision_region_data(
            durability,
            antibacterial,
            cost,
            subset,
        )
    )

    triangulation = mtri.Triangulation(
        x_values,
        y_values,
    )

    figure, axis = plt.subplots(
        figsize=(7.5, 6.7)
    )

    levels = (
        np.arange(len(winner_names) + 1)
        - 0.5
    )

    contour = axis.tricontourf(
        triangulation,
        winner_codes,
        levels=levels,
    )

    colorbar = figure.colorbar(
        contour,
        ax=axis,
        ticks=np.arange(len(winner_names)),
        pad=0.03,
    )
    colorbar.ax.set_yticklabels(
        winner_names
    )
    colorbar.set_label(
        "Highest-ranked formulation"
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal", adjustable="box")

    axis.set_xlabel(
        "Durability weight, $w_D$"
    )
    axis.set_ylabel(
        "Antibacterial weight, $w_A$"
    )

    axis.text(
        0.02,
        0.02,
        "Cost only",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
    )
    axis.text(
        0.98,
        0.02,
        "Durability only",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
    )
    axis.text(
        0.02,
        0.98,
        "Antibacterial only",
        transform=axis.transAxes,
        ha="left",
        va="top",
    )

    axis.grid(alpha=0.25)
    figure.tight_layout()

    save_figure(
        figure,
        filename_stem,
    )


def make_dc_weight_curves() -> None:
    durability_weights = np.linspace(
        0.0,
        1.0,
        101,
    )

    figure, axis = plt.subplots(
        figsize=(8.5, 5.8)
    )

    for formulation in FORMULATIONS:
        scores = (
            durability_weights
            * BASELINE_DURABILITY[
                formulation
            ]
            + (1.0 - durability_weights)
            * BASELINE_COST_SCORE[
                formulation
            ]
        )

        axis.plot(
            durability_weights,
            scores,
            linewidth=1.8,
            label=formulation,
        )

    axis.set_xlabel(
        "Durability weight, $w_D$"
    )
    axis.set_ylabel(
        "Durability-Cost index"
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)

    axis.legend(
        loc="lower right",
        frameon=True,
    )

    figure.tight_layout()

    save_figure(
        figure,
        "Fig1b_DC_weight_curves",
    )


def make_material_average_figure() -> None:
    material_members = {
        "TiO2 average": (
            "TiO2 Two-Step Dipping",
            "TiO2 Alkaline Hydrolysis",
        ),
        "Hybrid average": (
            "Hybrid Padding-Squeezing",
            "Hybrid In-Situ",
        ),
        "ZnO average": (
            "ZnO Starch",
            "ZnO SDS",
        ),
        "Ag average": (
            "Ag Two-Step",
            "Ag One-Step",
        ),
    }

    durability_weights = np.linspace(
        0.0,
        1.0,
        101,
    )

    figure, axis = plt.subplots(
        figsize=(8.2, 5.7)
    )

    for label, members in (
        material_members.items()
    ):
        average_durability = np.mean([
            BASELINE_DURABILITY[member]
            for member in members
        ])

        average_cost = np.mean([
            BASELINE_COST_SCORE[member]
            for member in members
        ])

        scores = (
            durability_weights
            * average_durability
            + (1.0 - durability_weights)
            * average_cost
        )

        axis.plot(
            durability_weights,
            scores,
            linewidth=2.0,
            label=label,
        )

    axis.set_xlabel(
        "Durability weight, $w_D$"
    )
    axis.set_ylabel(
        "Material-average Durability-Cost index"
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")

    figure.tight_layout()

    save_figure(
        figure,
        "Fig5_material_average_scores",
    )


def make_pareto_figure() -> None:
    pareto = set(
        pareto_front(
            {
                "Durability":
                    BASELINE_DURABILITY,
                "Antibacterial":
                    BASELINE_ANTIBACTERIAL,
                "Cost":
                    BASELINE_COST_SCORE,
            },
            DAC_FORMULATIONS,
        )
    )

    label_offsets = {
        "Hybrid Padding-Squeezing":
            (0.006, -0.022),
        "Hybrid In-Situ":
            (0.008, 0.012),
        "ZnO Starch":
            (0.008, 0.004),
        "ZnO SDS":
            (0.008, -0.012),
        "Ag Two-Step":
            (-0.090, 0.006),
        "Ag One-Step":
            (0.008, -0.012),
    }

    figure, axis = plt.subplots(
        figsize=(8.8, 6.3)
    )

    for formulation in DAC_FORMULATIONS:
        marker = (
            "*"
            if formulation in pareto
            else "o"
        )

        bubble_size = (
            110.0
            + 900.0
            * BASELINE_COST_SCORE[
                formulation
            ]
        )

        axis.scatter(
            BASELINE_DURABILITY[
                formulation
            ],
            BASELINE_ANTIBACTERIAL[
                formulation
            ],
            s=bubble_size,
            marker=marker,
            alpha=0.72,
            linewidth=1.0,
        )

        offset_x, offset_y = (
            label_offsets[formulation]
        )

        label = formulation

        if formulation in pareto:
            label += " (Pareto)"

        axis.annotate(
            label,
            (
                BASELINE_DURABILITY[
                    formulation
                ],
                BASELINE_ANTIBACTERIAL[
                    formulation
                ],
            ),
            xytext=(
                BASELINE_DURABILITY[
                    formulation
                ] + offset_x,
                BASELINE_ANTIBACTERIAL[
                    formulation
                ] + offset_y,
            ),
            fontsize=9,
            ha=(
                "right"
                if formulation == "Ag Two-Step"
                else "left"
            ),
            fontweight=(
                "bold"
                if formulation in pareto
                else "normal"
            ),
        )

    for score in (
        0.0,
        0.5,
        1.0,
    ):
        axis.scatter(
            [],
            [],
            s=110.0 + 900.0 * score,
            alpha=0.72,
            label=f"Cost score = {score:.1f}",
        )

    axis.set_xlabel(
        "Durability score"
    )
    axis.set_ylabel(
        "Antibacterial score"
    )

    axis.set_xlim(0.50, 0.95)
    axis.set_ylim(0.35, 1.025)
    axis.grid(alpha=0.25)

    axis.legend(
        loc="lower left",
        frameon=True,
        labelspacing=1.1,
    )

    figure.tight_layout()

    save_figure(
        figure,
        "Fig7_criteria_space_Pareto",
    )


def make_heatmap(
    values: np.ndarray,
    colorbar_label: str,
    filename_stem: str,
) -> None:
    figure, axis = plt.subplots(
        figsize=(7.2, 5.6)
    )

    image = axis.imshow(
        values,
        origin="lower",
        aspect="auto",
    )

    axis.set_xticks(
        np.arange(len(ALPHA_VALUES))
    )
    axis.set_xticklabels([
        f"{value:.1f}"
        for value in ALPHA_VALUES
    ])

    axis.set_yticks(
        np.arange(len(Q10_VALUES))
    )
    axis.set_yticklabels([
        f"{value:.1f}"
        for value in Q10_VALUES
    ])

    axis.set_xlabel(
        "Temperature weighting exponent, $\\alpha$"
    )
    axis.set_ylabel(
        "Thermal-sensitivity coefficient, $Q_{10}$"
    )

    for row in range(values.shape[0]):
        for column in range(
            values.shape[1]
        ):
            axis.text(
                column,
                row,
                f"{values[row, column]:.1f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        pad=0.03,
    )
    colorbar.set_label(
        colorbar_label
    )

    figure.tight_layout()

    save_figure(
        figure,
        filename_stem,
    )


def make_wie_parameter_heatmaps() -> None:
    lookup = {
        (
            result["Q10"],
            result["alpha"],
        ): result
        for result in WIE_PARAMETER_RESULTS
    }

    specifications = (
        (
            "DAC share",
            "Ag Two-Step",
            "Supp_Fig2a_Ag_TwoStep_DAC",
        ),
        (
            "DAC share",
            "Hybrid Padding-Squeezing",
            "Supp_Fig2b_Hybrid_Pad_DAC",
        ),
        (
            "DC share",
            "Ag Two-Step",
            "Supp_Fig2c_Ag_TwoStep_DC",
        ),
        (
            "DC share",
            "Hybrid Padding-Squeezing",
            "Supp_Fig2d_Hybrid_Pad_DC",
        ),
    )

    for model_key, formulation, stem in (
        specifications
    ):
        values = np.zeros(
            (
                len(Q10_VALUES),
                len(ALPHA_VALUES),
            )
        )

        for row, q10 in enumerate(
            Q10_VALUES
        ):
            for column, alpha in enumerate(
                ALPHA_VALUES
            ):
                values[row, column] = (
                    lookup[(q10, alpha)]
                    [model_key]
                    [formulation]
                )

        make_heatmap(
            values,
            (
                "Winning share of tested "
                "weight space (%)"
            ),
            stem,
        )


def make_ag_wie_figure(
    model_key: str,
    subset: Sequence[str],
    filename_stem: str,
) -> None:
    x_values = np.array([
        result["Ag WIE"]
        for result in AG_WIE_RESULTS
    ])

    figure, axis = plt.subplots(
        figsize=(8.5, 5.8)
    )

    for formulation in subset:
        values = np.array([
            result[model_key][formulation]
            for result in AG_WIE_RESULTS
        ])

        if np.max(values) <= 0.0:
            continue

        axis.plot(
            x_values,
            values,
            linewidth=1.9,
            label=formulation,
        )

    axis.set_xlabel(
        "Assumed Ag Washing Intensity Equivalent"
    )
    axis.set_ylabel(
        "Winning share of tested weight space (%)"
    )

    axis.set_xlim(1.0, 0.18)
    axis.set_ylim(0.0, 100.0)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")

    figure.tight_layout()

    save_figure(
        figure,
        filename_stem,
    )


def make_horizon_line_figure(
    measure_key: str,
    subset: Sequence[str],
    y_label: str,
    filename_stem: str,
    y_limits: Tuple[float, float],
) -> None:
    horizons = list(TARGET_HORIZONS)

    figure, axis = plt.subplots(
        figsize=(8.5, 5.8)
    )

    for formulation in subset:
        values = [
            HORIZON_RESULTS[horizon]
            [measure_key]
            [formulation]
            for horizon in horizons
        ]

        if (
            "share" in measure_key.lower()
            and max(values) <= 0.0
        ):
            continue

        axis.plot(
            horizons,
            values,
            marker="o",
            linewidth=1.8,
            label=formulation,
        )

    axis.set_xlabel(
        "Common reference-equivalent wash horizon"
    )
    axis.set_ylabel(y_label)
    axis.set_xticks(
        horizons,
        [
            f"R{int(horizon)}"
            for horizon in horizons
        ],
    )
    axis.set_ylim(*y_limits)
    axis.grid(alpha=0.25)
    axis.legend(
        loc="best",
        bbox_to_anchor=(1.02, 1.0),
    )

    figure.tight_layout()

    save_figure(
        figure,
        filename_stem,
    )


# =====================================================================
# Replacement summary
# =====================================================================

def first_ag_wie_event(
    predicate,
) -> float | None:
    for result in AG_WIE_RESULTS:
        if predicate(result):
            return result["Ag WIE"]

    return None


def round_one(value: float) -> str:
    return f"{value:.1f}"


def write_replacement_summary() -> None:
    low_ag = AG_WIE_RESULTS[-1]

    hybrid_pareto_entry = first_ag_wie_event(
        lambda result:
            "Hybrid In-Situ"
            in result["DAC Pareto"]
    )

    hybrid_nonzero_dac = first_ag_wie_event(
        lambda result:
            result["DAC share"][
                "Hybrid In-Situ"
            ] > 0.0
    )

    zno_nonzero_dc = first_ag_wie_event(
        lambda result:
            result["DC share"][
                "ZnO Starch"
            ] > 0.0
    )

    zno_nonzero_dac = first_ag_wie_event(
        lambda result:
            result["DAC share"][
                "ZnO Starch"
            ] > 0.0
    )

    zno_dac_nonzero_scenarios = [
        result
        for result in WIE_PARAMETER_RESULTS
        if result["DAC share"][
            "ZnO Starch"
        ] > 0.0
    ]

    max_zno_dac = max(
        result["DAC share"][
            "ZnO Starch"
        ]
        for result in WIE_PARAMETER_RESULTS
    )

    max_zno_dc = max(
        result["DC share"][
            "ZnO Starch"
        ]
        for result in WIE_PARAMETER_RESULTS
    )

    min_pair_dac = min(
        result["DAC share"][
            "Hybrid Padding-Squeezing"
        ]
        + result["DAC share"][
            "Ag Two-Step"
        ]
        for result in WIE_PARAMETER_RESULTS
    )

    min_pair_dc = min(
        result["DC share"][
            "Hybrid Padding-Squeezing"
        ]
        + result["DC share"][
            "Ag Two-Step"
        ]
        for result in WIE_PARAMETER_RESULTS
    )

    max_noag_hybrid = max(
        result["Ag-excluded share"][
            "Hybrid In-Situ"
        ]
        for result in WIE_PARAMETER_RESULTS
    )

    min_noag_pad = min(
        result["Ag-excluded share"][
            "Hybrid Padding-Squeezing"
        ]
        for result in WIE_PARAMETER_RESULTS
    )

    max_noag_zno = max(
        result["Ag-excluded share"][
            "ZnO Starch"
        ]
        for result in WIE_PARAMETER_RESULTS
    )

    zno_baseline = (
        ZNO_SCENARIOS["Baseline"]
        ["Ag-excluded share"]
    )
    zno_antibacterial = (
        ZNO_SCENARIOS["Antibacterial-only"]
        ["Ag-excluded share"]
    )
    zno_durability = (
        ZNO_SCENARIOS["Durability-only"]
        ["Ag-excluded share"]
    )
    zno_joint = (
        ZNO_SCENARIOS["Joint optimistic"]
        ["Ag-excluded share"]
    )

    material_zno_average = np.mean([
        BASELINE_DURABILITY["ZnO Starch"],
        BASELINE_DURABILITY["ZnO SDS"],
    ])

    lines = [
        "REQUIRED NUMERICAL REPLACEMENTS",
        "================================",
        "",
        "Core durability equation",
        "------------------------",
        "For Wmax < 2, use:",
        "",
        "R10 = 1 - (107/105) [1 - Rmax (Wmax/2)]",
        "",
        "For 2 <= Wmax < 10, use:",
        "",
        "c(Wmax) = F(Wmax)/F(10)",
        "",
        "R10 = 1 - (1-Rmax)/c(Wmax)",
        "",
        "where F(W) = 0.75 + 0.05(W-2)/28.",
        "",
        f"F(2) = {F2:.6f}",
        f"F(10) = {F10:.6f}",
        (
            "F(10)/F(2) = "
            f"{LOSS_SCALER_2_TO_10:.9f} "
            "= 107/105"
        ),
        "",
        "Baseline ZnO scores",
        "-------------------",
        (
            "ZnO-Starch durability: "
            f"{BASELINE_DURABILITY['ZnO Starch']:.3f}"
        ),
        (
            "ZnO-SDS durability: "
            f"{BASELINE_DURABILITY['ZnO SDS']:.3f}"
        ),
        (
            "ZnO material-average durability: "
            f"{material_zno_average:.3f}"
        ),
        "",
        "Ag-WIE sensitivity",
        "------------------",
        (
            "Hybrid In-Situ first enters the "
            "three-criterion Pareto front: "
            f"Ag WIE = {hybrid_pareto_entry:.2f}"
        ),
        (
            "Hybrid In-Situ first obtains a "
            "non-zero DAC winning region: "
            f"Ag WIE = {hybrid_nonzero_dac:.2f}"
        ),
        (
            "ZnO-Starch first obtains a "
            "non-zero DC winning region: "
            f"Ag WIE = {zno_nonzero_dc:.2f}"
        ),
        (
            "ZnO-Starch first obtains a "
            "non-zero DAC winning region: "
            f"Ag WIE = {zno_nonzero_dac:.2f}"
        ),
        "",
        "At Ag WIE = 0.18:",
        (
            "Ag Two-Step durability = "
            f"{low_ag['Durability']['Ag Two-Step']:.3f}"
        ),
        (
            "Ag Two-Step antibacterial = "
            f"{low_ag['Antibacterial']['Ag Two-Step']:.3f}"
        ),
        (
            "Ag One-Step durability = "
            f"{low_ag['Durability']['Ag One-Step']:.3f}"
        ),
        (
            "Ag One-Step antibacterial = "
            f"{low_ag['Antibacterial']['Ag One-Step']:.3f}"
        ),
        (
            "DC Hybrid Padding-Squeezing = "
            f"{low_ag['DC share']['Hybrid Padding-Squeezing']:.1f}%"
        ),
        (
            "DC Ag Two-Step = "
            f"{low_ag['DC share']['Ag Two-Step']:.1f}%"
        ),
        (
            "DC ZnO-Starch = "
            f"{low_ag['DC share']['ZnO Starch']:.1f}%"
        ),
        (
            "DAC Hybrid Padding-Squeezing = "
            f"{low_ag['DAC share']['Hybrid Padding-Squeezing']:.1f}%"
        ),
        (
            "DAC Ag Two-Step = "
            f"{low_ag['DAC share']['Ag Two-Step']:.1f}%"
        ),
        (
            "DAC Hybrid In-Situ = "
            f"{low_ag['DAC share']['Hybrid In-Situ']:.2f}%"
        ),
        (
            "DAC ZnO-Starch = "
            f"{low_ag['DAC share']['ZnO Starch']:.2f}%"
        ),
        (
            "Combined leading-pair DC share = "
            f"{low_ag['DC share']['Hybrid Padding-Squeezing'] + low_ag['DC share']['Ag Two-Step']:.1f}%"
        ),
        (
            "Combined leading-pair DAC share = "
            f"{low_ag['DAC share']['Hybrid Padding-Squeezing'] + low_ag['DAC share']['Ag Two-Step']:.1f}%"
        ),
        "",
        "WIE-parameter sensitivity",
        "-------------------------",
        (
            "ZnO-Starch enters DAC in "
            f"{len(zno_dac_nonzero_scenarios)} "
            "of 20 scenarios."
        ),
        (
            "Maximum ZnO-Starch DAC share = "
            f"{max_zno_dac:.3f}%"
        ),
        (
            "Maximum ZnO-Starch DC share = "
            f"{max_zno_dc:.1f}%"
        ),
        (
            "Minimum combined leading-pair "
            f"DAC share = {min_pair_dac:.1f}%"
        ),
        (
            "Minimum combined leading-pair "
            f"DC share = {min_pair_dc:.1f}%"
        ),
        (
            "Ag-excluded Hybrid Padding-Squeezing "
            f"range starts at {min_noag_pad:.1f}%"
        ),
        (
            "Ag-excluded maximum Hybrid In-Situ "
            f"share = {max_noag_hybrid:.1f}%"
        ),
        (
            "Ag-excluded maximum ZnO-Starch "
            f"share = {max_noag_zno:.1f}%"
        ),
        "",
        "ZnO optimistic-persistence sensitivity",
        "--------------------------------------",
        (
            "Ag-excluded baseline: "
            "Hybrid Padding-Squeezing "
            f"{zno_baseline['Hybrid Padding-Squeezing']:.1f}%, "
            "Hybrid In-Situ "
            f"{zno_baseline['Hybrid In-Situ']:.1f}%, "
            "ZnO-Starch "
            f"{zno_baseline['ZnO Starch']:.1f}%"
        ),
        (
            "Ag-excluded antibacterial-only: "
            "Hybrid Padding-Squeezing "
            f"{zno_antibacterial['Hybrid Padding-Squeezing']:.1f}%, "
            "Hybrid In-Situ "
            f"{zno_antibacterial['Hybrid In-Situ']:.1f}%, "
            "ZnO-Starch "
            f"{zno_antibacterial['ZnO Starch']:.1f}%"
        ),
        (
            "Ag-excluded durability-only: "
            "Hybrid Padding-Squeezing "
            f"{zno_durability['Hybrid Padding-Squeezing']:.1f}%, "
            "Hybrid In-Situ "
            f"{zno_durability['Hybrid In-Situ']:.1f}%, "
            "ZnO-Starch "
            f"{zno_durability['ZnO Starch']:.1f}%"
        ),
        (
            "Ag-excluded joint optimistic: "
            "Hybrid Padding-Squeezing "
            f"{zno_joint['Hybrid Padding-Squeezing']:.1f}%, "
            "Hybrid In-Situ "
            f"{zno_joint['Hybrid In-Situ']:.1f}%, "
            "ZnO-Starch "
            f"{zno_joint['ZnO Starch']:.1f}%"
        ),
    ]

    path = (
        OUTPUT_DIR
        / "required_numerical_replacements.txt"
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# =====================================================================
# Run outputs
# =====================================================================

write_baseline_csv()
write_material_average_csv()
write_ag_wie_csv()
write_wie_parameter_csv()
write_zno_scenarios_csv()
write_horizon_csv()
write_replacement_summary()

make_decision_region_figure(
    BASELINE_DURABILITY,
    BASELINE_ANTIBACTERIAL,
    BASELINE_COST_SCORE,
    DAC_FORMULATIONS,
    "Fig1a_DAC_decision_regions",
)

make_dc_weight_curves()

make_material_average_figure()

make_decision_region_figure(
    BASELINE_DURABILITY,
    BASELINE_ANTIBACTERIAL,
    BASELINE_COST_SCORE,
    AG_EXCLUDED_FORMULATIONS,
    "Fig6_Ag_excluded_DAC_regions",
)

make_pareto_figure()

make_wie_parameter_heatmaps()

make_ag_wie_figure(
    "DAC share",
    DAC_FORMULATIONS,
    "Supp_Fig3a_Ag_WIE_DAC",
)

make_ag_wie_figure(
    "DC share",
    FORMULATIONS,
    "Supp_Fig3b_Ag_WIE_DC",
)

make_horizon_line_figure(
    "Durability",
    FORMULATIONS,
    "Durability score",
    "Supp_S5a_durability_by_horizon",
    (0.0, 1.02),
)

make_horizon_line_figure(
    "Antibacterial",
    DAC_FORMULATIONS,
    "Antibacterial score",
    "Supp_S5b_antibacterial_by_horizon",
    (0.0, 1.02),
)

make_horizon_line_figure(
    "DC share",
    FORMULATIONS,
    "Winning share of tested weight space (%)",
    "Supp_S5c_DC_shares_by_horizon",
    (0.0, 100.0),
)

make_horizon_line_figure(
    "DAC share",
    DAC_FORMULATIONS,
    "Winning share of tested weight space (%)",
    "Supp_S5d_DAC_matched_horizon",
    (0.0, 100.0),
)

make_horizon_line_figure(
    "DAC A fixed",
    DAC_FORMULATIONS,
    "Winning share of tested weight space (%)",
    "Supp_S5e_DAC_antibacterial_fixed",
    (0.0, 100.0),
)

print("Revision complete.")
print(f"F(2) = {F2:.9f}")
print(f"F(10) = {F10:.9f}")
print(
    "F(10)/F(2) = "
    f"{LOSS_SCALER_2_TO_10:.9f}"
)
print(
    "ZnO-Starch durability = "
    f"{BASELINE_DURABILITY['ZnO Starch']:.6f}"
)
print(
    "ZnO-SDS durability = "
    f"{BASELINE_DURABILITY['ZnO SDS']:.6f}"
)
print(
    "Outputs saved to: "
    f"{OUTPUT_DIR.resolve()}"
)
