
"""
Sensitivity analysis for changing the common laundering horizon from R10
to R2, R5, and R8.

The raw durability and antibacterial data below are transcribed directly
from "Original Data Extracted From Sources(7).xlsx".

Outputs
-------
1. horizon_sensitivity_scores.csv
2. horizon_sensitivity_winning_shares.csv
3. horizon_sensitivity_durability_scores.png
4. horizon_sensitivity_DC_winning_shares.png
5. horizon_sensitivity_DAC_winning_shares.png

Methodological logic
--------------------
The original baseline assumes:
- 75% of cumulative 30-cycle loss occurs by wash 2;
- cumulative loss by wash 10 is 80% of cumulative 30-cycle loss;
- the additional loss between washes 2 and 10 is distributed uniformly.

For any target horizon H in [2, 10]:

    F(H) = 0.75 + (0.80 - 0.75) * (H - 2) / (10 - 2)

where F(H) is the assumed fraction of cumulative 30-cycle loss occurring
by horizon H.

If Wmax < 2:
    D_H = 1 - [F(H)/0.75] * [1 - Rmax * (Wmax/2)]

If 2 <= Wmax < H:
    c_H(Wmax) = F(Wmax) / F(H)
    D_H = 1 - (1 - Rmax) / c_H(Wmax)

If Wmax >= H:
    D_H is obtained directly or by linear interpolation at horizon H.

At H = 10, these equations reduce to the original R10 rules:
    F(10)/0.75 = 0.80/0.75 = 16/15
and
    c_10(W) = 15/16 + (1 - 15/16) * (W - 2)/8.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------

OUTPUT_DIR = Path("horizon_sensitivity_outputs")
TARGET_HORIZONS = (2.0, 5.0, 8.0, 10.0)

# True: durability and antibacterial performance are both evaluated at
# the same target horizon. This is the internally consistent default.
#
# False: only durability changes from R10 to R2/R5/R8, while the
# antibacterial criterion remains fixed at its R10 score.
RECALCULATE_ANTIBACTERIAL_AT_SAME_HORIZON = True

WEIGHT_STEP = 0.01
Q10 = 2.0
TEMPERATURE_WEIGHT = 0.5
REFERENCE_TEMPERATURE_C = 40.0
REFERENCE_DURATION_MIN = 45.0


# ---------------------------------------------------------------------
# RAW DATA TRANSCRIBED FROM THE UPLOADED EXCEL FILE
# Values are (unstandardized washing cycles, reported quantity).
# ---------------------------------------------------------------------

DURABILITY_RAW: Dict[str, List[Tuple[float, float]]] = {
    "TiO2 Two-Step Dipping": [
        (5, 23.16), (10, 12.48), (15, 10.83), (20, 4.45)
    ],
    "TiO2 Alkaline Hydrolysis": [
        (0, 3.3), (10, 2.3)
    ],
    "Hybrid Padding-Squeezing": [
        (0, 1.449), (15, 0.112), (30, 0.057)
    ],
    "Hybrid In-Situ": [
        (0, 44500), (30, 9500)
    ],
    "ZnO Starch": [
        (0, 35), (5, 34.9), (10, 33.4)
    ],
    "ZnO SDS": [
        (0, 6.74), (5, 5.91), (10, 4.32)
    ],
    "Ag Two-Step": [
        (0, 180), (10, 165), (30, 148)
    ],
    "Ag One-Step": [
        (0, 1059), (20, 871)
    ],
}

ANTIBACTERIAL_RAW: Dict[str, Dict[str, List[Tuple[float, float]]]] = {
    "Hybrid Padding-Squeezing": {
        "E. coli": [(0, 99.870), (15, 99.774), (30, 97.019)],
        "S. aureus": [(0, 99.931), (15, 99.855), (30, 99.445)],
    },
    "Hybrid In-Situ": {
        "E. coli": [(0, 100.0), (30, 97.8)],
        "S. aureus": [(0, 100.0), (10, 99.3)],
    },
    "ZnO Starch": {
        "E. coli": [(0, 100.0), (5, 100.0), (10, 76.4)],
        "S. aureus": [(0, 100.0), (5, 98.1), (10, 96.2)],
    },
    "ZnO SDS": {
        "E. coli": [(0, 91.0), (10, 89.0)],
        "S. aureus": [(0, 92.3), (10, 90.2)],
    },
    "Ag Two-Step": {
        "E. coli": [(0, 99.99), (30, 98.92)],
        "S. aureus": [(0, 99.99), (30, 99.08)],
    },
    "Ag One-Step": {
        "E. coli": [(5, 95.87), (10, 93.59)],
        "S. aureus": [(5, 94.59), (10, 92.23)],
    },
}


# Laundering conditions used to calculate WIE.
# None means the source did not report temperature/duration and WIE=1
# is retained as in the baseline model.
WASH_CONDITIONS: Dict[str, Tuple[float | None, float | None]] = {
    "TiO2 Two-Step Dipping": (40.0, 45.0),
    "TiO2 Alkaline Hydrolysis": (60.0, 20.0),
    "Hybrid Padding-Squeezing": (60.0, 30.0),
    "Hybrid In-Situ": (40.0, 45.0),
    "ZnO Starch": (22.0, 5.0),  # duration imputed as in the manuscript
    "ZnO SDS": (22.0, 5.0),
    "Ag Two-Step": (None, None),
    "Ag One-Step": (None, None),
}


# Laboratory-scale raw-material costs in USD per gram of fabric.
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
# BASIC CALCULATION FUNCTIONS
# ---------------------------------------------------------------------

def calculate_wie(
    temperature_c: float | None,
    duration_min: float | None,
) -> float:
    """Calculate WIE using the baseline manuscript equation."""
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


def cumulative_loss_fraction_30(horizon: float) -> float:
    """
    Fraction of cumulative 30-cycle loss assumed to have occurred
    by the selected horizon.

    The function is only intended for 2 <= horizon <= 10 in this
    sensitivity analysis.
    """
    if not 2.0 <= horizon <= 10.0:
        raise ValueError(
            "This sensitivity model is defined for horizons from 2 to 10."
        )

    return 0.75 + (0.80 - 0.75) * (horizon - 2.0) / 8.0


def estimated_initial_quantity(
    standardized_points: Sequence[Tuple[float, float]],
) -> float:
    """
    Return the reported cycle-0 quantity, or estimate the intercept
    by linear least squares when cycle 0 is missing.
    """
    for cycle, quantity in standardized_points:
        if math.isclose(cycle, 0.0, abs_tol=1e-12):
            return float(quantity)

    x = np.array([p[0] for p in standardized_points], dtype=float)
    y = np.array([p[1] for p in standardized_points], dtype=float)

    if len(x) < 2:
        raise ValueError(
            "At least two observations are required to estimate cycle 0."
        )

    slope, intercept = np.polyfit(x, y, 1)

    if intercept <= 0:
        raise ValueError(
            f"Estimated cycle-0 intercept is non-positive: {intercept}"
        )

    return float(intercept)


def interpolate_with_cycle_zero(
    standardized_points: Sequence[Tuple[float, float]],
    target_horizon: float,
    initial_quantity: float,
) -> float:
    """
    Obtain the quantity at the target horizon by exact selection or
    linear interpolation. If cycle 0 was absent, the fitted intercept
    is inserted as the cycle-0 point.
    """
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
        if math.isclose(cycle, target_horizon, abs_tol=1e-12):
            return quantity

    if target_horizon < points[0][0] or target_horizon > points[-1][0]:
        raise ValueError(
            "Interpolation requested outside the observed range."
        )

    for (cycle_a, quantity_a), (cycle_b, quantity_b) in zip(
        points[:-1], points[1:]
    ):
        if cycle_a < target_horizon < cycle_b:
            fraction = (
                (target_horizon - cycle_a) / (cycle_b - cycle_a)
            )
            return quantity_a + fraction * (quantity_b - quantity_a)

    raise RuntimeError("No interpolation interval was found.")


def durability_score_at_horizon(
    formulation: str,
    target_horizon: float,
) -> float:
    """
    Calculate D_H for one formulation at R2, R5, R8, or R10.

    Cases
    -----
    1. Wmax >= H:
       obtain retention directly or by interpolation.

    2. Wmax < 2:
       D_H = 1 - [F(H)/0.75] *
                   [1 - Rmax * (Wmax/2)]

    3. 2 <= Wmax < H:
       c_H(Wmax) = F(Wmax)/F(H)
       D_H = 1 - (1 - Rmax)/c_H(Wmax)
    """
    wie = WIE[formulation]

    standardized_points = sorted(
        (raw_cycle * wie, quantity)
        for raw_cycle, quantity in DURABILITY_RAW[formulation]
    )

    initial_quantity = estimated_initial_quantity(standardized_points)
    max_cycle, max_quantity = standardized_points[-1]
    max_retention = max_quantity / initial_quantity

    if max_cycle >= target_horizon:
        quantity_h = interpolate_with_cycle_zero(
            standardized_points,
            target_horizon,
            initial_quantity,
        )
        score = quantity_h / initial_quantity

    elif max_cycle < 2.0:
        target_loss_fraction = cumulative_loss_fraction_30(
            target_horizon
        )

        estimated_retention_at_two = (
            max_retention * (max_cycle / 2.0)
        )

        score = 1.0 - (
            target_loss_fraction / 0.75
        ) * (1.0 - estimated_retention_at_two)

    else:
        observed_loss_fraction = cumulative_loss_fraction_30(
            max_cycle
        )
        target_loss_fraction = cumulative_loss_fraction_30(
            target_horizon
        )
        evidence_coverage = (
            observed_loss_fraction / target_loss_fraction
        )

        score = 1.0 - (
            1.0 - max_retention
        ) / evidence_coverage

    return float(np.clip(score, 0.0, 1.0))


def linear_interpolate_or_extrapolate(
    points: Sequence[Tuple[float, float]],
    target_horizon: float,
    wie: float,
) -> float:
    """
    Linear interpolation within the observed range, or linear
    extrapolation from the nearest two observations outside it.
    """
    standardized = sorted(
        (raw_cycle * wie, value)
        for raw_cycle, value in points
    )

    for cycle, value in standardized:
        if math.isclose(cycle, target_horizon, abs_tol=1e-12):
            return float(value)

    if len(standardized) < 2:
        raise ValueError(
            "At least two antibacterial observations are required."
        )

    if target_horizon < standardized[0][0]:
        point_a, point_b = standardized[0], standardized[1]
    elif target_horizon > standardized[-1][0]:
        point_a, point_b = standardized[-2], standardized[-1]
    else:
        for candidate_a, candidate_b in zip(
            standardized[:-1], standardized[1:]
        ):
            if (
                candidate_a[0]
                < target_horizon
                < candidate_b[0]
            ):
                point_a, point_b = candidate_a, candidate_b
                break
        else:
            raise RuntimeError(
                "No antibacterial interpolation interval was found."
            )

    cycle_a, value_a = point_a
    cycle_b, value_b = point_b

    value_h = value_a + (
        (target_horizon - cycle_a) / (cycle_b - cycle_a)
    ) * (value_b - value_a)

    return float(np.clip(value_h, 0.0, 100.0))


def antibacterial_score_at_horizon(
    formulation: str,
    target_horizon: float,
) -> float:
    """Average the E. coli and S. aureus reduction scores."""
    species_scores: List[float] = []

    for species_points in ANTIBACTERIAL_RAW[formulation].values():
        reduction_percent = linear_interpolate_or_extrapolate(
            species_points,
            target_horizon,
            WIE[formulation],
        )
        species_scores.append(reduction_percent / 100.0)

    return float(np.mean(species_scores))


def calculate_cost_scores() -> Dict[str, float]:
    """
    Reverse logarithmic min-max normalization.

    Costs below USD 0.01/g fabric use USD 0.01 as the lower
    normalization anchor, matching the manuscript.
    """
    anchored_costs = {
        formulation: max(cost, 0.01)
        for formulation, cost
        in RAW_MATERIAL_COST_USD_PER_G_FABRIC.items()
    }

    minimum = min(anchored_costs.values())
    maximum = max(anchored_costs.values())

    denominator = math.log(maximum) - math.log(minimum)

    return {
        formulation: (
            math.log(maximum) - math.log(cost)
        ) / denominator
        for formulation, cost in anchored_costs.items()
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
            value, maximum, rel_tol=0.0, abs_tol=tolerance
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
    shares = {name: 0.0 for name in formulations}

    number_of_steps = int(round(1.0 / WEIGHT_STEP))
    tested_weights = [
        i * WEIGHT_STEP for i in range(number_of_steps + 1)
    ]

    for durability_weight in tested_weights:
        scores = {
            name: (
                durability_weight * durability_scores[name]
                + (1.0 - durability_weight) * COST_SCORE[name]
            )
            for name in formulations
        }

        winners = top_winners(scores)
        tie_share = 1.0 / len(winners)

        for winner in winners:
            shares[winner] += tie_share

    total = len(tested_weights)
    return {
        name: 100.0 * count / total
        for name, count in shares.items()
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
    shares = {name: 0.0 for name in formulations}
    total_combinations = 0

    number_of_steps = int(round(1.0 / WEIGHT_STEP))

    for durability_index in range(number_of_steps + 1):
        durability_weight = durability_index * WEIGHT_STEP

        for antibacterial_index in range(
            number_of_steps - durability_index + 1
        ):
            antibacterial_weight = (
                antibacterial_index * WEIGHT_STEP
            )
            cost_weight = (
                1.0
                - durability_weight
                - antibacterial_weight
            )

            scores = {
                name: (
                    durability_weight * durability_scores[name]
                    + antibacterial_weight
                    * antibacterial_scores[name]
                    + cost_weight * COST_SCORE[name]
                )
                for name in formulations
            }

            winners = top_winners(scores)
            tie_share = 1.0 / len(winners)

            for winner in winners:
                shares[winner] += tie_share

            total_combinations += 1

    return {
        name: 100.0 * count / total_combinations
        for name, count in shares.items()
    }


def pareto_front(
    criteria: Mapping[str, Sequence[float]],
) -> Dict[str, bool]:
    """
    Return True for alternatives not dominated by another alternative.
    All criteria are benefit criteria: higher is better.
    """
    names = list(criteria)
    front = {name: True for name in names}

    for name_i in names:
        values_i = criteria[name_i]

        for name_j in names:
            if name_i == name_j:
                continue

            values_j = criteria[name_j]

            at_least_as_good = all(
                value_j >= value_i - 1e-12
                for value_i, value_j
                in zip(values_i, values_j)
            )
            strictly_better = any(
                value_j > value_i + 1e-12
                for value_i, value_j
                in zip(values_i, values_j)
            )

            if at_least_as_good and strictly_better:
                front[name_i] = False
                break

    return front


# ---------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------

def run_analysis() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    horizons = list(TARGET_HORIZONS)
    formulations = list(DURABILITY_RAW)
    antibacterial_formulations = list(ANTIBACTERIAL_RAW)

    durability_by_horizon: Dict[
        float, Dict[str, float]
    ] = {}

    antibacterial_by_horizon: Dict[
        float, Dict[str, float]
    ] = {}

    for horizon in horizons:
        durability_by_horizon[horizon] = {
            name: durability_score_at_horizon(name, horizon)
            for name in formulations
        }

        antibacterial_horizon = (
            horizon
            if RECALCULATE_ANTIBACTERIAL_AT_SAME_HORIZON
            else 10.0
        )

        antibacterial_by_horizon[horizon] = {
            name: antibacterial_score_at_horizon(
                name, antibacterial_horizon
            )
            for name in antibacterial_formulations
        }

    # -------------------------------------------------------------
    # Score table
    # -------------------------------------------------------------
    score_csv = OUTPUT_DIR / "horizon_sensitivity_scores.csv"

    score_headers = [
        "Formulation",
        "WIE",
        "Raw material cost (USD/g fabric)",
        "Cost score",
    ]

    for horizon in horizons:
        label = int(horizon)
        score_headers.append(f"Durability R{label}")
        score_headers.append(f"Antibacterial R{label}")

    with score_csv.open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=score_headers
        )
        writer.writeheader()

        for name in formulations:
            row = {
                "Formulation": name,
                "WIE": WIE[name],
                "Raw material cost (USD/g fabric)":
                    RAW_MATERIAL_COST_USD_PER_G_FABRIC[name],
                "Cost score": COST_SCORE[name],
            }

            for horizon in horizons:
                label = int(horizon)
                row[f"Durability R{label}"] = (
                    durability_by_horizon[horizon][name]
                )
                row[f"Antibacterial R{label}"] = (
                    antibacterial_by_horizon[horizon].get(
                        name, ""
                    )
                )

            writer.writerow(row)

    # -------------------------------------------------------------
    # MCDA winning shares and Pareto membership
    # -------------------------------------------------------------
    winning_csv = (
        OUTPUT_DIR / "horizon_sensitivity_winning_shares.csv"
    )

    winning_rows: List[Dict[str, object]] = []

    for horizon in horizons:
        durability = durability_by_horizon[horizon]
        antibacterial = antibacterial_by_horizon[horizon]

        dc_shares = winning_shares_dc(durability)
        dc_front = pareto_front({
            name: (durability[name], COST_SCORE[name])
            for name in formulations
        })

        for name in formulations:
            winning_rows.append({
                "Model": "Durability-Cost",
                "Horizon": f"R{int(horizon)}",
                "Formulation": name,
                "Winning share (%)": dc_shares[name],
                "Pareto optimal": dc_front[name],
            })

        dac_shares = winning_shares_dac(
            durability, antibacterial
        )
        dac_front = pareto_front({
            name: (
                durability[name],
                antibacterial[name],
                COST_SCORE[name],
            )
            for name in antibacterial_formulations
        })

        for name in antibacterial_formulations:
            winning_rows.append({
                "Model": "Durability-Antibacterial-Cost",
                "Horizon": f"R{int(horizon)}",
                "Formulation": name,
                "Winning share (%)": dac_shares[name],
                "Pareto optimal": dac_front[name],
            })

    with winning_csv.open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Model",
                "Horizon",
                "Formulation",
                "Winning share (%)",
                "Pareto optimal",
            ],
        )
        writer.writeheader()
        writer.writerows(winning_rows)

    # -------------------------------------------------------------
    # Figure 1: durability score by target horizon
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    for name in formulations:
        plt.plot(
            horizons,
            [
                durability_by_horizon[h][name]
                for h in horizons
            ],
            marker="o",
            label=name,
        )

    plt.xlabel("Common reference-equivalent wash horizon")
    plt.ylabel("Durability score")
    plt.title(
        "Sensitivity of formulation durability scores "
        "to the evaluation horizon"
    )
    plt.xticks(horizons, [f"R{int(h)}" for h in horizons])
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "horizon_sensitivity_durability_scores.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # -------------------------------------------------------------
    # Figure 2: DC winning shares
    # -------------------------------------------------------------
    dc_plot_data: Dict[str, List[float]] = {
        name: [] for name in formulations
    }

    for horizon in horizons:
        shares = winning_shares_dc(
            durability_by_horizon[horizon]
        )
        for name in formulations:
            dc_plot_data[name].append(shares[name])

    plt.figure(figsize=(9, 6))

    for name in formulations:
        if max(dc_plot_data[name]) > 0:
            plt.plot(
                horizons,
                dc_plot_data[name],
                marker="o",
                label=name,
            )

    plt.xlabel("Common reference-equivalent wash horizon")
    plt.ylabel("Winning share of tested weight space (%)")
    plt.title(
        "Durability-Cost model sensitivity "
        "to the evaluation horizon"
    )
    plt.xticks(horizons, [f"R{int(h)}" for h in horizons])
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "horizon_sensitivity_DC_winning_shares.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # -------------------------------------------------------------
    # Figure 3: DAC winning shares
    # -------------------------------------------------------------
    dac_plot_data: Dict[str, List[float]] = {
        name: [] for name in antibacterial_formulations
    }

    for horizon in horizons:
        shares = winning_shares_dac(
            durability_by_horizon[horizon],
            antibacterial_by_horizon[horizon],
        )
        for name in antibacterial_formulations:
            dac_plot_data[name].append(shares[name])

    plt.figure(figsize=(9, 6))

    for name in antibacterial_formulations:
        if max(dac_plot_data[name]) > 0:
            plt.plot(
                horizons,
                dac_plot_data[name],
                marker="o",
                label=name,
            )

    antibacterial_note = (
        "Antibacterial score recalculated at the same horizon"
        if RECALCULATE_ANTIBACTERIAL_AT_SAME_HORIZON
        else "Antibacterial score held at R10"
    )

    plt.xlabel("Common reference-equivalent wash horizon")
    plt.ylabel("Winning share of tested weight space (%)")
    plt.title(
        "Durability-Antibacterial-Cost model sensitivity\n"
        f"({antibacterial_note})"
    )
    plt.xticks(horizons, [f"R{int(h)}" for h in horizons])
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "horizon_sensitivity_DAC_winning_shares.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # -------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------
    print("\nWIE values")
    print("-" * 72)
    for name in formulations:
        print(f"{name:32s} {WIE[name]:.6f}")

    print("\nDurability scores")
    print("-" * 90)
    header = f"{'Formulation':32s}" + "".join(
        f"{f'R{int(h)}':>12s}" for h in horizons
    )
    print(header)

    for name in formulations:
        values = "".join(
            f"{durability_by_horizon[h][name]:12.3f}"
            for h in horizons
        )
        print(f"{name:32s}{values}")

    print("\nNon-zero winning shares")
    print("-" * 90)

    for horizon in horizons:
        dc = winning_shares_dc(
            durability_by_horizon[horizon]
        )
        dac = winning_shares_dac(
            durability_by_horizon[horizon],
            antibacterial_by_horizon[horizon],
        )

        dc_nonzero = {
            name: round(value, 2)
            for name, value in dc.items()
            if value > 1e-12
        }
        dac_nonzero = {
            name: round(value, 2)
            for name, value in dac.items()
            if value > 1e-12
        }

        print(f"R{int(horizon)} DC : {dc_nonzero}")
        print(f"R{int(horizon)} DAC: {dac_nonzero}")

    print(f"\nFiles saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    run_analysis()
