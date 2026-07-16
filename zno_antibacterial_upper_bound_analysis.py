
"""
ZnO antibacterial extrapolation upper-bound check

Purpose
-------
Compare the baseline linear-extrapolation treatment of the two ZnO formulations
with an optimistic upper-bound scenario in which the final observed antibacterial
value is carried forward to the 10-reference-wash horizon.

Data sources
------------
1. Formulation-level durability, antibacterial, and cost scores are taken from
   Table 4 of the manuscript.
2. Raw ZnO antibacterial observations are copied from:
   "Original Data Extracted From Sources(3).xlsx"

Outputs
-------
- zno_antibacterial_scores.csv
- winner_shares.csv
- winner_map_baseline_with_ag.png
- winner_map_optimistic_with_ag.png
- winner_map_baseline_without_ag.png
- winner_map_optimistic_without_ag.png

Required packages
-----------------
numpy, matplotlib
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent / "zno_upper_bound_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_REFERENCE_WASHES = 10.0
WEIGHT_STEP = 0.01

# Baseline WIE assumptions used in the manuscript.
T_REF_C = 40.0
T_STUDY_C = 22.0
T_REF_MIN = 45.0
T_STUDY_MIN = 5.0  # ZnO-Starch duration imputed from ZnO-SDS.
Q10 = 2.0
ALPHA = 0.5


@dataclass(frozen=True)
class Formulation:
    name: str
    durability: float
    antibacterial: float
    cost: float


# ---------------------------------------------------------------------------
# 2. Raw ZnO antibacterial data from the uploaded Excel workbook
#    Each tuple is: (unstandardized washing cycles, bacterial reduction %)
# ---------------------------------------------------------------------------

ZNO_RAW_DATA: Mapping[str, Mapping[str, Sequence[Tuple[float, float]]]] = {
    "ZnO Starch": {
        "E. coli": [(0.0, 100.0), (5.0, 100.0), (10.0, 76.4)],
        "S. aureus": [(0.0, 100.0), (5.0, 98.1), (10.0, 96.2)],
    },
    "ZnO SDS": {
        "E. coli": [(0.0, 91.0), (10.0, 89.0)],
        "S. aureus": [(0.0, 92.3), (10.0, 90.2)],
    },
}


# ---------------------------------------------------------------------------
# 3. WIE and antibacterial-score calculation
# ---------------------------------------------------------------------------

def calculate_wie(
    t_study_c: float,
    duration_study_min: float,
    q10: float = Q10,
    alpha: float = ALPHA,
    t_ref_c: float = T_REF_C,
    duration_ref_min: float = T_REF_MIN,
) -> float:
    """Calculate the Washing Intensity Equivalent used in the manuscript."""
    if duration_study_min <= 0 or duration_ref_min <= 0:
        raise ValueError("Wash durations must be positive.")
    if q10 <= 0:
        raise ValueError("Q10 must be positive.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie between 0 and 1.")

    thermal_term = q10 ** ((t_study_c - t_ref_c) / 10.0)
    duration_term = duration_study_min / duration_ref_min
    return (thermal_term ** alpha) * (duration_term ** (1.0 - alpha))


def convert_to_reference_cycles(
    observations: Sequence[Tuple[float, float]],
    wie: float,
) -> List[Tuple[float, float]]:
    """Convert reported washing cycles into reference-equivalent cycles."""
    return [(cycles * wie, value) for cycles, value in observations]


def baseline_linear_estimate(
    observations: Sequence[Tuple[float, float]],
    wie: float,
    target_cycles: float = TARGET_REFERENCE_WASHES,
) -> float:
    """
    Estimate bacterial reduction at the target reference-wash horizon.

    - Interpolate if the target lies within the observed standardized range.
    - Extrapolate from the final two standardized observations if the target
      lies beyond the reported range.
    - Clip the estimate to 0-100%.
    """
    converted = sorted(convert_to_reference_cycles(observations, wie))
    x = np.asarray([p[0] for p in converted], dtype=float)
    y = np.asarray([p[1] for p in converted], dtype=float)

    if len(x) < 2:
        raise ValueError("At least two observations are required.")

    # Direct observation.
    exact = np.where(np.isclose(x, target_cycles, atol=1e-12))[0]
    if exact.size:
        estimate = float(y[exact[0]])

    # Interpolation.
    elif x[0] < target_cycles < x[-1]:
        estimate = float(np.interp(target_cycles, x, y))

    # Forward extrapolation from the two latest observations.
    elif target_cycles > x[-1]:
        x1, x2 = x[-2], x[-1]
        y1, y2 = y[-2], y[-1]
        if math.isclose(x1, x2):
            raise ValueError("The final two standardized cycle values are identical.")
        slope = (y2 - y1) / (x2 - x1)
        estimate = float(y2 + slope * (target_cycles - x2))

    # Backward extrapolation is not expected here, but is handled explicitly.
    else:
        x1, x2 = x[0], x[1]
        y1, y2 = y[0], y[1]
        if math.isclose(x1, x2):
            raise ValueError("The first two standardized cycle values are identical.")
        slope = (y2 - y1) / (x2 - x1)
        estimate = float(y1 + slope * (target_cycles - x1))

    return float(np.clip(estimate, 0.0, 100.0))


def optimistic_last_observation(
    observations: Sequence[Tuple[float, float]],
) -> float:
    """
    Optimistic upper-bound assumption:
    no further antibacterial decline after the final reported observation.
    """
    if not observations:
        raise ValueError("Observations cannot be empty.")
    return float(sorted(observations)[-1][1])


def calculate_zno_scores() -> Dict[str, Dict[str, float]]:
    """Calculate baseline and optimistic overall antibacterial scores."""
    wie = calculate_wie(T_STUDY_C, T_STUDY_MIN)

    results: Dict[str, Dict[str, float]] = {}
    for formulation, species_data in ZNO_RAW_DATA.items():
        baseline_species: Dict[str, float] = {}
        optimistic_species: Dict[str, float] = {}

        for species, observations in species_data.items():
            baseline_species[species] = (
                baseline_linear_estimate(observations, wie) / 100.0
            )
            optimistic_species[species] = (
                optimistic_last_observation(observations) / 100.0
            )

        results[formulation] = {
            "WIE": wie,
            "maximum_reference_washes": max(
                cycle for cycle, _ in convert_to_reference_cycles(
                    next(iter(species_data.values())), wie
                )
            ),
            "baseline_E_coli": baseline_species["E. coli"],
            "baseline_S_aureus": baseline_species["S. aureus"],
            "baseline_overall": float(np.mean(list(baseline_species.values()))),
            "optimistic_E_coli": optimistic_species["E. coli"],
            "optimistic_S_aureus": optimistic_species["S. aureus"],
            "optimistic_overall": float(np.mean(list(optimistic_species.values()))),
        }

    # Checks against the rounded baseline scores reported in the manuscript.
    if not math.isclose(
        results["ZnO Starch"]["baseline_overall"], 0.395, abs_tol=0.002
    ):
        raise RuntimeError("ZnO-Starch baseline score does not reproduce the manuscript.")

    if not math.isclose(
        results["ZnO SDS"]["baseline_overall"], 0.802, abs_tol=0.002
    ):
        raise RuntimeError("ZnO-SDS baseline score does not reproduce the manuscript.")

    return results


# ---------------------------------------------------------------------------
# 4. Formulation-level MCDA data from manuscript Table 4
# ---------------------------------------------------------------------------

def build_formulations(
    zno_scores: Mapping[str, Mapping[str, float]],
    scenario: str,
) -> List[Formulation]:
    """
    Construct the six-formulation DAC dataset.

    scenario:
        "baseline"   -> linear extrapolation for ZnO antibacterial values
        "optimistic" -> last observation carried forward for ZnO values
    """
    if scenario not in {"baseline", "optimistic"}:
        raise ValueError("scenario must be 'baseline' or 'optimistic'.")

    score_key = f"{scenario}_overall"

    return [
        Formulation("Hybrid Padding-Squeezing", 0.623, 0.999, 1.000),
        Formulation("Hybrid In-Situ",           0.738, 0.993, 0.496),
        Formulation(
            "ZnO Starch",
            0.805,
            zno_scores["ZnO Starch"][score_key],
            0.000,
        ),
        Formulation(
            "ZnO SDS",
            0.540,
            zno_scores["ZnO SDS"][score_key],
            0.377,
        ),
        Formulation("Ag Two-Step",              0.917, 0.997, 0.663),
        Formulation("Ag One-Step",              0.911, 0.929, 0.203),
    ]


# ---------------------------------------------------------------------------
# 5. Weight-space analysis
# ---------------------------------------------------------------------------

def generate_weight_grid(step: float = WEIGHT_STEP) -> np.ndarray:
    """
    Generate all non-negative (w_D, w_A, w_C) combinations summing to 1.

    step=0.01 produces 5,151 feasible combinations.
    """
    if step <= 0 or step > 1:
        raise ValueError("Weight step must lie in (0, 1].")

    divisions = round(1.0 / step)
    if not math.isclose(divisions * step, 1.0, abs_tol=1e-12):
        raise ValueError("Weight step must divide 1 exactly.")

    rows: List[Tuple[float, float, float]] = []
    for i in range(divisions + 1):
        w_d = i / divisions
        for j in range(divisions + 1 - i):
            w_a = j / divisions
            w_c = 1.0 - w_d - w_a
            rows.append((w_d, w_a, w_c))

    return np.asarray(rows, dtype=float)


def analyse_weight_space(
    formulations: Sequence[Formulation],
    weights: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Return map winners and winning shares.

    Winning shares split tied weight points equally among tied formulations.
    The displayed map uses the first tied formulation only at exact boundaries.
    """
    criteria = np.asarray(
        [[f.durability, f.antibacterial, f.cost] for f in formulations],
        dtype=float,
    )

    scores = weights @ criteria.T
    maxima = scores.max(axis=1, keepdims=True)
    tied = np.isclose(scores, maxima, atol=1e-12, rtol=0.0)

    fractional_wins = tied / tied.sum(axis=1, keepdims=True)
    shares = fractional_wins.sum(axis=0) / len(weights) * 100.0

    map_winner_indices = np.argmax(scores, axis=1)
    share_dict = {
        formulation.name: float(shares[index])
        for index, formulation in enumerate(formulations)
    }
    return map_winner_indices, share_dict


# ---------------------------------------------------------------------------
# 6. Ternary winner-map plotting
# ---------------------------------------------------------------------------

def barycentric_to_xy(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangle vertices:
        lower-left  = Cost
        lower-right = Durability
        top         = Antibacterial
    """
    w_d = weights[:, 0]
    w_a = weights[:, 1]

    x = w_d + 0.5 * w_a
    y = (math.sqrt(3.0) / 2.0) * w_a
    return x, y


def plot_winner_map(
    formulations: Sequence[Formulation],
    weights: np.ndarray,
    winner_indices: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Create one standalone winner-map figure."""
    x, y = barycentric_to_xy(weights)

    # Use Matplotlib's current default color cycle rather than custom colors.
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    formulation_colors = {
        formulation.name: default_colors[index % len(default_colors)]
        for index, formulation in enumerate(formulations)
    }

    figure = plt.figure(figsize=(8.0, 7.2))
    axis = figure.add_subplot(111)

    for index, formulation in enumerate(formulations):
        mask = winner_indices == index
        if np.any(mask):
            axis.scatter(
                x[mask],
                y[mask],
                s=13,
                marker="s",
                linewidths=0,
                color=formulation_colors[formulation.name],
                label=formulation.name,
            )

    triangle_x = [0.0, 1.0, 0.5, 0.0]
    triangle_y = [0.0, 0.0, math.sqrt(3.0) / 2.0, 0.0]
    axis.plot(triangle_x, triangle_y, linewidth=1.0)

    axis.text(-0.035, -0.035, "Cost priority", ha="left", va="top")
    axis.text(1.035, -0.035, "Durability priority", ha="right", va="top")
    axis.text(
        0.5,
        math.sqrt(3.0) / 2.0 + 0.035,
        "Antibacterial priority",
        ha="center",
        va="bottom",
    )

    axis.set_title(title)
    axis.set_aspect("equal")
    axis.set_xlim(-0.08, 1.08)
    axis.set_ylim(-0.08, math.sqrt(3.0) / 2.0 + 0.10)
    axis.axis("off")
    axis.legend(loc="upper right", bbox_to_anchor=(1.22, 1.00), frameon=True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# 7. Output tables
# ---------------------------------------------------------------------------

def write_zno_score_table(
    zno_scores: Mapping[str, Mapping[str, float]],
    output_path: Path,
) -> None:
    fieldnames = [
        "Formulation",
        "WIE",
        "Maximum reference-equivalent washes",
        "Baseline E. coli score",
        "Baseline S. aureus score",
        "Baseline overall antibacterial score",
        "Optimistic E. coli score",
        "Optimistic S. aureus score",
        "Optimistic overall antibacterial score",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for formulation, result in zno_scores.items():
            writer.writerow(
                {
                    "Formulation": formulation,
                    "WIE": f"{result['WIE']:.9f}",
                    "Maximum reference-equivalent washes": (
                        f"{result['maximum_reference_washes']:.6f}"
                    ),
                    "Baseline E. coli score": f"{result['baseline_E_coli']:.6f}",
                    "Baseline S. aureus score": (
                        f"{result['baseline_S_aureus']:.6f}"
                    ),
                    "Baseline overall antibacterial score": (
                        f"{result['baseline_overall']:.6f}"
                    ),
                    "Optimistic E. coli score": (
                        f"{result['optimistic_E_coli']:.6f}"
                    ),
                    "Optimistic S. aureus score": (
                        f"{result['optimistic_S_aureus']:.6f}"
                    ),
                    "Optimistic overall antibacterial score": (
                        f"{result['optimistic_overall']:.6f}"
                    ),
                }
            )


def write_winner_share_table(
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Scenario", "Dataset", "Formulation", "Winning share (%)"],
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 8. Main analysis
# ---------------------------------------------------------------------------

def main() -> None:
    zno_scores = calculate_zno_scores()
    weights = generate_weight_grid(WEIGHT_STEP)

    print(f"WIE for the two ZnO studies: {zno_scores['ZnO Starch']['WIE']:.9f}")
    print(f"Number of feasible DAC weight combinations: {len(weights):,}")
    print()

    write_zno_score_table(
        zno_scores,
        OUTPUT_DIR / "zno_antibacterial_scores.csv",
    )

    winner_share_rows: List[Mapping[str, object]] = []

    for scenario in ("baseline", "optimistic"):
        all_formulations = build_formulations(zno_scores, scenario)
        no_ag_formulations = [
            formulation
            for formulation in all_formulations
            if not formulation.name.startswith("Ag ")
        ]

        analyses = [
            ("with Ag", all_formulations),
            ("without Ag", no_ag_formulations),
        ]

        for dataset_label, formulations in analyses:
            winners, shares = analyse_weight_space(formulations, weights)

            filename_label = dataset_label.replace(" ", "_")
            output_path = OUTPUT_DIR / (
                f"winner_map_{scenario}_{filename_label}.png"
            )

            plot_winner_map(
                formulations=formulations,
                weights=weights,
                winner_indices=winners,
                title=(
                    f"DAC winner map: {scenario.capitalize()} ZnO "
                    f"antibacterial scenario ({dataset_label})"
                ),
                output_path=output_path,
            )

            print(f"{scenario.capitalize()} scenario — {dataset_label}")
            for formulation_name, share in shares.items():
                print(f"  {formulation_name:<28} {share:7.3f}%")
                winner_share_rows.append(
                    {
                        "Scenario": scenario,
                        "Dataset": dataset_label,
                        "Formulation": formulation_name,
                        "Winning share (%)": f"{share:.6f}",
                    }
                )
            print()

    write_winner_share_table(
        winner_share_rows,
        OUTPUT_DIR / "winner_shares.csv",
    )

    print(f"Files saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
