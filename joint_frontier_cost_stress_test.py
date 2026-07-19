
"""
Joint frontier-cost stress test

Purpose
-------
Test how strongly the principal Hybrid Padding-Squeezing / Ag Two-Step
decision structure depends on possible systematic underestimation of the
raw-material costs of BOTH leading formulations.

A common multiplier k is applied simultaneously:

    Cost_Hybrid(k) = 0.01 * k
    Cost_AgTwoStep(k) = 0.13 * k

All other raw-material costs and all durability/antibacterial scores remain
fixed. At every k, cost scores are recalculated for the full dataset using
the same reverse logarithmic min-max transformation as the manuscript.

The analysis records:
1. Pareto-front membership;
2. first non-zero winning region for a formulation outside the baseline pair;
3. first >=5% winning share for formulations outside the baseline pair;
4. DC and DAC winning shares over the full tested range.

Data
----
Raw costs and normalized performance scores are taken from Table 4 of the
current manuscript.

Outputs
-------
- joint_frontier_cost_stress_test.csv
- joint_frontier_cost_summary.txt
- joint_frontier_cost_dc.png
- joint_frontier_cost_dac.png
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

OUTPUT_DIR = Path(__file__).resolve().parent / "joint_frontier_cost_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K_MIN = 1.00
K_MAX = 100.00
K_STEP = 0.01

# A 5% weight-space share is used as the predeclared threshold for a
# materially non-negligible winning region. It is a decision-region reporting
# threshold, not a statistical significance threshold.
MATERIAL_WINNING_SHARE_THRESHOLD = 5.0

# The current manuscript's baseline MCDA uses the three-decimal cost scores
# shown in Table 4. Keeping this setting at 3 reproduces the reported baseline
# shares. Set to None to use full-precision cost scores.
MCDA_COST_SCORE_DECIMALS: int | None = 3

DC_WEIGHT_STEP = 0.01
DAC_WEIGHT_STEP = 0.01


# ---------------------------------------------------------------------------
# 2. Current manuscript data
# ---------------------------------------------------------------------------

DC_NAMES = [
    "TiO2 Two-Step Dipping",
    "TiO2 Alkaline Hydrolysis",
    "Hybrid Padding-Squeezing",
    "Hybrid In-Situ",
    "ZnO Starch",
    "ZnO SDS",
    "Ag Two-Step",
    "Ag One-Step",
]

DURABILITY = np.asarray(
    [0.459, 0.772, 0.623, 0.738, 0.850, 0.564, 0.917, 0.911],
    dtype=float,
)

RAW_COSTS = np.asarray(
    [2.70, 1.66, 0.01, 0.46, 20.01, 1.14, 0.13, 4.28],
    dtype=float,
)

DAC_NAMES = [
    "Hybrid Padding-Squeezing",
    "Hybrid In-Situ",
    "ZnO Starch",
    "ZnO SDS",
    "Ag Two-Step",
    "Ag One-Step",
]

DAC_DC_INDICES = np.asarray([2, 3, 4, 5, 6, 7], dtype=int)

ANTIBACTERIAL = np.asarray(
    [0.999, 0.993, 0.395, 0.802, 0.997, 0.929],
    dtype=float,
)

HYBRID_NAME = "Hybrid Padding-Squeezing"
AG_NAME = "Ag Two-Step"
HYBRID_DC_INDEX = DC_NAMES.index(HYBRID_NAME)
AG_DC_INDEX = DC_NAMES.index(AG_NAME)
HYBRID_DAC_INDEX = DAC_NAMES.index(HYBRID_NAME)
AG_DAC_INDEX = DAC_NAMES.index(AG_NAME)

BASELINE_PAIR_DC = {HYBRID_DC_INDEX, AG_DC_INDEX}
BASELINE_PAIR_DAC = {HYBRID_DAC_INDEX, AG_DAC_INDEX}


# ---------------------------------------------------------------------------
# 3. Cost-score calculation
# ---------------------------------------------------------------------------

def recalculate_cost_scores(
    common_multiplier: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the same multiplier to both principal formulations and recalculate
    every formulation's reverse-log-min-max cost score.

    Higher cost score = more favorable (lower raw-material cost).
    """
    if common_multiplier < 1.0:
        raise ValueError("The common multiplier must be at least 1.")

    scenario_costs = RAW_COSTS.copy()
    scenario_costs[HYBRID_DC_INDEX] *= common_multiplier
    scenario_costs[AG_DC_INDEX] *= common_multiplier

    minimum_cost = float(np.min(scenario_costs))
    maximum_cost = float(np.max(scenario_costs))

    if minimum_cost <= 0:
        raise ValueError("All raw-material costs must be positive.")
    if math.isclose(minimum_cost, maximum_cost):
        raise ValueError("Cost normalization requires a non-zero range.")

    scores = (
        np.log(maximum_cost) - np.log(scenario_costs)
    ) / (
        np.log(maximum_cost) - np.log(minimum_cost)
    )

    if MCDA_COST_SCORE_DECIMALS is not None:
        scores = np.round(scores, MCDA_COST_SCORE_DECIMALS)

    return scenario_costs, scores


# ---------------------------------------------------------------------------
# 4. Weight-space and Pareto functions
# ---------------------------------------------------------------------------

def generate_dc_weights(step: float = DC_WEIGHT_STEP) -> np.ndarray:
    """Generate [w_D, w_C] pairs from 0 to 1 at the requested interval."""
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
    """Generate all feasible [w_D, w_A, w_C] combinations."""
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
    """
    Calculate each formulation's winning share.

    Exact ties are split equally among tied formulations.
    """
    weighted_scores = weights @ criteria.T
    maximum_scores = weighted_scores.max(axis=1, keepdims=True)

    tied = np.isclose(
        weighted_scores,
        maximum_scores,
        atol=1e-12,
        rtol=0.0,
    )

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
    """Return non-dominated formulations; all criteria are higher-is-better."""
    front: List[str] = []

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
            front.append(candidate_name)

    return front


# ---------------------------------------------------------------------------
# 5. Scenario analysis
# ---------------------------------------------------------------------------

def analyse_multiplier(
    common_multiplier: float,
    dc_weights: np.ndarray,
    dac_weights: np.ndarray,
) -> Dict[str, object]:
    scenario_costs, cost_scores = recalculate_cost_scores(
        common_multiplier
    )

    dc_criteria = np.column_stack(
        [DURABILITY, cost_scores]
    )

    dac_criteria = np.column_stack(
        [
            DURABILITY[DAC_DC_INDICES],
            ANTIBACTERIAL,
            cost_scores[DAC_DC_INDICES],
        ]
    )

    dc_share_array = calculate_winning_shares(
        dc_criteria,
        dc_weights,
    )
    dac_share_array = calculate_winning_shares(
        dac_criteria,
        dac_weights,
    )

    dc_shares = {
        name: float(dc_share_array[index])
        for index, name in enumerate(DC_NAMES)
    }
    dac_shares = {
        name: float(dac_share_array[index])
        for index, name in enumerate(DAC_NAMES)
    }

    outside_dc_share = float(
        sum(
            dc_share_array[index]
            for index in range(len(DC_NAMES))
            if index not in BASELINE_PAIR_DC
        )
    )

    outside_dac_share = float(
        sum(
            dac_share_array[index]
            for index in range(len(DAC_NAMES))
            if index not in BASELINE_PAIR_DAC
        )
    )

    outside_dc_winner = max(
        (
            (float(dc_share_array[index]), DC_NAMES[index])
            for index in range(len(DC_NAMES))
            if index not in BASELINE_PAIR_DC
        ),
        key=lambda item: item[0],
    )

    outside_dac_winner = max(
        (
            (float(dac_share_array[index]), DAC_NAMES[index])
            for index in range(len(DAC_NAMES))
            if index not in BASELINE_PAIR_DAC
        ),
        key=lambda item: item[0],
    )

    return {
        "k": common_multiplier,
        "Hybrid raw cost": float(
            scenario_costs[HYBRID_DC_INDEX]
        ),
        "Ag Two-Step raw cost": float(
            scenario_costs[AG_DC_INDEX]
        ),
        "cost scores": {
            name: float(cost_scores[index])
            for index, name in enumerate(DC_NAMES)
        },
        "DC shares": dc_shares,
        "DAC shares": dac_shares,
        "DC Pareto": pareto_front(
            DC_NAMES,
            dc_criteria,
        ),
        "DAC Pareto": pareto_front(
            DAC_NAMES,
            dac_criteria,
        ),
        "Outside-pair DC share": outside_dc_share,
        "Outside-pair DAC share": outside_dac_share,
        "Leading outside DC formulation": outside_dc_winner[1],
        "Leading outside DC share": outside_dc_winner[0],
        "Leading outside DAC formulation": outside_dac_winner[1],
        "Leading outside DAC share": outside_dac_winner[0],
    }


def build_multiplier_values() -> np.ndarray:
    """Generate 1.00, 1.01, ..., 100.00."""
    count = round((K_MAX - K_MIN) / K_STEP)

    return np.round(
        K_MIN + np.arange(count + 1) * K_STEP,
        10,
    )


# ---------------------------------------------------------------------------
# 6. Threshold extraction
# ---------------------------------------------------------------------------

def first_result_where(
    results: Sequence[Mapping[str, object]],
    predicate,
) -> Mapping[str, object] | None:
    for result in results:
        if predicate(result):
            return result
    return None


def pareto_outsiders(
    pareto_names: Sequence[str],
) -> List[str]:
    """Return Pareto formulations outside the baseline leading pair."""
    return [
        name
        for name in pareto_names
        if name not in {HYBRID_NAME, AG_NAME}
    ]


# ---------------------------------------------------------------------------
# 7. Output files
# ---------------------------------------------------------------------------

def write_results_csv(
    results: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    fieldnames = [
        "Common cost multiplier",
        "Hybrid Padding-Squeezing raw cost",
        "Ag Two-Step raw cost",
        "DC Pareto front",
        "DAC Pareto front",
        "Outside-pair DC winning share (%)",
        "Outside-pair DAC winning share (%)",
        "Leading outside DC formulation",
        "Leading outside DC share (%)",
        "Leading outside DAC formulation",
        "Leading outside DAC share (%)",
    ]

    fieldnames.extend(
        [f"Cost score - {name}" for name in DC_NAMES]
    )
    fieldnames.extend(
        [f"DC winning share (%) - {name}" for name in DC_NAMES]
    )
    fieldnames.extend(
        [f"DAC winning share (%) - {name}" for name in DAC_NAMES]
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
                "Common cost multiplier": (
                    f"{float(result['k']):.2f}"
                ),
                "Hybrid Padding-Squeezing raw cost": (
                    f"{float(result['Hybrid raw cost']):.6f}"
                ),
                "Ag Two-Step raw cost": (
                    f"{float(result['Ag Two-Step raw cost']):.6f}"
                ),
                "DC Pareto front": "; ".join(
                    result["DC Pareto"]  # type: ignore[arg-type]
                ),
                "DAC Pareto front": "; ".join(
                    result["DAC Pareto"]  # type: ignore[arg-type]
                ),
                "Outside-pair DC winning share (%)": (
                    f"{float(result['Outside-pair DC share']):.9f}"
                ),
                "Outside-pair DAC winning share (%)": (
                    f"{float(result['Outside-pair DAC share']):.9f}"
                ),
                "Leading outside DC formulation": result[
                    "Leading outside DC formulation"
                ],
                "Leading outside DC share (%)": (
                    f"{float(result['Leading outside DC share']):.9f}"
                ),
                "Leading outside DAC formulation": result[
                    "Leading outside DAC formulation"
                ],
                "Leading outside DAC share (%)": (
                    f"{float(result['Leading outside DAC share']):.9f}"
                ),
            }

            cost_scores = result["cost scores"]  # type: ignore[assignment]
            dc_shares = result["DC shares"]  # type: ignore[assignment]
            dac_shares = result["DAC shares"]  # type: ignore[assignment]

            for name in DC_NAMES:
                row[f"Cost score - {name}"] = (
                    f"{float(cost_scores[name]):.6f}"
                )
                row[f"DC winning share (%) - {name}"] = (
                    f"{float(dc_shares[name]):.9f}"
                )

            for name in DAC_NAMES:
                row[f"DAC winning share (%) - {name}"] = (
                    f"{float(dac_shares[name]):.9f}"
                )

            writer.writerow(row)


def plot_selected_shares(
    results: Sequence[Mapping[str, object]],
    model_key: str,
    formulation_names: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    multipliers = np.asarray(
        [float(result["k"]) for result in results],
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
            multipliers,
            shares,
            linewidth=1.4,
            label=formulation,
        )

    axis.axhline(
        MATERIAL_WINNING_SHARE_THRESHOLD,
        linewidth=1.0,
        linestyle="--",
        label="5% reporting threshold",
    )

    axis.set_xscale("log")
    axis.set_xlabel(
        "Common multiplier applied to Hybrid and Ag Two-Step costs"
    )
    axis.set_ylabel("Winning weight-space share (%)")
    axis.set_title(title)
    axis.set_ylim(bottom=0)
    axis.grid(True, linewidth=0.5)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def format_threshold(
    result: Mapping[str, object] | None,
    model: str,
    event: str,
) -> str:
    if result is None:
        return f"{model}: {event} was not reached."

    if model == "DC":
        formulation = result["Leading outside DC formulation"]
        share = float(result["Leading outside DC share"])
    else:
        formulation = result["Leading outside DAC formulation"]
        share = float(result["Leading outside DAC share"])

    return (
        f"{model}: {event} at k={float(result['k']):.2f}; "
        f"Hybrid cost=${float(result['Hybrid raw cost']):.4f}; "
        f"Ag Two-Step cost=${float(result['Ag Two-Step raw cost']):.4f}; "
        f"leading outside formulation={formulation}; "
        f"outside formulation share={share:.3f}%."
    )


def write_summary(
    results: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    baseline = results[0]

    first_dc_pareto = first_result_where(
        results,
        lambda result: bool(
            pareto_outsiders(
                result["DC Pareto"]  # type: ignore[arg-type]
            )
        ),
    )

    first_dac_pareto = first_result_where(
        results,
        lambda result: bool(
            pareto_outsiders(
                result["DAC Pareto"]  # type: ignore[arg-type]
            )
        ),
    )

    first_dc_nonzero = first_result_where(
        results,
        lambda result: float(
            result["Outside-pair DC share"]
        ) > 1e-12,
    )

    first_dac_nonzero = first_result_where(
        results,
        lambda result: float(
            result["Outside-pair DAC share"]
        ) > 1e-12,
    )

    first_dc_material = first_result_where(
        results,
        lambda result: float(
            result["Outside-pair DC share"]
        ) >= MATERIAL_WINNING_SHARE_THRESHOLD,
    )

    first_dac_material = first_result_where(
        results,
        lambda result: float(
            result["Outside-pair DAC share"]
        ) >= MATERIAL_WINNING_SHARE_THRESHOLD,
    )

    k20 = min(
        results,
        key=lambda result: abs(float(result["k"]) - 20.0),
    )

    lines = [
        "Joint frontier-cost stress test",
        "",
        (
            f"Tested common multiplier range: "
            f"{K_MIN:.2f} to {K_MAX:.2f} "
            f"in increments of {K_STEP:.2f}."
        ),
        "",
        "Baseline (k=1.00):",
        (
            "  DC: Hybrid Padding-Squeezing "
            f"{baseline['DC shares'][HYBRID_NAME]:.3f}%, "
            "Ag Two-Step "
            f"{baseline['DC shares'][AG_NAME]:.3f}%."
        ),
        (
            "  DAC: Hybrid Padding-Squeezing "
            f"{baseline['DAC shares'][HYBRID_NAME]:.3f}%, "
            "Ag Two-Step "
            f"{baseline['DAC shares'][AG_NAME]:.3f}%."
        ),
        "",
        (
            "First outside formulation entering the DC Pareto front: "
            f"k={float(first_dc_pareto['k']):.2f}; "
            f"{', '.join(pareto_outsiders(first_dc_pareto['DC Pareto']))}."
        ),
        (
            "First outside formulation entering the DAC Pareto front: "
            f"k={float(first_dac_pareto['k']):.2f}; "
            f"{', '.join(pareto_outsiders(first_dac_pareto['DAC Pareto']))}."
        ),
        "",
        format_threshold(
            first_dc_nonzero,
            "DC",
            "first non-zero outside-pair winning region",
        ),
        format_threshold(
            first_dac_nonzero,
            "DAC",
            "first non-zero outside-pair winning region",
        ),
        "",
        format_threshold(
            first_dc_material,
            "DC",
            (
                f"outside-pair share first reached "
                f"{MATERIAL_WINNING_SHARE_THRESHOLD:.0f}%"
            ),
        ),
        format_threshold(
            first_dac_material,
            "DAC",
            (
                f"outside-pair share first reached "
                f"{MATERIAL_WINNING_SHARE_THRESHOLD:.0f}%"
            ),
        ),
        "",
        "At k=20.00:",
        (
            "  Assumed costs: Hybrid Padding-Squeezing "
            f"${float(k20['Hybrid raw cost']):.2f}; "
            "Ag Two-Step "
            f"${float(k20['Ag Two-Step raw cost']):.2f} "
            "per g fabric."
        ),
        (
            "  DC shares: Hybrid Padding-Squeezing "
            f"{k20['DC shares'][HYBRID_NAME]:.3f}%, "
            "Ag Two-Step "
            f"{k20['DC shares'][AG_NAME]:.3f}%, "
            "Hybrid In-Situ "
            f"{k20['DC shares']['Hybrid In-Situ']:.3f}%."
        ),
        (
            "  DAC shares: Hybrid Padding-Squeezing "
            f"{k20['DAC shares'][HYBRID_NAME]:.3f}%, "
            "Ag Two-Step "
            f"{k20['DAC shares'][AG_NAME]:.3f}%, "
            "Hybrid In-Situ "
            f"{k20['DAC shares']['Hybrid In-Situ']:.3f}%."
        ),
    ]

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main() -> None:
    dc_weights = generate_dc_weights()
    dac_weights = generate_dac_weights()

    results = [
        analyse_multiplier(
            common_multiplier=float(multiplier),
            dc_weights=dc_weights,
            dac_weights=dac_weights,
        )
        for multiplier in build_multiplier_values()
    ]

    csv_path = (
        OUTPUT_DIR / "joint_frontier_cost_stress_test.csv"
    )
    summary_path = (
        OUTPUT_DIR / "joint_frontier_cost_summary.txt"
    )
    dc_figure_path = (
        OUTPUT_DIR / "joint_frontier_cost_dc.png"
    )
    dac_figure_path = (
        OUTPUT_DIR / "joint_frontier_cost_dac.png"
    )

    write_results_csv(results, csv_path)
    write_summary(results, summary_path)

    plot_selected_shares(
        results=results,
        model_key="DC shares",
        formulation_names=[
            HYBRID_NAME,
            AG_NAME,
            "Hybrid In-Situ",
            "TiO2 Alkaline Hydrolysis",
            "Ag One-Step",
        ],
        title=(
            "Durability-Cost winning shares under joint "
            "frontier-cost inflation"
        ),
        output_path=dc_figure_path,
    )

    plot_selected_shares(
        results=results,
        model_key="DAC shares",
        formulation_names=[
            HYBRID_NAME,
            AG_NAME,
            "Hybrid In-Situ",
            "Ag One-Step",
            "ZnO Starch",
        ],
        title=(
            "Durability-Antibacterial-Cost winning shares under "
            "joint frontier-cost inflation"
        ),
        output_path=dac_figure_path,
    )

    print(summary_path.read_text(encoding="utf-8"))
    print()
    print(f"Results directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
