
"""
Targeted optimistic-persistence checks for sparse ZnO evidence

This script evaluates two scenarios requested for the manuscript:

1. Durability-only optimistic scenario
   - ZnO-Starch and ZnO-SDS retain their final observed durability ratios
     unchanged to the 10-reference-wash horizon.
   - Antibacterial scores remain at the manuscript baseline.

2. Joint optimistic scenario
   - The optimistic durability assumption above is applied.
   - The final observed antibacterial value for each bacterial species is also
     carried forward unchanged to the 10-reference-wash horizon.

The script reports:
- Durability-Cost (DC) winning shares for all eight formulations:
  baseline vs durability-only optimistic.
- Durability-Antibacterial-Cost (DAC) winning shares:
  baseline, durability-only optimistic, and joint optimistic.
- DAC results are reported both with and without Ag formulations.

No figures are generated.

Data sources
------------
- Formulation-level baseline scores: manuscript Tables 2 and 4.
- Raw ZnO durability and antibacterial observations:
  "Original Data Extracted From Sources(3).xlsx"

Output
------
zno_durability_and_joint_optimistic_results.csv
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Output settings
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent / "zno_optimistic_sensitivity_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DC_WEIGHT_STEP = 0.01
DAC_WEIGHT_STEP = 0.01


# ---------------------------------------------------------------------------
# 2. Raw ZnO data from the uploaded Excel workbook
# ---------------------------------------------------------------------------

# Durability data are reported as:
# (unstandardized washing cycles, retained quantity)
ZNO_DURABILITY_RAW: Mapping[str, Sequence[Tuple[float, float]]] = {
    "ZnO Starch": [(0.0, 35.0), (5.0, 34.9), (10.0, 33.4)],
    "ZnO SDS": [(0.0, 6.74), (5.0, 5.91), (10.0, 4.32)],
}

# Antibacterial data are reported as:
# (unstandardized washing cycles, bacterial reduction %)
ZNO_ANTIBACTERIAL_RAW: Mapping[
    str, Mapping[str, Sequence[Tuple[float, float]]]
] = {
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
# 3. Baseline manuscript scores
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


# The manuscript's evidence-coverage factor for approximately 1.8
# reference-equivalent washes:
#
# (75 / 80) * (1.8 / 2) = 0.84375
#
# It is applied only in the baseline durability scores for the two ZnO routes.
ZNO_BASELINE_COVERAGE_FACTOR = (75.0 / 80.0) * (1.8 / 2.0)


def final_retention_ratio(
    observations: Sequence[Tuple[float, float]],
) -> float:
    """Return final observed retained quantity divided by the cycle-0 value."""
    ordered = sorted(observations)
    initial = ordered[0][1]
    final = ordered[-1][1]

    if initial <= 0:
        raise ValueError("Initial retained quantity must be positive.")

    return final / initial


def calculate_zno_durability_scores() -> Dict[str, Dict[str, float]]:
    """
    Calculate baseline and optimistic ZnO durability scores.

    Baseline:
        final retention ratio × manuscript evidence-coverage factor.

    Optimistic:
        final retention ratio carried forward unchanged to 10 washes.
    """
    results: Dict[str, Dict[str, float]] = {}

    for formulation, observations in ZNO_DURABILITY_RAW.items():
        retention_ratio = final_retention_ratio(observations)
        results[formulation] = {
            "retention_ratio": retention_ratio,
            "baseline": retention_ratio * ZNO_BASELINE_COVERAGE_FACTOR,
            "optimistic": retention_ratio,
        }

    return results


def calculate_zno_optimistic_antibacterial_scores() -> Dict[str, float]:
    """
    Calculate optimistic overall antibacterial scores.

    For each species, the final reported bacterial-reduction percentage is
    carried forward unchanged. The two species-specific values are then averaged.
    """
    results: Dict[str, float] = {}

    for formulation, species_data in ZNO_ANTIBACTERIAL_RAW.items():
        final_species_scores: List[float] = []

        for observations in species_data.values():
            final_value_percent = sorted(observations)[-1][1]
            final_species_scores.append(final_value_percent / 100.0)

        results[formulation] = float(np.mean(final_species_scores))

    return results


# Baseline antibacterial scores reported in manuscript Table 4.
ZNO_BASELINE_ANTIBACTERIAL = {
    "ZnO Starch": 0.395,
    "ZnO SDS": 0.802,
}


# ---------------------------------------------------------------------------
# 4. Construct MCDA datasets
# ---------------------------------------------------------------------------

def build_dc_formulations(
    zno_durability: Mapping[str, Mapping[str, float]],
    scenario: str,
) -> List[DCFormulation]:
    """Build the eight-formulation Durability-Cost dataset."""
    if scenario not in {"baseline", "durability_optimistic"}:
        raise ValueError("Invalid DC scenario.")

    zno_key = "baseline" if scenario == "baseline" else "optimistic"

    return [
        DCFormulation("TiO2 Two-Step Dipping",       0.459, 0.264),
        DCFormulation("TiO2 Alkaline Hydrolysis",    0.772, 0.327),
        DCFormulation("Hybrid Padding-Squeezing",    0.623, 1.000),
        DCFormulation("Hybrid In-Situ",              0.738, 0.496),
        DCFormulation(
            "ZnO Starch",
            zno_durability["ZnO Starch"][zno_key],
            0.000,
        ),
        DCFormulation(
            "ZnO SDS",
            zno_durability["ZnO SDS"][zno_key],
            0.377,
        ),
        DCFormulation("Ag Two-Step",                 0.917, 0.663),
        DCFormulation("Ag One-Step",                 0.911, 0.203),
    ]


def build_dac_formulations(
    zno_durability: Mapping[str, Mapping[str, float]],
    zno_optimistic_antibacterial: Mapping[str, float],
    scenario: str,
) -> List[DACFormulation]:
    """
    Build the six-formulation Durability-Antibacterial-Cost dataset.

    Supported scenarios:
    - baseline
    - durability_optimistic
    - joint_optimistic
    """
    if scenario not in {
        "baseline",
        "durability_optimistic",
        "joint_optimistic",
    }:
        raise ValueError("Invalid DAC scenario.")

    use_optimistic_durability = scenario in {
        "durability_optimistic",
        "joint_optimistic",
    }
    use_optimistic_antibacterial = scenario == "joint_optimistic"

    durability_key = "optimistic" if use_optimistic_durability else "baseline"

    starch_antibacterial = (
        zno_optimistic_antibacterial["ZnO Starch"]
        if use_optimistic_antibacterial
        else ZNO_BASELINE_ANTIBACTERIAL["ZnO Starch"]
    )
    sds_antibacterial = (
        zno_optimistic_antibacterial["ZnO SDS"]
        if use_optimistic_antibacterial
        else ZNO_BASELINE_ANTIBACTERIAL["ZnO SDS"]
    )

    return [
        DACFormulation("Hybrid Padding-Squeezing", 0.623, 0.999, 1.000),
        DACFormulation("Hybrid In-Situ",           0.738, 0.993, 0.496),
        DACFormulation(
            "ZnO Starch",
            zno_durability["ZnO Starch"][durability_key],
            starch_antibacterial,
            0.000,
        ),
        DACFormulation(
            "ZnO SDS",
            zno_durability["ZnO SDS"][durability_key],
            sds_antibacterial,
            0.377,
        ),
        DACFormulation("Ag Two-Step",              0.917, 0.997, 0.663),
        DACFormulation("Ag One-Step",              0.911, 0.929, 0.203),
    ]


# ---------------------------------------------------------------------------
# 5. Weight-space generation and winner-share calculations
# ---------------------------------------------------------------------------

def generate_dc_weights(step: float = DC_WEIGHT_STEP) -> np.ndarray:
    """Generate durability weights from 0 to 1 inclusive."""
    divisions = round(1.0 / step)

    if not math.isclose(divisions * step, 1.0, abs_tol=1e-12):
        raise ValueError("DC weight step must divide 1 exactly.")

    return np.arange(divisions + 1, dtype=float) / divisions


def generate_dac_weights(step: float = DAC_WEIGHT_STEP) -> np.ndarray:
    """
    Generate all feasible non-negative (w_D, w_A, w_C) combinations summing to 1.

    With step 0.01, this gives 5,151 combinations.
    """
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


def calculate_dc_winning_shares(
    formulations: Sequence[DCFormulation],
    durability_weights: np.ndarray,
) -> Dict[str, float]:
    """Calculate DC winning shares, splitting exact ties equally."""
    durability = np.asarray([f.durability for f in formulations], dtype=float)
    cost = np.asarray([f.cost for f in formulations], dtype=float)

    scores = (
        durability_weights[:, None] * durability[None, :]
        + (1.0 - durability_weights[:, None]) * cost[None, :]
    )

    maxima = scores.max(axis=1, keepdims=True)
    tied = np.isclose(scores, maxima, atol=1e-12, rtol=0.0)
    fractional_wins = tied / tied.sum(axis=1, keepdims=True)
    shares = fractional_wins.sum(axis=0) / len(durability_weights) * 100.0

    return {
        formulation.name: float(shares[index])
        for index, formulation in enumerate(formulations)
    }


def calculate_dac_winning_shares(
    formulations: Sequence[DACFormulation],
    weights: np.ndarray,
) -> Dict[str, float]:
    """Calculate DAC winning shares, splitting exact ties equally."""
    criteria = np.asarray(
        [
            [f.durability, f.antibacterial, f.cost]
            for f in formulations
        ],
        dtype=float,
    )

    scores = weights @ criteria.T
    maxima = scores.max(axis=1, keepdims=True)
    tied = np.isclose(scores, maxima, atol=1e-12, rtol=0.0)
    fractional_wins = tied / tied.sum(axis=1, keepdims=True)
    shares = fractional_wins.sum(axis=0) / len(weights) * 100.0

    return {
        formulation.name: float(shares[index])
        for index, formulation in enumerate(formulations)
    }


def remove_ag(
    formulations: Sequence[DACFormulation],
) -> List[DACFormulation]:
    """Return only the four non-Ag DAC formulations."""
    return [
        formulation
        for formulation in formulations
        if not formulation.name.startswith("Ag ")
    ]


# ---------------------------------------------------------------------------
# 6. Output
# ---------------------------------------------------------------------------

def print_share_table(
    heading: str,
    shares: Mapping[str, float],
) -> None:
    print(heading)

    for name, share in shares.items():
        print(f"  {name:<28} {share:7.3f}%")

    print()


def main() -> None:
    zno_durability = calculate_zno_durability_scores()
    zno_optimistic_antibacterial = (
        calculate_zno_optimistic_antibacterial_scores()
    )

    dc_weights = generate_dc_weights()
    dac_weights = generate_dac_weights()

    output_rows: List[Dict[str, object]] = []

    # -----------------------------------------------------------------------
    # A. Durability-Cost: baseline vs durability-only optimistic
    # -----------------------------------------------------------------------

    for scenario in ("baseline", "durability_optimistic"):
        formulations = build_dc_formulations(zno_durability, scenario)
        shares = calculate_dc_winning_shares(formulations, dc_weights)

        print_share_table(
            f"DC — {scenario.replace('_', ' ').title()}",
            shares,
        )

        for formulation, share in shares.items():
            output_rows.append(
                {
                    "Model": "DC",
                    "Dataset": "All eight formulations",
                    "Scenario": scenario,
                    "Formulation": formulation,
                    "Winning share (%)": f"{share:.6f}",
                }
            )

    # -----------------------------------------------------------------------
    # B. DAC: baseline, durability-only optimistic, joint optimistic
    # -----------------------------------------------------------------------

    for scenario in (
        "baseline",
        "durability_optimistic",
        "joint_optimistic",
    ):
        all_formulations = build_dac_formulations(
            zno_durability,
            zno_optimistic_antibacterial,
            scenario,
        )

        for dataset_name, formulations in (
            ("With Ag", all_formulations),
            ("Without Ag", remove_ag(all_formulations)),
        ):
            shares = calculate_dac_winning_shares(
                formulations,
                dac_weights,
            )

            print_share_table(
                f"DAC — {scenario.replace('_', ' ').title()} — {dataset_name}",
                shares,
            )

            for formulation, share in shares.items():
                output_rows.append(
                    {
                        "Model": "DAC",
                        "Dataset": dataset_name,
                        "Scenario": scenario,
                        "Formulation": formulation,
                        "Winning share (%)": f"{share:.6f}",
                    }
                )

    # -----------------------------------------------------------------------
    # C. Print the ZnO score changes used in the scenarios
    # -----------------------------------------------------------------------

    print("ZnO scores used")
    for formulation in ("ZnO Starch", "ZnO SDS"):
        print(
            f"  {formulation}: "
            f"durability {zno_durability[formulation]['baseline']:.6f} "
            f"-> {zno_durability[formulation]['optimistic']:.6f}; "
            f"antibacterial "
            f"{ZNO_BASELINE_ANTIBACTERIAL[formulation]:.6f} "
            f"-> {zno_optimistic_antibacterial[formulation]:.6f}"
        )
    print()

    # -----------------------------------------------------------------------
    # D. Save one consolidated CSV
    # -----------------------------------------------------------------------

    output_path = OUTPUT_DIR / (
        "zno_durability_and_joint_optimistic_results.csv"
    )

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Model",
                "Dataset",
                "Scenario",
                "Formulation",
                "Winning share (%)",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved results to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
