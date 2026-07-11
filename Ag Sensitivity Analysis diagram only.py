# Ag sensitivity analysis: generate two figures only
# - Same formulation = same color in both figures
# - No figure heading/title
# - Saves only two PNG files

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. USER SETTINGS
# ============================================================

MAX_REDUCTION = 0.30
REDUCTION_STEP = 0.01

DC_WEIGHT_STEP = 0.001
DAC_WEIGHT_STEP = 0.01

OUTPUT_FOLDER = Path.cwd() / "ag_sensitivity_figures"


# ============================================================
# 2. BASELINE NORMALIZED SCORES
# ============================================================

formulations = [
    "TiO2 - Two-step Dipping",
    "TiO2 - Alkaline Hydrolysis",
    "Hybrid - Padding-Squeezing",
    "Hybrid - In Situ",
    "ZnO - Ultrasound with starch",
    "ZnO - Ultrasound with SDS",
    "Ag - Two-step",
    "Ag - One-step",
]

scores = pd.DataFrame(
    {
        "Formulation": formulations,
        "Durability": [
            0.459,
            0.772,
            0.623,
            0.738,
            0.805,
            0.540,
            0.917,
            0.911,
        ],
        "Antibacterial": [
            np.nan,
            np.nan,
            0.999,
            0.996,
            0.395,
            0.793,
            0.997,
            0.929,
        ],
        "Cost": [
            0.264,
            0.327,
            1.000,
            0.496,
            0.000,
            0.377,
            0.663,
            0.203,
        ],
    }
)


# ============================================================
# 3. FIXED COLOR MAP
#    The same formulation keeps the same color in both figures.
# ============================================================

COLOR_MAP = {
    "TiO2 - Two-step Dipping": "#8C564B",
    "TiO2 - Alkaline Hydrolysis": "#1F77B4",
    "Hybrid - Padding-Squeezing": "#FF7F0E",
    "Hybrid - In Situ": "#2CA02C",
    "ZnO - Ultrasound with starch": "#D62728",
    "ZnO - Ultrasound with SDS": "#17BECF",
    "Ag - Two-step": "#9467BD",
    "Ag - One-step": "#E377C2",
}


# ============================================================
# 4. FUNCTIONS
# ============================================================

def reduce_ag_scores(data: pd.DataFrame, reduction: float) -> pd.DataFrame:
    """
    Reduce both durability and antibacterial scores of both Ag formulations.
    Cost scores and all non-Ag scores remain unchanged.
    """
    adjusted = data.copy()
    ag_mask = adjusted["Formulation"].str.startswith("Ag -")

    adjusted.loc[ag_mask, "Durability"] = np.clip(
        adjusted.loc[ag_mask, "Durability"] - reduction,
        0,
        1,
    )

    adjusted.loc[ag_mask, "Antibacterial"] = np.clip(
        adjusted.loc[ag_mask, "Antibacterial"] - reduction,
        0,
        1,
    )

    return adjusted


def dc_winner_share(data: pd.DataFrame, step: float) -> dict:
    """
    Test all Durability-Cost weightings:

        Index = w_D * Durability + (1 - w_D) * Cost

    Return the proportion of tested weightings for which each formulation
    ranks first.
    """
    weights = np.arange(0, 1 + step / 2, step)

    names = data["Formulation"].tolist()
    durability = data["Durability"].to_numpy(dtype=float)
    cost = data["Cost"].to_numpy(dtype=float)

    counts = {name: 0.0 for name in names}

    for w_d in weights:
        totals = w_d * durability + (1 - w_d) * cost
        best = totals.max()

        winners = np.where(np.isclose(totals, best, atol=1e-12))[0]

        for winner in winners:
            counts[names[winner]] += 1 / len(winners)

    return {
        name: count / len(weights)
        for name, count in counts.items()
    }


def generate_dac_weights(step: float):
    """
    Generate all non-negative D-A-C weights summing to 1.
    """
    units = round(1 / step)

    for d in range(units + 1):
        for a in range(units + 1 - d):
            c = units - d - a

            yield (
                d / units,
                a / units,
                c / units,
            )


def dac_winner_share(data: pd.DataFrame, step: float) -> dict:
    """
    Test all Durability-Antibacterial-Cost weight combinations:

        Index = w_D * Durability
              + w_A * Antibacterial
              + w_C * Cost

        where w_D + w_A + w_C = 1

    Return the proportion of tested weight combinations for which each
    formulation ranks first.
    """
    available = data.dropna(subset=["Antibacterial"]).copy()

    names = available["Formulation"].tolist()
    values = available[
        ["Durability", "Antibacterial", "Cost"]
    ].to_numpy(dtype=float)

    counts = {name: 0.0 for name in names}
    total_weight_combinations = 0

    for w_d, w_a, w_c in generate_dac_weights(step):
        weights = np.array([w_d, w_a, w_c])
        totals = values @ weights
        best = totals.max()

        winners = np.where(np.isclose(totals, best, atol=1e-12))[0]

        for winner in winners:
            counts[names[winner]] += 1 / len(winners)

        total_weight_combinations += 1

    return {
        name: count / total_weight_combinations
        for name, count in counts.items()
    }


# ============================================================
# 5. RUN ALL REDUCTION LEVELS
# ============================================================

reductions = np.round(
    np.arange(
        0,
        MAX_REDUCTION + REDUCTION_STEP / 2,
        REDUCTION_STEP,
    ),
    10,
)

dc_rows = []
dac_rows = []

for reduction in reductions:
    adjusted_scores = reduce_ag_scores(scores, reduction)

    dc_result = dc_winner_share(
        adjusted_scores,
        DC_WEIGHT_STEP,
    )
    dc_result["Ag_score_reduction"] = reduction
    dc_rows.append(dc_result)

    dac_result = dac_winner_share(
        adjusted_scores,
        DAC_WEIGHT_STEP,
    )
    dac_result["Ag_score_reduction"] = reduction
    dac_rows.append(dac_result)

dc_results = (
    pd.DataFrame(dc_rows)
    .set_index("Ag_score_reduction")
    .sort_index()
)

dac_results = (
    pd.DataFrame(dac_rows)
    .set_index("Ag_score_reduction")
    .sort_index()
)


# ============================================================
# 6. CREATE OUTPUT FOLDER
# ============================================================

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# 7. FIGURE 1: DURABILITY-COST
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

for formulation in formulations:
    if formulation not in dc_results.columns:
        continue

    if dc_results[formulation].max() <= 0:
        continue

    ax.plot(
        dc_results.index,
        dc_results[formulation],
        marker="o",
        markersize=3,
        linewidth=1.8,
        color=COLOR_MAP[formulation],
        label=formulation,
    )

ax.set_xlabel("Reduction in Ag durability score")
ax.set_ylabel("Proportion of tested D-C weightings ranked first")
ax.set_xlim(0, MAX_REDUCTION)
ax.set_ylim(0, 1)

# No ax.set_title(): this removes the heading.
ax.legend(fontsize=8)
fig.tight_layout()

fig.savefig(
    OUTPUT_FOLDER / "dc_winner_share_vs_ag_reduction.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 8. FIGURE 2: DURABILITY-ANTIBACTERIAL-COST
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

for formulation in formulations:
    if formulation not in dac_results.columns:
        continue

    if dac_results[formulation].max() <= 0:
        continue

    ax.plot(
        dac_results.index,
        dac_results[formulation],
        marker="o",
        markersize=3,
        linewidth=1.8,
        color=COLOR_MAP[formulation],
        label=formulation,
    )

ax.set_xlabel(
    "Simultaneous reduction in Ag durability and antibacterial scores"
)
ax.set_ylabel(
    "Proportion of tested D-A-C weight combinations ranked first"
)
ax.set_xlim(0, MAX_REDUCTION)
ax.set_ylim(0, 1)

# No ax.set_title(): this removes the heading.
ax.legend(fontsize=8)
fig.tight_layout()

fig.savefig(
    OUTPUT_FOLDER / "dac_winner_share_vs_ag_reduction.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)
