import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -----------------------------
# Data from your table
# D = Durability score
# A = Antibacterial overall score
# C = Cost score
# Higher score is better for all three.
# -----------------------------

formulations = [
    {
        "name": "Hybrid - Padding-squeezing",
        "material": "Hybrid",
        "D": 0.623,
        "A": 0.999,
        "C": 1.000,
    },
    {
        "name": "Hybrid - In situ",
        "material": "Hybrid",
        "D": 0.738,
        "A": 0.996,
        "C": 0.496,
    },
    {
        "name": "ZnO - Ultrasound with starch",
        "material": "ZnO",
        "D": 0.805,
        "A": 0.395,
        "C": 0.000,
    },
    {
        "name": "ZnO - Ultrasound with SDS",
        "material": "ZnO",
        "D": 0.540,
        "A": 0.793,
        "C": 0.377,
    },
    {
        "name": "Ag - Two-step grafting-then-reduction",
        "material": "Ag",
        "D": 0.917,
        "A": 0.997,
        "C": 0.663,
    },
    {
        "name": "Ag - One-step in-situ reduction-and-deposition",
        "material": "Ag",
        "D": 0.911,
        "A": 0.929,
        "C": 0.203,
    },
]


def make_winner_map(selected_formulations, title, filename):
    # Grid resolution. Increase to 401 or 501 if you want smoother boundaries.
    n = 301

    # x-axis: durability priority
    # y-axis: antibacterial priority
    w_D = np.linspace(0, 1, n)
    w_A = np.linspace(0, 1, n)

    WD, WA = np.meshgrid(w_D, w_A)
    WC = 1 - WD - WA  # cost priority

    # Valid region: all weights must be >= 0
    valid = WC >= 0

    # Calculate score for each formulation at every priority combination
    score_list = []
    for f in selected_formulations:
        score = WD * f["D"] + WA * f["A"] + WC * f["C"]
        score_list.append(score)

    scores = np.stack(score_list, axis=0)

    # Winner = formulation with highest score at each point
    winner = np.argmax(scores, axis=0).astype(float)

    # Hide invalid region where w_D + w_A > 1
    winner[~valid] = np.nan

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))

    cmap = plt.get_cmap("tab10", len(selected_formulations))
    cmap.set_bad("white")

    img = ax.imshow(
        winner,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=cmap,
        vmin=-0.5,
        vmax=len(selected_formulations) - 0.5,
        interpolation="nearest",
        aspect="equal",
    )

    # Draw boundary of valid triangle
    ax.plot([0, 1], [1, 0], linewidth=1.5)
    ax.plot([0, 0], [0, 1], linewidth=1.5)
    ax.plot([0, 1], [0, 0], linewidth=1.5)

    # Labels
    ax.set_xlabel("Durability priority, $w_D$")
    ax.set_ylabel("Antibacterial priority, $w_A$")
    ax.set_title(title)

    # Corner annotations
    ax.text(0.02, 0.02, "Cost only\n$w_C=1$", ha="left", va="bottom")
    ax.text(0.98, 0.02, "Durability only\n$w_D=1$", ha="right", va="bottom")
    ax.text(0.02, 0.98, "Antibacterial only\n$w_A=1$", ha="left", va="top")

    # Legend
    legend_handles = []
    for i, f in enumerate(selected_formulations):
        patch = mpatches.Patch(color=cmap(i), label=f["name"])
        legend_handles.append(patch)

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=True,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


# -----------------------------
# Figure 1: Include Ag formulations
# -----------------------------
make_winner_map(
    selected_formulations=formulations,
    title="Cost-Durability-Antibacterial Index: Including Ag Formulations",
    filename="winner_map_including_ag.png",
)

# -----------------------------
# Figure 2: Exclude Ag formulations
# -----------------------------
formulations_without_ag = [f for f in formulations if f["material"] != "Ag"]

make_winner_map(
    selected_formulations=formulations_without_ag,
    title="Cost-Durability-Antibacterial Index: Excluding Ag Formulations",
    filename="winner_map_excluding_ag.png",
)
