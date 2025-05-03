import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from grids.engine.deprecated_calculate import PERCENTILES


def generate_ref_percentiles_plot(bootstrap_table: pd.DataFrame) -> plt.Figure:
    ci_range = bootstrap_table["Precision (±)"].apply(lambda x: float(x.strip("%±")))

    # Create visualization of precision across percentiles
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        PERCENTILES,
        ci_range,
        "o-",
        linewidth=2,
        markersize=10,
    )
    ax.set_title("Relative Precision (±%) Across Percentiles")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Relative Error (%)")
    ax.set_ylim(0, 2.5)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(PERCENTILES)
    ax.set_yticks(np.arange(0, 2.5, 0.25))
    ax.axhline(
        y=0.5, color="green", linestyle="--", alpha=0.7, label="High Precision (0.5%)"
    )
    ax.axhline(
        y=1.0,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label="Moderate Precision (1.0%)",
    )
    ax.axhline(
        y=2.0, color="red", linestyle="--", alpha=0.7, label="Low Precision (2.0%)"
    )
    ax.legend()
    fig.tight_layout()

    return fig


def parse_value(value: str) -> tuple[float, str]:
    """
    Parse values that might be strings like '>95' or '<5' or np.float64 values.
    Returns a float value for visualization and the original value for display.
    """
    # For visualization purposes:
    if value.startswith("<"):
        # "<5" becomes 2.5 (slightly below 5)
        num = float(value[1:])
        return num / 2, value

    elif value.startswith(">"):
        # ">95" becomes 97.5 (slightly above 95)
        num = float(value[1:])
        return num + (100 - num) / 2, value

    else:
        try:
            return float(value), value
        except ValueError:
            return np.nan, value


def create_custom_colormap():
    """
    Create a custom colormap that highlights outliers:
    - Low values (<5) in bright red
    - High values (>95) in bright green
    - Normal values in neutral blues
    """
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    colors = [
        (0.8, 0.0, 0.0),  # Red for low outliers
        (0.9, 0.2, 0),  # Orange for transition
        (0.2, 0.4, 1),  # Blue for middle - lower range
        (0.1, 0.3, 0.8),  # Deeper blue for true middle
        (0.2, 0.4, 1),  # Blue for middle - upper range
        (0.9, 0.2, 0),  # Orange for transition
        (0.8, 0.0, 0.0),  # Red for high outliers
    ]

    # Positions with expanded blue region in the middle
    positions = [0.0, 0.05, 0.4, 0.5, 0.6, 0.95, 1.0]
    return LinearSegmentedColormap.from_list(
        "outlier_cmap", list(zip(positions, colors))
    )


def create_data_heatmap(patient_percentiles: pd.DataFrame) -> plt.Figure:
    """
    Create a heatmap for data that contains a mix of float and string values.
    """
    # Parse the values for visualization
    values_for_vis = []
    original_values = []
    labels = []

    for row in patient_percentiles.itertuples():
        labels.append(row.Structure)
        vis_value, orig_value = parse_value(row.Percentile)
        values_for_vis.append(vis_value)
        original_values.append(orig_value)

    df = pd.DataFrame(
        {"Category": labels, "Value": values_for_vis, "Display": original_values}
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    # Transform the DataFrame to a heatmap-friendly format
    heatmap_data = pd.DataFrame(
        df["Value"].values, index=df["Category"], columns=["Value"]
    )

    cmap = create_custom_colormap()

    sns.heatmap(
        heatmap_data,
        annot=df["Display"].values.reshape(-1, 1),
        fmt="",
        cmap=cmap,
        ax=ax,
        cbar_kws={"label": "Percentile"},
        vmin=0,
        vmax=100,
        alpha=0.9
    )

    plt.title("Brain Structure Percentiles")
    plt.ylabel("")
    plt.tight_layout()

    return fig


def create_boxplot(patient_percentiles: pd.DataFrame) -> plt.Figure:
    sorted_structures = patient_percentiles.sort_values("Percentile")[
        "Structure"
    ].tolist()
    fig, ax = plt.subplots(figsize=(14, 8))

    for i, structure in enumerate(sorted_structures):
        # Create box representing the 25-75 percentile range
        box_start = 25
        box_width = 50  # 75 - 25
        ax.fill_between(
            [box_start, box_start + box_width],
            [i - 0.3] * 2,
            [i + 0.3] * 2,
            color="lightblue",
            alpha=0.5,
        )

        # Create whiskers for 5-25 and 75-95 percentiles
        whisker1_start = 5
        whisker1_width = 20  # 25 - 5
        ax.fill_between(
            [whisker1_start, whisker1_start + whisker1_width],
            [i - 0.1] * 2,
            [i + 0.1] * 2,
            color="lightblue",
            alpha=0.3,
        )

        whisker2_start = 75
        whisker2_width = 20  # 95 - 75
        ax.fill_between(
            [whisker2_start, whisker2_start + whisker2_width],
            [i - 0.1] * 2,
            [i + 0.1] * 2,
            color="lightblue",
            alpha=0.3,
        )

        # Create extreme range markers for 1-5 and 95-99 percentiles
        ax.plot([1, 5], [i, i], "b-", alpha=0.2)
        ax.plot([95, 99], [i, i], "b-", alpha=0.2)

        # Add median marker
        ax.plot([50, 50], [i - 0.3, i + 0.3], "b-")

        # Patient's percentile as a red dot
        perc_float, perc_string = parse_value(
            patient_percentiles[
                patient_percentiles["Structure"] == structure
            ].Percentile.iloc[0]
        )
        ax.plot(perc_float, i, "ro", markersize=8)

        # Add percentile value text
        ax.text(
            perc_float + 2,
            i,
            f"{perc_string}",
            va="center",
            fontsize=9,
            color="red",
        )

    ax.set_yticks(range(len(sorted_structures)))
    ax.set_yticklabels(sorted_structures)
    ax.set_xlabel("Percentile")
    ax.set_title("Brain Structure Percentiles")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()

    return fig
