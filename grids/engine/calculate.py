from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from engine.data_cache import disk_cache
from web_interface.src.db_utils import load_db_data

PERCENTILES = [5, 10, 25, 50, 75, 90, 95]


@disk_cache()
def calculate_reference_percentiles(reference_data: pd.DataFrame):
    calculated_volumes = {}

    for column in reference_data.columns:
        calculated_volumes[column] = {
            percentile: np.percentile(reference_data[column], percentile)
            for percentile in PERCENTILES
        }

    return calculated_volumes


@disk_cache()
def bootstrap_percentiles(
    reference_data: pd.DataFrame,
    ref_percentiles: dict[int, float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> dict[int, dict[str, float]]:
    results = {}

    for structure in reference_data.columns:
        results[structure] = {}
        volume_data = reference_data[structure].values
        bootstrap_percentiles = np.zeros((n_bootstrap, len(PERCENTILES)))

        for i in range(n_bootstrap):
            # Generate bootstrap sample with replacement
            sample = np.random.choice(volume_data, size=len(volume_data), replace=True)

            # Calculate percentiles for this bootstrap sample
            for j, p in enumerate(PERCENTILES):
                bootstrap_percentiles[i, j] = np.percentile(sample, p)

        # For confidence 95%, alpha needs to be half of the 5% for left and right tail of the distribution
        alpha = (1 - confidence) / 2

        for j, p in enumerate(PERCENTILES):
            # Bootstrap confidence intervals
            lower_ci = np.percentile(
                bootstrap_percentiles[:, j], 100 * alpha
            )  # 0.025 tail
            upper_ci = np.percentile(
                bootstrap_percentiles[:, j], 100 * (1 - alpha)
            )  # 0.975 tail

            results[structure][p] = {
                "Volume": ref_percentiles[structure][p],
                "Lower CI": lower_ci,
                "Upper CI": upper_ci,
            }

    return results


def calculate_patient_percentiles(
    patient_data: pd.DataFrame, reference_percentiles: dict
):
    patient_percentiles = {}

    for structure in reference_percentiles.keys():
        patient_volume = patient_data[structure].iloc[0]
        ref_percentiles = reference_percentiles[structure]

        for p in sorted(ref_percentiles.keys()):
            if patient_volume <= ref_percentiles[p]:
                high_percentile = p
                break
            low_percentile = p

        if patient_volume <= reference_percentiles[structure][5]:
            patient_percentiles[structure] = "<5"
        elif patient_volume >= reference_percentiles[structure][90]:
            patient_percentiles[structure] = ">90"
        elif patient_volume == reference_percentiles[structure][high_percentile]:
            patient_percentiles[structure] = high_percentile
        else:
            low_volume = ref_percentiles[low_percentile]
            high_volume = ref_percentiles[high_percentile]

            interpolated = low_percentile + (high_percentile - low_percentile) * (
                patient_volume - low_volume
            ) / (high_volume - low_volume)

        patient_percentiles[structure] = round(interpolated, 1)

    return patient_percentiles


def create_bootstrap_table(
    bootstrap_results: dict[str, dict[int, dict[str, float]]],
    structure: str,
):
    comparison = pd.DataFrame(
        index=PERCENTILES,
        columns=[
            "Percentile Value",
            "Lower CI",
            "Upper CI",
            "CI Range (±)",
            "CI Range (%)",
            "Precision Rating",
        ],
    )

    for p in PERCENTILES:
        value = bootstrap_results[structure][p]["Volume"]
        lower = bootstrap_results[structure][p]["Lower CI"]
        upper = bootstrap_results[structure][p]["Upper CI"]

        # Calculate the half-width of the CI
        half_width = (upper - lower) / 2

        # Calculate the percentage error
        percent_error = (half_width / value) * 100

        # Assign a precision rating
        if percent_error < 0.3:
            precision = "Very High"
        elif percent_error < 0.5:
            precision = "High"
        elif percent_error < 1.0:
            precision = "Moderate"
        else:
            precision = "Low"

        comparison.loc[p, "Percentile Value"] = value
        comparison.loc[p, "Lower CI"] = lower
        comparison.loc[p, "Upper CI"] = upper
        comparison.loc[p, "CI Range (±)"] = half_width
        comparison.loc[p, "CI Range (%)"] = percent_error
        comparison.loc[p, "Precision Rating"] = precision

    # Format the comparison table for display
    for p in PERCENTILES:
        value = comparison.loc[p, "Percentile Value"]
        half_width = comparison.loc[p, "CI Range (±)"]
        percent = comparison.loc[p, "CI Range (%)"]
        precision = comparison.loc[p, "Precision Rating"]

    # Create a more human-readable table with ± notation
    readable_table = pd.DataFrame(
        index=PERCENTILES,
        columns=[
            "Percentile Value",
            "Confidence Interval (95%)",
            "Precision (±)",
            "Precision Rating",
        ],
    )

    for p in PERCENTILES:
        value = comparison.loc[p, "Percentile Value"]
        half_width = comparison.loc[p, "CI Range (±)"]
        percent = comparison.loc[p, "CI Range (%)"]
        precision = comparison.loc[p, "Precision Rating"]

        readable_table.loc[p, "Percentile Value"] = f"{value:.2f}"
        readable_table.loc[p, "Confidence Interval (95%)"] = (
            f"{value:.2f} ± {half_width:.2f}"
        )
        readable_table.loc[p, "Precision (±)"] = f"±{percent:.2f}%"
        readable_table.loc[p, "Precision Rating"] = precision

    return readable_table


def generate_ref_percentiles_plot(bootstrap_table: pd.DataFrame):
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
    ax.grid(True, alpha=0.3)
    ax.set_xticks(PERCENTILES)
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


def reference_bootstrap_percentiles(reference_data: pd.DataFrame):
    ref_percentiles = calculate_reference_percentiles(reference_data)
    bootstrap_results = bootstrap_percentiles(reference_data, ref_percentiles)

    return [
        create_bootstrap_table(bootstrap_results, structure)
        for structure in bootstrap_results.keys()
    ]


def analyze_patient(patient_data: pd.DataFrame, min_age: int, max_age: int):
    # Load reference data with given age range
    reference_data = load_db_data("PatientSummary", patient_data, min_age, max_age)

    # Calculate percentiles for reference data (return dict or class)
    reference_percentiles = calculate_reference_percentiles(reference_data.iloc[:, 6:])

    # Calculate patient's percentiles (return dict or class)
    patient_percentiles = calculate_patient_percentiles(
        patient_data, reference_percentiles
    )

    # Create visualization

    # Generate raport?
