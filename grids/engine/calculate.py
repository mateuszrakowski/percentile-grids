import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_mean(
    data: pd.DataFrame, age_range: tuple[int, int], structure: str
) -> float:
    age_mask = (data["AgeYears"] >= age_range[0]) & (data["AgeYears"] <= age_range[1])
    return np.mean(data.loc[age_mask, structure])


def calculate_std(
    data: pd.DataFrame, age_range: tuple[int, int], structure: str
) -> float:
    age_mask = (data["AgeYears"] >= age_range[0]) & (data["AgeYears"] <= age_range[1])
    return np.std(data.loc[age_mask, structure])


def calculate_percentiles(
    data: pd.DataFrame, age_range: tuple[int, int], structure: str
) -> tuple[float, float, float]:
    age_mask = (data["AgeYears"] >= age_range[0]) & (data["AgeYears"] <= age_range[1])
    return np.percentile(data.loc[age_mask, structure], [3, 10, 25, 50, 75, 90, 97])


def plot_age_histogram(
    data: pd.DataFrame, age_range: tuple[int, int], structure: str
) -> None:
    age_mask = (data["AgeYears"] >= age_range[0]) & (data["AgeYears"] <= age_range[1])
    volumes = data.loc[age_mask, structure]
    fig, ax = plt.subplots()
    ax.hist(volumes, bins=10)
    ax.set_title(f"Age Histogram for {structure}")
    ax.set_xlabel("Volume")
    ax.set_ylabel("Frequency")
    return fig


def plot_percentile_histogram(
    percentiles: tuple[float, float, float], age_range: tuple[int, int], structure: str
) -> None:
    fig, ax = plt.subplots()
    ages = age_range[0] + np.arange(len(percentiles))
    ax.plot(ages, percentiles)
    ax.set_title(f"Percentiles for {structure}")
    ax.set_ylabel("Volume")
    ax.set_xlabel("Age")
    ax.legend()
    return fig
