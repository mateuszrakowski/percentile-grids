from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from web_interface.src.db_utils import load_db_data


def calculate_reference_percentiles(reference_data: pd.DataFrame):
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95]
    calculated_volumes = {}

    for column in reference_data.columns:
        calculated_volumes[column] = {
            percentile: np.percentile(reference_data[column], percentile)
            for percentile in percentiles
        }

    return calculated_volumes


def calculate_patient_percentiles(
    patient_data: pd.DataFrame, reference_percentiles: dict
):
    patient_percentiles = {}

    for structure in reference_percentiles.keys():
        patient_volume = patient_data[structure]
        ref_percentiles = reference_percentiles[structure]

        for p in sorted(ref_percentiles.keys()):
            if patient_volume <= ref_percentiles[p]:
                high_percentile = p
                break
            low_percentile = p

        if patient_volume <= reference_percentiles[10]:
            patient_percentiles[structure] = "<10"
        elif patient_volume >= reference_percentiles[90]:
            patient_percentiles[structure] = ">90"
        elif patient_volume == reference_percentiles[high_percentile]:
            patient_percentiles[structure] = high_percentile
        else:
            low_volume = ref_percentiles[low_percentile]
            high_volume = ref_percentiles[high_percentile]

            interpolated = low_percentile + (high_percentile - low_percentile) * (
                patient_volume - low_volume
            ) / (high_volume - low_volume)

        patient_percentiles[structure] = round(interpolated, 1)

    return patient_percentiles


def analyze_patient(patient_data: pd.DataFrame, min_age: int, max_age: int):
    # Load reference data with given age range
    reference_data = load_db_data("PatientSummary", patient_data, min_age, max_age)

    # Calculate percentiles for reference data (return dict or class)
    reference_percentiles = calculate_reference_percentiles(reference_data.iloc[:, 6:])

    # Calculate patient's percentiles (return dict or class)
    patient_percentiles = calculate_patient_percentiles(patient_data, reference_percentiles)

    # Create visualization

    # Generate raport?
