import os
import re
from datetime import datetime

import pandas as pd
from resources.brain_structures import (
    CerebralCerebellumCortex,
    CerebralCortex,
    CerebrospinalFluidTotal,
    NeuralStructuresTotal,
    SubcorticalGreyMatter,
    TotalStructuresVolume,
    VentricularSupratentorialSystem,
    WhiteMatterCerebral,
    WhiteMatterTotal,
)
from streamlit.runtime.uploaded_file_manager import UploadedFile


def _parse_input_file(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses the input DataFrame into header and body sections.

    Args:
        df: The input DataFrame from a CSV or Excel file.

    Returns:
        A tuple containing the header and body DataFrames.
    """
    head = (
        df.head(5)
        .drop(columns=["Unnamed: 2"])
        .dropna()
        .set_index("Pacjent")
        .T.reset_index(drop=True)
        .iloc[:, [1, 0, 2, 3]]
    )
    head.rename(
        columns={
            "Identyfikator pacjenta": "PatientID",
            "Data urodzenia": "BirthDate",
            "Data badania": "StudyDate",
            "Opis badania": "StudyDescription",
        },
        inplace=True,
    )

    body = df[7:].copy()
    body.columns = df.iloc[6].tolist()
    body = body.set_index("Struktura").T.iloc[1:].reset_index(drop=True)

    body.columns = [
        col.replace(" – ", "_").replace(" - ", "_").replace(" ", "_").replace("-", "_")
        for col in body.columns
    ]
    return head, body


def _calculate_age(birth_date: datetime, study_date: datetime) -> tuple[int, int]:
    """
    Calculates age in years and months from birth and study dates.

    Args:
        birth_date: Birth date as a datetime object.
        study_date: Study date as a datetime object.

    Returns:
        A tuple containing age in years and age in months.
    """
    age_years = (
        study_date.year
        - birth_date.year
        - ((study_date.month, study_date.day) < (birth_date.month, birth_date.day))
    )

    age_months = study_date.month - birth_date.month
    if study_date.day < birth_date.day:
        age_months -= 1
    age_months %= 12

    return age_years, age_months


def process_csv_input(df: pd.DataFrame) -> pd.DataFrame:
    head, body = _parse_input_file(df)

    birth_date = pd.to_datetime(head["BirthDate"].iloc[0])
    study_date = pd.to_datetime(head["StudyDate"].iloc[0])

    head["AgeYears"], head["AgeMonths"] = _calculate_age(birth_date, study_date)

    head = head[
        [
            "PatientID",
            "AgeYears",
            "AgeMonths",
            "BirthDate",
            "StudyDate",
            "StudyDescription",
        ]
    ]

    processed_dataframe = pd.concat([head, body], axis=1)
    return processed_dataframe


def sum_structure_volumes(structures_df: pd.DataFrame) -> pd.DataFrame:
    structure_classes = [
        CerebralCortex,
        CerebralCerebellumCortex,
        SubcorticalGreyMatter,
        WhiteMatterCerebral,
        WhiteMatterTotal,
        NeuralStructuresTotal,
        VentricularSupratentorialSystem,
        CerebrospinalFluidTotal,
        TotalStructuresVolume,
    ]
    summary_table = structures_df.iloc[:, :6].copy()

    for structure_class in structure_classes:
        volume_cols = list(structure_class().model_dump().values())
        summed_volumes = structures_df[volume_cols].astype(float).sum(axis=1)
        summary_table[structure_class.__name__] = summed_volumes.round(2)

    return summary_table


def convert_to_dataframes(input_files: list[UploadedFile]) -> list[pd.DataFrame]:
    readers = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
    }
    dataframes = []
    for file in input_files:
        _, extension = os.path.splitext(file.name)
        reader = readers.get(extension.lower())
        if reader:
            dataframes.append(reader(file))
        else:
            raise ValueError(f"Unsupported file format: {file.name}")
    return dataframes


def load_checkbox_dataframe(
    current_df_state: pd.DataFrame | None, uploaded_files: list[UploadedFile]
) -> pd.DataFrame:
    sum_structures_dataframes = [
        sum_structure_volumes(process_csv_input(dataframe))
        for dataframe in convert_to_dataframes(uploaded_files)
    ]

    dfs_to_concat = [
        df for df in [current_df_state] + sum_structures_dataframes if df is not None
    ]

    if not dfs_to_concat:
        return pd.DataFrame()

    joined_dataframes = pd.concat(dfs_to_concat, ignore_index=True).dropna()

    df_with_checkboxes = joined_dataframes.copy()
    if "Select" not in df_with_checkboxes.columns:
        df_with_checkboxes.insert(0, "Select", False)
    df_with_checkboxes["Select"] = False

    return df_with_checkboxes
