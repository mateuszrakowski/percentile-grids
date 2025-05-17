from io import StringIO

import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile
from web_interface.db.data_structures import (
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


def process_csv_input(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    head = (
        df.head(5)
        .drop(columns=["Unnamed: 2"])
        .dropna()
        .set_index("Pacjent")
        .T.reset_index(drop=True)
        .iloc[:, [1, 0, 2, 3]]
    )

    body = df[7:].copy()
    body.columns = df.iloc[6].tolist()
    body = body.set_index("Struktura").T.iloc[1:].reset_index(drop=True)

    head.rename(
        columns={
            "Identyfikator pacjenta": "PatientID",
            "Data urodzenia": "BirthDate",
            "Data badania": "StudyDate",
            "Opis badania": "StudyDescription",
        },
        inplace=True,
    )

    study_date = pd.to_datetime(head["StudyDate"])
    birth_date = pd.to_datetime(head["BirthDate"])

    head["AgeYears"] = (
        study_date.dt.year[0]
        - birth_date.dt.year[0]
        - (
            (study_date.dt.month[0], study_date.dt.day[0])
            < (
                birth_date.dt.month[0],
                birth_date.dt.day[0],
            )
        )
    )

    head["AgeMonths"] = (study_date.dt.month[0] - birth_date.dt.month) % 12

    head.loc[study_date.dt.day < birth_date.dt.day, "AgeMonths"] = (
        head.loc[study_date.dt.day < birth_date.dt.day, "AgeMonths"] - 1
    ) % 12

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

    body.columns = [
        col.replace(" – ", "_").replace(" - ", "_").replace(" ", "_").replace("-", "_")
        for col in body.columns
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
    summary_table = structures_df.iloc[:, :6]

    for structure_class in structure_classes:
        structure_volumes = [
            float(getattr(structures_df, structure_v)[0])
            for structure_v in structure_class().model_dump().values()
        ]
        summary_table[structure_class.__name__] = round(sum(structure_volumes), 2)
    return summary_table


def convert_to_dataframes(input_files: list[UploadedFile]) -> list[pd.DataFrame]:
    return [pd.read_csv(StringIO(file.read().decode("utf-8"))) for file in input_files]


def load_checkbox_dataframe(
    current_df_state: pd.DataFrame | None, uploaded_files: list[UploadedFile]
) -> pd.DataFrame:
    dataframes = convert_to_dataframes(uploaded_files)
    standardized_dataframes = [process_csv_input(dataframe) for dataframe in dataframes]
    sum_structures_dataframes = [
        sum_structure_volumes(dataframe) for dataframe in standardized_dataframes
    ]

    joined_dataframes = pd.concat(
        [current_df_state, pd.concat(sum_structures_dataframes, axis=0)]
    ).reset_index(drop=True)

    df_with_checkboxes = joined_dataframes.copy()
    if "Select" not in df_with_checkboxes.columns:
        df_with_checkboxes.insert(0, "Select", False)
    df_with_checkboxes["Select"] = False

    return df_with_checkboxes
