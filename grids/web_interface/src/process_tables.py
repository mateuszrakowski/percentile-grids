from io import StringIO

import pandas as pd
from src.structures_data import (
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

    today = pd.Timestamp.today()

    head["AgeYears"] = (
        today.year
        - pd.to_datetime(head["BirthDate"]).dt.year
        - (
            (today.month, today.day)
            < (
                pd.to_datetime(head["BirthDate"]).dt.month[0],
                pd.to_datetime(head["BirthDate"]).dt.day[0],
            )
        )
    )

    head["AgeMonths"] = (today.month - pd.to_datetime(head["BirthDate"]).dt.month) % 12

    birth_days = pd.to_datetime(head["BirthDate"]).dt.day
    head.loc[today.day < birth_days, "AgeMonths"] = (
        head.loc[today.day < birth_days, "AgeMonths"] - 1
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
        f"{col.replace(" – ", "_").replace(" - ", "_").replace(" ", "_").replace("-", '_')}"
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


def load_dataframe(
    current_df_state: pd.DataFrame | None, uploaded_files: list[UploadedFile]
) -> pd.DataFrame:
    dataframes = convert_to_dataframes(uploaded_files)
    dataframes = [process_csv_input(dataframe)[1] for dataframe in dataframes]

    dataframes = pd.concat(
        [current_df_state, pd.concat(dataframes, axis=0)]
    ).reset_index(drop=True)

    df_with_checkboxes = dataframes.copy()
    if "Select" not in df_with_checkboxes.columns:
        df_with_checkboxes.insert(0, "Select", False)
    df_with_checkboxes["Select"] = False

    return df_with_checkboxes
