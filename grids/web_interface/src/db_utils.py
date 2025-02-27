import os
import sqlite3

import pandas as pd
import sqlalchemy
from web_interface.src.process_tables import (
    convert_to_dataframes,
    process_csv_input,
    sum_structure_volumes,
)
from streamlit.runtime.uploaded_file_manager import UploadedFile


def init_database(name: str = "grids/reference_dataset.db") -> bool:
    if not os.path.exists(name):
        conn = sqlite3.connect(name)
        conn.close()


def db_table_missing(name: str) -> bool:
    init_database()

    con = sqlite3.connect("grids/reference_dataset.db")
    cur = con.cursor()

    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = :table_name",
        {"table_name": name},
    )
    return cur.fetchone() is None


def create_db_tables(cur: sqlite3.Cursor, sample_dataframe: pd.DataFrame) -> None:
    processed_dataframe = process_csv_input(sample_dataframe)
    structures_summary = sum_structure_volumes(processed_dataframe)

    patient_metadata = [
        f"{col} TEXT" if not "Age" in col else f"{col} INTEGER"
        for col in processed_dataframe.columns[:6]
    ]

    structure_columns = patient_metadata + [
        f"{col} REAL" for col in processed_dataframe.columns[6:]
    ]
    summary_columns = patient_metadata + [
        f"{col} REAL" for col in structures_summary.columns[6:]
    ]

    if db_table_missing("PatientStructures"):
        cur.execute(
            f"CREATE TABLE PatientStructures ({', '.join(structure_columns)}, "
            f"UNIQUE(PatientID, StudyDate, StudyDescription) ON CONFLICT IGNORE)"
        )

    if db_table_missing("PatientSummary"):
        cur.execute(
            f"CREATE TABLE PatientSummary ({', '.join(summary_columns)}, "
            f"UNIQUE(PatientID, StudyDate, StudyDescription) ON CONFLICT IGNORE)"
        )


def load_db_data(
    table_name: str, min_value: int = 0, max_value: int = 100
) -> pd.DataFrame | None:
    if db_table_missing(table_name):
        return None

    engine = sqlalchemy.create_engine("sqlite:///grids/reference_dataset.db")
    return pd.read_sql(
        f"SELECT * FROM {table_name} WHERE AgeYears BETWEEN :min_value AND :max_value",
        engine,
        params={
            "min_value": min_value,
            "max_value": max_value,
        },
    )


def update_db(input_files: list[UploadedFile]) -> None:
    dataframes = convert_to_dataframes(input_files)

    con = sqlite3.connect("grids/reference_dataset.db")
    cur = con.cursor()

    create_db_tables(cur, dataframes[0])

    structures_data = []
    summary_data = []

    for input_csv in dataframes:
        processed_dataframe = process_csv_input(input_csv)
        structures_summary = sum_structure_volumes(processed_dataframe)
        structures_data.append(processed_dataframe.iloc[0].to_dict())
        summary_data.append(structures_summary.iloc[0].to_dict())

    cur.executemany(
        f"INSERT INTO PatientStructures VALUES ({", ".join((f":{col}" for col in processed_dataframe.columns))})",
        structures_data,
    )

    cur.executemany(
        f"INSERT INTO PatientSummary VALUES ({", ".join((f":{col}" for col in structures_summary.columns))})",
        summary_data,
    )

    con.commit()
    con.close()


def display_db_age(table_name: str, min_value: int, max_value: int) -> None:
    engine = sqlalchemy.create_engine("sqlite:///grids/reference_dataset.db")
    return pd.read_sql(
        f"SELECT * FROM :table WHERE AgeYears BETWEEN :min_value AND :max_value",
        engine,
        params={"table": table_name, "min_value": min_value, "max_value": max_value},
    )
