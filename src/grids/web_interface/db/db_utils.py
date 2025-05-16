import os
import sqlite3

import pandas as pd
import sqlalchemy
from streamlit.runtime.uploaded_file_manager import UploadedFile
from web_interface.db.process_tables import (
    convert_to_dataframes,
    process_csv_input,
    sum_structure_volumes,
)


def init_database(name: str = "src/grids/reference_dataset.db") -> bool:
    if not os.path.exists(name):
        conn = sqlite3.connect(name)
        conn.close()


def db_table_missing(name: str) -> bool:
    init_database()

    con = sqlite3.connect("src/grids/reference_dataset.db")
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
        f"{col} TEXT" if "Age" not in col else f"{col} INTEGER"
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


def load_db_data(table_name: str) -> pd.DataFrame | None:
    if db_table_missing(table_name):
        return None

    engine = sqlalchemy.create_engine("sqlite:///src/grids/reference_dataset.db")
    return pd.read_sql(
        f"SELECT * FROM {table_name}",
        engine,
    )


def update_db(input_files: list[UploadedFile]) -> None:
    dataframes = convert_to_dataframes(input_files)

    con = sqlite3.connect("src/grids/reference_dataset.db")
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
        (
            f"INSERT INTO PatientStructures VALUES "
            f"({", ".join((f":{col}" for col in processed_dataframe.columns))})"
        ),
        structures_data,
    )

    cur.executemany(
        (
            f"INSERT INTO PatientSummary VALUES "
            f"({", ".join((f":{col}" for col in structures_summary.columns))})"
        ),
        summary_data,
    )

    con.commit()
    con.close()
