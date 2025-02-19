import os
import sqlite3

import pandas as pd
import sqlalchemy
from src.process_tables import convert_to_dataframes, process_csv_input
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
    records, structures = process_csv_input(sample_dataframe)

    record_columns = [
        f"{col} TEXT" if not "Age" in col else f"{col} INTEGER"
        for col in records.columns
    ]
    structure_columns = record_columns + [
        f"{col} REAL" for col in structures.columns[6:]
    ]

    if db_table_missing("PatientRecords"):
        cur.execute(
            f"CREATE TABLE PatientRecords ({', '.join(record_columns)}, "
            f"UNIQUE(PatientID, StudyDate, StudyDescription) ON CONFLICT IGNORE)"
        )

    if db_table_missing("PatientStructures"):
        cur.execute(
            f"CREATE TABLE PatientStructures ({', '.join(structure_columns)}, "
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

    records_data = []
    structures_data = []

    for input_csv in dataframes:
        records, structures = process_csv_input(input_csv)
        records_data.append(records.iloc[0].to_dict())
        structures_data.append(structures.iloc[0].to_dict())

    cur.executemany(
        f"INSERT INTO PatientRecords VALUES ({", ".join((f":{col}" for col in records.columns))})",
        records_data,
    )

    cur.executemany(
        f"INSERT INTO PatientStructures VALUES ({", ".join((f":{col}" for col in structures.columns))})",
        structures_data,
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
