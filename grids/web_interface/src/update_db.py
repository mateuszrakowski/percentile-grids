import sqlite3
from io import StringIO

import pandas as pd
from src.process_input import process_input
from streamlit.runtime.uploaded_file_manager import UploadedFile


def table_missing(name: str) -> bool:
    con = sqlite3.connect("grids/reference_dataset.db")
    cur = con.cursor()
    cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{name}'")
    return cur.fetchone() is None


def create_tables(
    cur: sqlite3.Cursor, records: pd.DataFrame, structures: pd.DataFrame
) -> None:
    record_columns = [f"{col} TEXT" for col in records.columns]
    structure_columns = record_columns + [
        f"{col} REAL" for col in structures.columns[4:]
    ]

    if table_missing("PatientRecords"):
        cur.execute(
            f"CREATE TABLE PatientRecords ({', '.join(record_columns)}, "
            f"UNIQUE(PatientID, StudyDate, StudyDescription) ON CONFLICT IGNORE)"
        )

    if table_missing("PatientStructures"):
        cur.execute(
            f"CREATE TABLE PatientStructures ({', '.join(structure_columns)}, "
            f"UNIQUE(PatientID, StudyDate, StudyDescription) ON CONFLICT IGNORE)"
        )


def convert_to_dataframe(input_files: list[UploadedFile]) -> list[pd.DataFrame]:
    return [pd.read_csv(StringIO(file.read().decode("utf-8"))) for file in input_files]


def update_db(input_files: list[UploadedFile]) -> None:
    dataframes = convert_to_dataframe(input_files)

    con = sqlite3.connect("grids/reference_dataset.db")
    cur = con.cursor()

    records, structures = process_input(dataframes[0])
    create_tables(cur, records, structures)

    records_data = []
    structures_data = []

    for input_csv in dataframes:
        records, structures = process_input(input_csv)
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
