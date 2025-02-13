import pandas as pd
import sqlalchemy
import streamlit as st
from src.update_db import table_missing, update_db


def main():
    st.title("Welcome to the Percentile Grids Web Interface")

    if table_missing("PatientRecords"):
        st.warning(
            "The database is empty. Please upload a CSV files to populate the database."
        )
    else:
        engine = sqlalchemy.create_engine("sqlite:///grids/reference_dataset.db")
        patient_structures = pd.read_sql("SELECT * FROM PatientStructures", engine)
        st.write(
            "Reference patient dataset:",
            patient_structures,
            "Number of patients: ",
            len(patient_structures),
        )

    uploaded_files = st.file_uploader(
        "Choose a CSV files", accept_multiple_files=True, type="csv, xlsx, xls"
    )
    if st.button("Send data to database."):
        if not uploaded_files:
            st.warning("Please select CSV files first.")
            return
        update_db(uploaded_files)


main()
