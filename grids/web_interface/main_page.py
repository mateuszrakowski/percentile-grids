from time import sleep

import streamlit as st
from src.db_utils import load_data, table_missing, update_db

st.title("Welcome to the Percentile Grids Web Interface")


if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1

current_data = load_data()

if current_data is None:
    st.warning(
        "The database is empty. Please upload a CSV file to populate the database."
    )
else:
    st.write("Reference patient dataset.")
    st.write("Number of patients:", len(current_data))
    st.dataframe(current_data)

uploaded_files = st.file_uploader(
    "Choose a CSV file",
    accept_multiple_files=True,
    type=["csv", "xlsx", "xls"],
    key=st.session_state["uploader_key"],
)

if st.button("Send data to database"):
    if not uploaded_files:
        st.warning("Please select CSV files first.")
    else:
        update_db(uploaded_files)
        with st.spinner("Sending data to the database..."):
            sleep(3)
        st.success("Data successfully sent to the database!")
        sleep(1)

        st.session_state["uploader_key"] += 1
        st.rerun()
