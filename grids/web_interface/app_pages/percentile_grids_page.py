from time import sleep

import pandas as pd
import streamlit as st
from src.process_tables import load_dataframe

st.title("Calculate percentile grids")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1

if "patient_table" not in st.session_state:
    st.session_state["patient_table"] = None

uploaded_files = st.sidebar.file_uploader(
    "Choose a CSV file for calculation:",
    accept_multiple_files=True,
    type=["csv", "xlsx", "xls"],
    key=st.session_state["uploader_key"],
)

if st.session_state["patient_table"] is not None:
    st.dataframe(st.session_state["patient_table"])

if st.sidebar.button("Send data"):
    if not uploaded_files:
        st.warning("Please select CSV files first.")
    else:
        st.session_state["patient_table"] = load_dataframe(
            st.session_state["patient_table"], uploaded_files
        )
        with st.spinner("Sending data to the database..."):
            sleep(2)
        st.success("Data successfully sent to the database!")
        sleep(1)

        st.session_state["uploader_key"] += 1
        st.rerun()
