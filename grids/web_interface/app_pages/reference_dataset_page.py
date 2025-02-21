from time import sleep

import streamlit as st
from src.db_utils import load_db_data, update_db


def update_slider():
    st.session_state["slider_reference_age"] = st.session_state[
        "slider_reference_age_key"
    ]


st.title("Reference dataset")
st.divider()

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1

if "slider_reference_age" not in st.session_state:
    st.session_state["slider_reference_age"] = (0, 100)


table_option = st.sidebar.selectbox(
    "Choose table to display:", ["PatientSummary", "PatientStructures"]
)

st.sidebar.divider()
st.sidebar.text("Display table options:")
age_attribute_range = st.sidebar.slider(
    "Select age range: ",
    min_value=0,
    max_value=100,
    value=st.session_state["slider_reference_age"],
    key="slider_reference_age_key",
    on_change=update_slider,
)
st.sidebar.divider()

current_data = load_db_data(table_option, *st.session_state["slider_reference_age"])

if current_data is None:
    st.warning(
        "The database is empty. Please upload a CSV file to populate the database."
    )
else:
    st.write("Number of patients:", len(current_data))
    st.dataframe(current_data)


uploaded_files = st.sidebar.file_uploader(
    "Choose a CSV file:",
    accept_multiple_files=True,
    type=["csv", "xlsx", "xls"],
    key=st.session_state["uploader_key"],
)

if st.sidebar.button("Send data to the database"):
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
