from time import sleep

import streamlit as st
from src.db_utils import load_db_data, update_db

st.title("Reference dataset")
st.divider()

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1

table_option = st.sidebar.selectbox(
    "Choose table to display:", ["PatientStructures", "PatientRecords"]
)

st.sidebar.divider()
st.sidebar.text("Display table options:")
age_attribute = st.sidebar.slider("Select age range: ", 0, 100, (0, 100))
st.sidebar.divider()

current_data = load_db_data(table_option, *age_attribute)

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
