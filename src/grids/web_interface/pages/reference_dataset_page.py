import math
from time import sleep

import streamlit as st
from engine.data_cache import clear_model_cache
from gamlss.gamlss import GAMLSS
from web_interface.db.db_utils import load_db_data, update_db

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1
if "gamlss_reference_plots" not in st.session_state:
    st.session_state.gamlss_reference_plots = None
if "structure_names" not in st.session_state:
    st.session_state.structure_names = None

st.title("Reference dataset")
st.divider()

table_option = st.sidebar.selectbox(
    "Choose table to display:", ["PatientSummary", "PatientStructures"]
)

current_data = load_db_data(table_name=table_option)

if current_data is None:
    st.warning(
        "The database is empty. Please upload a CSV file to populate the database."
    )
else:
    st.dataframe(current_data)
    st.sidebar.write("Number of patients for current range:", len(current_data))

st.sidebar.divider()

if table_option == "PatientSummary":
    calc_button = st.sidebar.button(
        "Calculate reference percentiles", key="calc_button", icon="🧮"
    )

last_run = GAMLSS.load_run_info()

if last_run:
    st.sidebar.write(
        f"Last model was calculated on: {last_run['dataset_length']} "
        f"patients at {last_run['timestamp']}."
    )

    if st.sidebar.button("Clear model", type="primary"):
        clear_model_cache()
        st.success("Models removed successfully!")
        st.session_state.gamlss_reference_plots = None
        sleep(1.5)
        st.rerun()

st.sidebar.divider()

uploaded_files = st.sidebar.file_uploader(
    "Choose a CSV file:",
    accept_multiple_files=True,
    type=["csv", "xlsx", "xls"],
    key=st.session_state["uploader_key"],
)

if st.sidebar.button("Send data to the database", icon="🚀"):
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


if table_option == "PatientSummary" and current_data is not None:
    st.divider()

    structure_names = list(current_data.columns[6:])
    st.session_state.structure_names = structure_names

    if calc_button:
        structure_data = current_data.iloc[:, 6:]
        progress_bar = st.progress(0)

        gamlss_reference_plots = []
        for i, col in enumerate(structure_data.columns):
            progress_bar.progress(
                i + 1, text=f"Fitting model {i+1} for structure {col}..."
            )

            gamlss = GAMLSS(current_data, "AgeYears", col)
            gamlss_reference_plots.append(gamlss.generate_grids())

        progress_bar.empty()
        st.session_state.gamlss_reference_plots = gamlss_reference_plots
        st.success("Reference percentiles successfully calculated!", icon="✅")

        sleep(1.5)
        st.rerun()

    if st.session_state.gamlss_reference_plots is not None:
        col1, col2 = st.columns(2)
        half_plots = math.ceil(len(st.session_state.structure_names) / 2)

        if st.session_state.gamlss_reference_plots is not None:
            col1_plots = st.session_state.gamlss_reference_plots[:half_plots]
            col2_plots = st.session_state.gamlss_reference_plots[half_plots:]

            with col1:
                for plot in col1_plots:
                    st.pyplot(plot, use_container_width=False)
            with col2:
                for plot in col2_plots:
                    st.pyplot(plot, use_container_width=False)
