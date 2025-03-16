import time
from time import sleep

import streamlit as st
from engine.calculate import reference_bootstrap_percentiles
from engine.visualization import generate_ref_percentiles_plot
from web_interface.src.db_utils import load_db_data, update_db


def update_slider():
    st.session_state["slider_reference_age"] = st.session_state[
        "slider_reference_age_key"
    ]


if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1
if "slider_reference_age" not in st.session_state:
    st.session_state["slider_reference_age"] = (0, 100)
if "bootstrap_perc_tables" not in st.session_state:
    st.session_state.bootstrap_perc_tables = None
if "structure_names" not in st.session_state:
    st.session_state.structure_names = None
if "calculation_done" not in st.session_state:
    st.session_state.calculation_done = False

st.title("Reference dataset")
st.divider()

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


current_data = load_db_data(
    table_name=table_option,
    min_value=st.session_state["slider_reference_age"][0],
    max_value=st.session_state["slider_reference_age"][1],
)


if current_data is None:
    st.warning(
        "The database is empty. Please upload a CSV file to populate the database."
    )
else:
    st.dataframe(current_data)
    st.sidebar.write("Number of patients for current range:", len(current_data))

if table_option == "PatientSummary":
    calc_button = st.sidebar.button(
        "Calculate reference percentiles", key="calc_button"
    )
st.sidebar.divider()

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


if table_option == "PatientSummary" and current_data is not None:
    st.divider()

    structure_names = list(current_data.columns[6:])
    st.session_state.structure_names = structure_names

    if calc_button:
        with st.spinner("Calculating bootstrap percentiles..."):
            structure_data = current_data.iloc[:, 6:]
            bootstrap_perc_tables = reference_bootstrap_percentiles(structure_data)
            st.session_state.bootstrap_perc_tables = bootstrap_perc_tables

        if st.session_state.structure_names:
            structure_tabs = st.tabs(st.session_state.structure_names)

            # Display tables if they've been calculated
            if st.session_state.bootstrap_perc_tables is not None:
                for tab, table in zip(
                    structure_tabs, st.session_state.bootstrap_perc_tables
                ):
                    with tab:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.dataframe(table, use_container_width=True)

                        with col2:
                            st.pyplot(
                                generate_ref_percentiles_plot(table),
                                use_container_width=True,
                            )
