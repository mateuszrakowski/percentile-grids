import math
from time import sleep

import pandas as pd
import streamlit as st
from data_processing.db_utils import load_db_data
from data_processing.process_input import load_checkbox_dataframe
from engine.model import GAMLSS


def handle_selection():
    # Get the edited rows information
    editor_data = st.session_state["editor_key"]

    if "edited_rows" in editor_data and editor_data["edited_rows"]:
        current_df = st.session_state["patient_table"].copy()
        current_df["Select"] = False
        edited_row_idx = list(editor_data["edited_rows"].keys())[0]
        current_df.at[int(edited_row_idx), "Select"] = True

        st.session_state["patient_table"] = current_df


st.title("Calculate percentile grids")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1
if "patient_table" not in st.session_state:
    st.session_state["patient_table"] = None
if "gamlss_patient_plots" not in st.session_state:
    st.session_state.gamlss_patient_plots = None
if "structure_names" not in st.session_state:
    st.session_state.structure_names = None
if "calculated_patient" not in st.session_state:
    st.session_state.calculated_patient = None

selected_df = pd.DataFrame()
if st.session_state["patient_table"] is not None:
    st.divider()
    edited_df = st.data_editor(
        st.session_state["patient_table"],
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "Select",
                help="Select row",
                default=False,
            )
        },
        disabled=st.session_state["patient_table"].columns[1:],
        hide_index=True,
        key="editor_key",
        on_change=handle_selection,
    )

    calc_button = st.sidebar.button(
        "Calculate patient percentiles", key="calc_button", icon="🧮"
    )

    selected_indices = edited_df.index[edited_df["Select"]].tolist()
    if len(selected_indices) == 1:
        st.write("Selected row:")
        selected_df = (
            st.session_state["patient_table"].iloc[:, 1:].iloc[selected_indices]
        )
        st.dataframe(selected_df)


ref_dataset = load_db_data("PatientSummary")

if ref_dataset is None:
    st.warning(
        "The reference database is empty! Please navigate to the "
        "'Reference dataset' page and upload a CSV files to start.",
        icon="⚠️",
    )
else:
    st.sidebar.write(
        "Number of patients in database:",
        len(ref_dataset),
    )

uploaded_files = st.sidebar.file_uploader(
    "Send patients for calculations:",
    accept_multiple_files=True,
    type=["csv", "xlsx", "xls"],
    key=st.session_state["uploader_key"],
)

if st.sidebar.button("Send data", icon="🚀"):
    if not uploaded_files:
        st.warning("Please select CSV files first.")
    else:
        st.session_state["patient_table"] = load_checkbox_dataframe(
            st.session_state["patient_table"], uploaded_files
        )
        with st.spinner("Sending data..."):
            sleep(2)
        st.success("Data sent successfully!")
        sleep(1)

        st.session_state["uploader_key"] += 1
        st.rerun()

st.divider()

if (
    st.session_state["patient_table"] is not None
    and calc_button
    and ref_dataset is not None
):
    if len(selected_indices) == 1:
        st.session_state.calculated_patient = selected_df.PatientID.iloc[0]
        st.session_state.structure_names = list(ref_dataset.columns[6:])

        progress_bar = st.progress(0)
        gamlss_patient_plots = []

        for i, col in enumerate(st.session_state.structure_names):
            progress_bar.progress(
                i + 1, text=f"Calculating percentiles for structure {col}..."
            )

            model_path = f"/app/data/models/gamlss_{col}.rds"
            model = GAMLSS.load_model(model_path, ref_dataset, "AgeYears", col)
            gamlss_patient_plots.append(
                model.generate_grids_oos(st.session_state["patient_table"])
            )

        progress_bar.empty()
        st.session_state.gamlss_patient_plots = gamlss_patient_plots
        st.success("Patient percentiles successfully calculated!", icon="✅")

        sleep(1.5)
        st.rerun()

    else:
        st.warning("Select a patient from the table to calculate its percentile grids.")

if st.session_state["patient_table"] is not None and selected_df.shape[0] == 1:
    if (
        st.session_state.gamlss_patient_plots is not None
        and st.session_state.calculated_patient == selected_df.PatientID.iloc[0]
    ):
        st.header(f"Patient {st.session_state.calculated_patient}")
        st.write(
            f"GAMLSS Model (BCPE) trained on samples from "
            f"{min(ref_dataset['AgeYears'])} - {max(ref_dataset['AgeYears'])} years."
        )

        col1, col2 = st.columns(2)
        half_plots = math.ceil(len(st.session_state.structure_names) / 2)

        col1_plots = st.session_state.gamlss_patient_plots[:half_plots]
        col2_plots = st.session_state.gamlss_patient_plots[half_plots:]

        with col1:
            for plot in col1_plots:
                st.pyplot(plot, use_container_width=False)
        with col2:
            for plot in col2_plots:
                st.pyplot(plot, use_container_width=False)
