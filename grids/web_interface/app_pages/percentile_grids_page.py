import time
from time import sleep

import engine.calculate as engine
import pandas as pd
import streamlit as st
from web_interface.src.db_utils import load_db_data
from web_interface.src.process_tables import load_dataframe


def update_slider():
    st.session_state["slider_percentile_age"] = st.session_state[
        "slider_percentile_age_key"
    ]


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

if "slider_percentile_age" not in st.session_state:
    st.session_state["slider_percentile_age"] = (-5, 5)


uploaded_files = st.sidebar.file_uploader(
    "Choose a CSV file for calculation:",
    accept_multiple_files=True,
    type=["csv", "xlsx", "xls"],
    key=st.session_state["uploader_key"],
)

st.sidebar.text("Percentile calculation settings:")
age_attribute = st.sidebar.slider(
    label="Patients age range:",
    min_value=-30,
    max_value=30,
    value=st.session_state["slider_percentile_age"],
    key="slider_percentile_age_key",
    on_change=update_slider,
    help=(
        "The range of patient's age, in years, for which the percentile grids will be calculated. "
        "The selected patient age is taken and calculated by adding/subtracting the specified value from both ends. "
        "For example, if the slider value is set to -5/5, the percentile grids will be calculated for patients 5 years older "
        "and 5 years younger than the selected patient (e.g., 30 years old patient will have percentile grids for 25 and 35 years old patients)."
    ),
)

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

    selected_indices = edited_df.index[edited_df["Select"]].tolist()
    if len(selected_indices) == 1:
        st.write("Selected rows:")
        selected_df = (
            st.session_state["patient_table"].iloc[:, 1:].iloc[selected_indices]
        )
        st.dataframe(selected_df)

        ref_dataset = load_db_data(
            "PatientSummary", selected_df, *age_attribute
        )
        st.sidebar.write(
            "Number of patients for current range:",
            len(ref_dataset),
        )

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

st.divider()

if not selected_df.empty:
    for structure in ref_dataset.iloc[:, 6:].columns:
        st.write(
            f"Mean for {structure}:",
            engine.calculate_mean(ref_dataset, age_attribute, structure),
        )
        st.write(
            f"Std for {structure}:",
            engine.calculate_std(ref_dataset, age_attribute, structure),
        )
        st.write(
            f"Percentiles for {structure}:",
            engine.calculate_percentiles(ref_dataset, age_attribute, structure),
        )

        # fig = engine.plot_age_histogram(ref_dataset, age_attribute, structure)
        # st.pyplot(fig)
        fig = engine.plot_percentile_histogram(
            engine.calculate_percentiles(ref_dataset, age_attribute, structure),
            age_attribute,
            structure,
        )
        st.pyplot(fig)
