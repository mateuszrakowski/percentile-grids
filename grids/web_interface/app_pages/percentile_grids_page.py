from time import sleep

import streamlit as st
from src.process_tables import load_dataframe

st.title("Calculate percentile grids")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1

if "patient_table" not in st.session_state:
    st.session_state["patient_table"] = None

st.sidebar.text("Percentile calculation settings:")
age_attribute = st.sidebar.slider(
    "Select patients age range from reference dataset: ", 0, 100, (0, 100)
)
st.sidebar.divider()

uploaded_files = st.sidebar.file_uploader(
    "Choose a CSV file for calculation:",
    accept_multiple_files=True,
    type=["csv", "xlsx", "xls"],
    key=st.session_state["uploader_key"],
)


if st.session_state["patient_table"] is not None:
    df = st.session_state["patient_table"]
    st.divider()
    edited_df = st.data_editor(
        df,
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "Select",
                help="Select row",
                default=False,
            )
        },
        disabled=df.columns[1:],
        hide_index=True,
    )

    selected_indices = edited_df.index[edited_df["Select"]].tolist()
    if selected_indices:
        st.write("Selected rows:")
        st.dataframe(df.iloc[:, 1:].iloc[selected_indices])

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
