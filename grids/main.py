import streamlit as st

st.set_page_config(layout="wide")


pg = st.navigation(
    [
        st.Page(
            page="web_interface/reference_dataset_page.py",
            title="Reference dataset",
        ),
        st.Page(
            page="web_interface/percentile_grids_page.py",
            title="Percentile grids",
        ),
    ]
)
pg.run()
