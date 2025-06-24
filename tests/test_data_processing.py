import io
import os
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from streamlit.runtime.uploaded_file_manager import UploadedFile

from grids.data_processing.db_utils import get_db_connection, load_db_data, update_db
from grids.data_processing.process_input import (
    convert_to_dataframes,
    load_checkbox_dataframe,
    process_csv_input,
    sum_structure_volumes,
)


class TestProcessInput:
    """Test cases for data processing input functions."""

    def test_process_csv_input(self, sample_xlsx_file_path):
        """Test XLSX input processing using a real file."""
        df = pd.read_excel(sample_xlsx_file_path)
        result = process_csv_input(df)
        assert isinstance(result, pd.DataFrame)
        assert "PatientID" in result.columns
        assert result.shape[0] == 1
        assert result["PatientID"].iloc[0] == "4341"
        assert result["AgeYears"].iloc[0] == 65
        assert "Skorupa_lewa" in result.columns
        assert result["Skorupa_lewa"].iloc[0] == 5.449

    def test_sum_structure_volumes(self, sample_processed_data):
        """Test structure volume summation."""
        result = sum_structure_volumes(sample_processed_data)
        assert isinstance(result, pd.DataFrame)
        assert "CerebralCortex" in result.columns
        assert result["CerebralCortex"].iloc[0] > 0

    def test_convert_to_dataframes_xlsx(self, sample_xlsx_file_path):
        """Test converting XLSX files to dataframes using a file-like object."""
        with open(sample_xlsx_file_path, "rb") as f:
            file_bytes = f.read()

        # Create a file-like object that pandas can read
        buffer = io.BytesIO(file_bytes)
        buffer.name = "sample_input_test.xlsx"  # Add the name attribute

        # Pass the buffer directly to the function
        result = convert_to_dataframes([buffer])  # type: ignore

        assert len(result) == 1
        assert isinstance(result[0], pd.DataFrame)
        assert not result[0].empty

    def test_convert_to_dataframes_unsupported_format(self):
        """Test handling of unsupported file formats."""
        mock_file = Mock(spec=UploadedFile)
        mock_file.name = "test.txt"
        with pytest.raises(ValueError, match="Unsupported file format"):
            convert_to_dataframes([mock_file])

    def test_load_checkbox_dataframe(self, sample_xlsx_file_path):
        """Test loading dataframe with checkboxes from an XLSX file."""
        mock_file = Mock(spec=UploadedFile)
        mock_file.name = "sample_input_test.xlsx"

        # We need to provide a realistic dataframe to process
        real_df = pd.read_excel(sample_xlsx_file_path, header=None)

        # Since convert_to_dataframes is part of the call stack, we patch it
        # to return our pre-loaded, realistic DataFrame.
        with patch(
            "grids.data_processing.process_input.convert_to_dataframes"
        ) as mock_convert:
            mock_convert.return_value = [real_df]
            result = load_checkbox_dataframe(None, [mock_file])

            assert isinstance(result, pd.DataFrame)
            assert "Select" in result.columns
            assert result["Select"].dtype == bool
            assert not result["Select"].any()


class TestDatabaseUtils:
    """Test cases for database utility functions."""

    def test_get_db_connection(self, temp_dir):
        """Test database connection and creation."""
        db_path = os.path.join(temp_dir, "test.db")
        conn = get_db_connection(db_path)
        assert conn is not None
        assert os.path.exists(db_path)
        conn.close()

    def test_load_db_data_creates_tables(self, sample_xlsx_file_path, temp_dir):
        """Test that load_db_data creates tables if they don't exist."""
        db_path = os.path.join(temp_dir, "test.db")
        table_name = "PatientSummary"
        sample_df = pd.read_excel(sample_xlsx_file_path, header=None)

        result_before = load_db_data(table_name, db_path)
        assert result_before is None

        load_db_data(table_name, db_path, sample_dataframe=sample_df)

        result_after = load_db_data(table_name, db_path)
        assert isinstance(result_after, pd.DataFrame)
        assert result_after.empty

    def test_load_db_data_nonexistent_table(self, temp_dir):
        """Test loading data from a non-existent table without providing a schema."""
        db_path = os.path.join(temp_dir, "test.db")
        get_db_connection(db_path).close()
        result = load_db_data("NonExistentTable", db_path)
        assert result is None

    @patch("grids.data_processing.db_utils.convert_to_dataframes")
    def test_update_db(self, mock_convert, sample_xlsx_file_path, temp_dir):
        """Test updating the database with new data from an XLSX file."""
        db_path = os.path.join(temp_dir, "test.db")
        sample_df = pd.read_excel(sample_xlsx_file_path, header=None)
        mock_convert.return_value = [sample_df]

        mock_files = [Mock(spec=UploadedFile)]
        mock_files[0].name = "sample_input_test.xlsx"

        update_db(mock_files, db_path)  # type: ignore

        summary_data = load_db_data("PatientSummary", db_path)
        structures_data = load_db_data("PatientStructures", db_path)

        assert summary_data is not None
        assert not summary_data.empty
        assert "CerebralCortex" in summary_data.columns

        assert structures_data is not None
        assert not structures_data.empty
        assert "Skorupa_lewa" in structures_data.columns


class TestDataProcessingIntegration:
    """Integration tests for data processing workflow."""

    def test_full_data_processing_workflow(self, sample_xlsx_file_path):
        """Test the complete data processing workflow using a real XLSX file."""
        df = pd.read_excel(sample_xlsx_file_path, header=None)
        processed_data = process_csv_input(df)
        assert isinstance(processed_data, pd.DataFrame)

        summary_data = sum_structure_volumes(processed_data)
        assert isinstance(summary_data, pd.DataFrame)

        expected_summary_columns = [
            "CerebralCortex",
            "CerebralCerebellumCortex",
            "SubcorticalGreyMatter",
        ]
        for col in expected_summary_columns:
            assert col in summary_data.columns
            assert summary_data[col].iloc[0] > 0
