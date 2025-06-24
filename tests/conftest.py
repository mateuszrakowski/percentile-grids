import os
import tempfile
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from streamlit.runtime.uploaded_file_manager import UploadedFile


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_xlsx_file_path():
    """Provides the path to the sample XLSX test file."""
    return os.path.join(os.path.dirname(__file__), "sample_input_test.xlsx")


@pytest.fixture
def sample_processed_data():
    """A sample of the final, processed single-row DataFrame."""
    return pd.DataFrame(
        {
            "PatientID": ["4341"],
            "AgeYears": [65],
            "AgeMonths": [789],
            "BirthDate": ["1959-03-28"],
            "StudyDate": ["2024-12-30"],
            "StudyDescription": ["RM glowy bez wzmocnienia kontrastowego"],
            "Kora_mózgu_lewa": [209.876],
            "Kora_mózgu_prawa": [206.363],
            "Istota_szara_móżdżku_lewa": [43.187],
            "Istota_szara_móżdżku_prawa": [0],
            "Wzgórze_lewe": [5.779],
            "Wzgórze_prawe": [5.801],
            "Jądro_ogoniaste_lewe": [3.575],
            "Jądro_ogoniaste_prawe": [3.63],
            "Gałka_blada_lewa": [1.703],
            "Gałka_blada_prawa": [45.238],
            "Ciało_migdałowate_lewe": [1.9],
            "Ciało_migdałowate_prawe": [2.045],
            "Jądro_półleżące_lewe": [0.642],
            "Jądro_półleżące_prawe": [0.604],
            "Skorupa_lewa": [5.449],
            "Skorupa_prawa": [5.271],
            "Hipokamp_lewy": [4.642],
            "Hipokamp_prawy": [4.496],
            "Istota_biała_lewa": [204.783],
            "Istota_biała_prawa": [204.36],
            "Pień_mózgu": [20.544],
            "Istota_biała_móżdżku_lewa": [12.794],
            "Istota_biała_móżdżku_prawa": [12.479],
            "Międzymózgowie_lewe": [3.498],
            "Międzymózgowie_prawe": [3.506],
            "Skrzyżowanie_wzrokowe": [0.521],
            "Układ_komorowy_mózgu_lewy": [13.651],
            "Układ_komorowy_mózgu_prawy": [12.419],
            "Komora_trzecia": [1.525],
            "Płyn_mózgowo_rdzeniowy": [406.46],
            "Komora_czwarta": [1.657],
        }
    )


@pytest.fixture
def mock_uploaded_file():
    """Mock uploaded file for testing."""
    mock_file = Mock(spec=UploadedFile)
    mock_file.name = "test_data.csv"
    return mock_file


@pytest.fixture
def test_db_path(temp_dir):
    """Path to a test database."""
    return os.path.join(temp_dir, "test_reference_dataset.db")


@pytest.fixture
def sample_model_data():
    """Sample data for model testing."""
    np.random.seed(42)
    ages = np.random.uniform(20, 80, 100)
    volumes = 100 + 0.5 * ages + np.random.normal(0, 10, 100)
    return pd.DataFrame({"Age": ages, "Volume": volumes})


@pytest.fixture
def cache_dir(temp_dir):
    """Temporary cache directory for testing."""
    cache_path = os.path.join(temp_dir, ".cache")
    os.makedirs(cache_path, exist_ok=True)
    return cache_path


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, temp_dir):
    """Setup test environment with temporary paths."""
    # Override default paths for testing
    monkeypatch.setenv("TESTING", "true")

    # Create necessary directories
    os.makedirs(os.path.join(temp_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "models"), exist_ok=True)
