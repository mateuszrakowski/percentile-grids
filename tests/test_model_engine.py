import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from matplotlib.figure import Figure

from grids.engine.model import GAMLSS, FittedGAMLSSModel


@pytest.fixture
def mock_r_env():
    """Provides a comprehensive mock for the R environment."""
    with patch("grids.engine.model.r_env", autospec=True) as mock_env:
        mock_env.gamlss_r.gamlss = MagicMock()
        mock_env.stats.AIC = MagicMock(return_value=[150.0])
        mock_env.stats.BIC = MagicMock(return_value=[160.0])
        mock_env.gamlss_r.predictAll = MagicMock()
        mock_env.gamlss_r.centiles_pred = MagicMock(return_value=[0.5])
        mock_env.gamlss_dist.qNO = MagicMock(return_value=[1.0] * 200)
        mock_env.base.saveRDS = MagicMock()
        mock_env.base.readRDS = MagicMock()
        yield mock_env


@pytest.fixture
def mock_r_model(mock_r_env):
    """Creates a mock R model object, simulating a fitted gamlss model."""
    mock_model = MagicMock()
    mock_model.rx2.side_effect = lambda key: {
        "converged": [True],
        "G.deviance": [100.5],
        "family": ["NO"],
        "parameters": ["mu", "sigma"],
    }.get(key)
    # When the model is created, it immediately calculates AIC/BIC.
    # We need to make sure the mock_r_env is used for that.
    with patch("grids.engine.model.r_env", mock_r_env):
        yield mock_model


@pytest.fixture
def source_data():
    """Provides a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "Age": range(20, 30),
            "Volume": [i + 50 for i in range(10)],
        }
    )


@pytest.fixture
def fitted_model(mock_r_model, source_data, mock_r_env):
    """Provides a pre-configured FittedGAMLSSModel instance for testing."""
    # The key is to patch the environment *before* the instance is created,
    # because __init__ immediately calls methods that use r_env.
    with patch("grids.engine.model.r_env", mock_r_env):
        model = FittedGAMLSSModel(
            r_model=mock_r_model,
            source_data=source_data,
            x_column="Age",
            y_column="Volume",
            percentiles=[0.05, 0.50, 0.95],
        )
        yield model


class TestFittedGAMLSSModel:
    """Unit tests for the FittedGAMLSSModel class."""

    def test_initialization(self, fitted_model, mock_r_model, mock_r_env):
        """Tests that the model initializes correctly and extracts key metrics."""
        assert fitted_model.model is mock_r_model
        assert fitted_model.converged is True
        assert fitted_model.deviance == 100.5
        assert fitted_model.aic == 150.0  # Check that it was set on init
        mock_r_env.stats.AIC.assert_called_with(mock_r_model)

    def test_save_model(self, fitted_model, tmp_path, mock_r_env):
        """Tests saving the model and its metadata."""
        model_path = tmp_path / "model.rds"
        run_info_path = tmp_path / "run_info.json"

        # No need to patch here if fitted_model fixture already does it
        fitted_model.save(str(model_path), str(run_info_path))

        mock_r_env.base.saveRDS.assert_called_once_with(
            fitted_model.model, file=str(model_path)
        )
        assert run_info_path.exists()
        with open(run_info_path) as f:
            run_info = json.load(f)
            assert run_info["dataset_length"] == 10
            assert run_info["aic"] == fitted_model.aic

    def test_calculate_percentiles(self, fitted_model, mock_r_env):
        """Tests that percentile curves are calculated by calling R."""
        mock_pred_params = MagicMock()
        mock_pred_params.names = ["mu", "sigma"]
        mock_pred_params.rx2.side_effect = lambda key: {"mu": [1], "sigma": [0.1]}.get(
            key
        )
        mock_r_env.gamlss_r.predictAll.return_value = mock_pred_params

        curves = fitted_model.calculate_percentiles()

        mock_r_env.gamlss_r.predictAll.assert_called()
        assert mock_r_env.gamlss_dist.qNO.call_count == len(fitted_model.percentiles)
        assert 0.50 in curves
        assert len(curves[0.50]) == 200  # Check the curve length

    def test_predict_patient_oos(self, fitted_model, mock_r_env):
        """Tests out-of-sample prediction for a single patient."""
        patient_data = pd.DataFrame([{"Age": 25, "Volume": 60}])
        z_score, percentile = fitted_model.predict_patient_oos(patient_data)

        assert z_score == 0.5
        assert percentile is not None
        mock_r_env.gamlss_r.centiles_pred.assert_called_once()

    def test_plot_percentiles(self, fitted_model):
        """Tests that plotting generates a matplotlib figure without error."""
        percentile_curves = {0.5: [1] * 200}
        fig = fitted_model.plot_percentiles(percentile_curves)
        assert isinstance(fig, Figure)

    def test_plot_oos_patient(self, fitted_model):
        """Tests that the out-of-sample plot generates successfully."""
        patient_data = pd.DataFrame([{"Age": 25, "Volume": 60}])
        percentile_curves = {0.5: [1] * 200}
        fig = fitted_model.plot_oos_patient(patient_data, percentile_curves, 0.5, 0.69)
        assert isinstance(fig, Figure)


class TestGAMLSS:
    """Unit tests for the GAMLSS factory class."""

    @pytest.fixture
    def gamlss_instance(self, source_data):
        """Provides a GAMLSS instance for testing."""
        return GAMLSS(data_table=source_data, x_column="Age", y_column="Volume")

    def test_fit(self, gamlss_instance, mock_r_env, mock_r_model):
        """Tests that the fit method calls the R gamlss function correctly."""
        mock_r_env.gamlss_r.gamlss.return_value = mock_r_model
        family = "NO"
        formula_mu = "Volume ~ Age"
        formula_sigma = "~ 1"

        with patch("grids.engine.model.r_env", mock_r_env):
            fitted_model = gamlss_instance.fit(family, formula_mu, formula_sigma)

        assert isinstance(fitted_model, FittedGAMLSSModel)
        assert fitted_model.model == mock_r_model
        mock_r_env.gamlss_r.gamlss.assert_called_once()

    def test_load_model(self, source_data, tmp_path, mock_r_env, mock_r_model):
        """Tests loading a pre-saved R model from a file."""
        model_path = tmp_path / "model.rds"
        model_path.touch()

        mock_r_env.base.readRDS.return_value = mock_r_model

        with patch("grids.engine.model.r_env", mock_r_env):
            loaded_model = GAMLSS.load_model(
                model_path=str(model_path),
                source_data=source_data,
                x_column="Age",
                y_column="Volume",
            )

        assert isinstance(loaded_model, FittedGAMLSSModel)
        assert loaded_model.model is mock_r_model
        mock_r_env.base.readRDS.assert_called_once_with(file=str(model_path))

    def test_load_model_not_found(self):
        """Tests that loading a non-existent model raises a FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            GAMLSS.load_model("non_existent_path.rds", pd.DataFrame(), "x", "y")
