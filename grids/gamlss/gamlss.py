import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from gamlss.r_singleton import REnvironment
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.stats import norm

r_env = REnvironment()


class GAMLSS:
    def __init__(
        self,
        data_table: pd.DataFrame,
        x_column: str,
        y_column: str,
        formula_mu: str = "pb",
        formula_sigma: str = "pb",
        formula_nu: str = "pb",
        formula_tau: str = "1",
        percentiles: list[int] = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        model_path: str = "/app/data/models/",
    ):
        # # --- 1. Setup rpy2 ---
        # pandas2ri.activate()
        self.base = REnvironment().base
        self.stats = REnvironment().stats
        self.grDevices = REnvironment().gr_devices
        self.gamlss_r = REnvironment().gamlss_r
        self.gamlss_dist = REnvironment().gamlss_dist

        # --- 2. Set data ---
        self.data_table = data_table
        self.x_column = x_column
        self.y_column = y_column
        self.percentiles = percentiles

        # --- 3. Set formula ---
        self.formula_mu = robjects.Formula(
            f"{self.y_column} ~ {formula_mu}({self.x_column})"
        )
        self.formula_sigma = robjects.Formula(f"~ {formula_sigma}({self.x_column})")
        self.formula_nu = robjects.Formula(f"~ {formula_nu}({self.x_column})")
        self.formula_tau = robjects.Formula(f"~ {formula_tau}")

        # --- 4. Set model ---
        self.model_path = os.path.abspath(model_path + f"gamlss_{self.y_column}.rds")
        self.model = self._initialize_model()

    def _convert_table_to_r(self, table: pd.DataFrame) -> robjects.DataFrame:
        with localconverter(robjects.default_converter + pandas2ri.converter):
            return robjects.conversion.py2rpy(table)

    def _initialize_model(self) -> robjects.vectors.ListVector | None:
        gamlss_model = None

        if os.path.exists(self.model_path):
            readRDS = self.base.readRDS
            gamlss_model = readRDS(self.model_path)

        return gamlss_model

    def _save_model(self) -> None:
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        save_rds = self.base.saveRDS
        save_rds(self.model, file=self.model_path)

    def _save_run_info(
        self, filename: str = "/app/data/models/run_stats.json"
    ) -> None:
        if not os.path.exists(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

        data_to_save = {
            "dataset_length": len(self.data_table),
            "timestamp": timestamp_str,
        }

        with open(filename, "w") as f:
            json.dump(data_to_save, f)

    @staticmethod
    def _split_structure_name(structure_name: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", " ", structure_name)

    @staticmethod
    def load_run_info(filename: str = "/app/data/models/run_stats.json") -> None:
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
                return data
        return None

    def fit(self) -> None:
        r_table = self._convert_table_to_r(self.data_table)

        self.model = self.gamlss_r.gamlss(
            formula=self.formula_mu,
            sigma_formula=self.formula_sigma,
            nu_formula=self.formula_nu,
            tau_formula=self.formula_tau,
            family="BCPE",
            data=r_table,
            control=self.gamlss_r.gamlss_control(n_cyc=3000, trace=False),
        )

        self._save_model()
        self._save_run_info()

    def calculate_bic(self) -> float:
        bic_r_object = self.stats.BIC(self.model)
        return np.array(bic_r_object)[0]

    def generate_worm_plot(self) -> None:
        plot_filepath = os.path.abspath("worm_plot.png")
        self.grDevices.png(file=plot_filepath, width=7, height=7, units="in", res=150)
        self.gamlss_r.wp(self.model)
        self.grDevices.dev_off()

    def calculate_percentiles(self) -> dict[float, np.array]:
        x_pred_points = np.linspace(
            self.data_table[self.x_column].min(),
            self.data_table[self.x_column].max(),
            200,
        )

        df_pred_r = self._convert_table_to_r(
            pd.DataFrame({self.x_column: x_pred_points})
        )

        pred_params = self.gamlss_r.predictAll(
            object=self.model, newdata=df_pred_r, type="response"
        )

        mu_pred = np.array(pred_params.rx2("mu"))
        sigma_pred = np.array(pred_params.rx2("sigma"))
        nu_pred = np.array(pred_params.rx2("nu"))
        tau_pred = np.array(pred_params.rx2("tau"))

        qBCPE = self.gamlss_dist.qBCPE

        percentile_curves = {}
        for p in self.percentiles:
            p_curve = qBCPE(p=p, mu=mu_pred, sigma=sigma_pred, nu=nu_pred, tau=tau_pred)
            percentile_curves[p] = np.array(p_curve)

        return percentile_curves

    def predict_patient_oos(self, patient_data: pd.DataFrame) -> tuple[float, float]:
        oos_zscore = None
        oos_percentile = None

        centiles_pred_func = self.gamlss_r.centiles_pred
        x_value_r = robjects.FloatVector([patient_data[self.x_column].values[0]])
        y_value_r = robjects.FloatVector([patient_data[self.y_column].values[0]])

        zscore_result_r = centiles_pred_func(
            self.model,
            xvalues=x_value_r,
            yval=y_value_r,
            type="z-scores",
            xname=self.x_column,
        )

        oos_zscore = np.array(zscore_result_r)[0]
        oos_percentile = norm.cdf(oos_zscore)

        return oos_zscore, oos_percentile

    def plot_percentiles(self, percentile_curves: dict[float, np.array]) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.scatter(
            self.data_table[self.x_column],
            self.data_table[self.y_column],
            alpha=0.1,
            label="Data Points (User)",
            s=5,
        )

        x_pred_points = np.linspace(
            self.data_table[self.x_column].min(),
            self.data_table[self.x_column].max(),
            200,
        )

        for p, curve_data in percentile_curves.items():
            linestyle = "-" if p == 0.50 else "--"
            linewidth = 1.5 if p == 0.50 else 1.0
            ax.plot(
                x_pred_points,
                curve_data,
                label=f"{int(p*100)}th Perc (BCPE)",
                linestyle=linestyle,
                linewidth=linewidth,
            )

        ax.set_xlabel("Age")
        ax.set_ylabel("Volume")
        ax.set_title(self._split_structure_name(self.y_column))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plot_min_y = np.percentile(self.data_table[self.y_column], 1)
        plot_max_y = np.percentile(self.data_table[self.y_column], 99)
        plot_range = plot_max_y - plot_min_y
        ax.set_ylim(plot_min_y - 0.1 * plot_range, plot_max_y + 0.1 * plot_range)

        return fig

    def plot_oos_patient(
        self,
        patient_data: pd.DataFrame,
        percentile_curves: dict[float, np.array],
        oos_zscore,
        oos_percentile,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 7))

        for p, curve_data in percentile_curves.items():
            linestyle = "-" if p == 0.50 else "--"
            linewidth = 1.5 if p == 0.50 else 1.0
            alpha_line = 0.5

            label = (
                f"{int(p*100)}th Perc (BCPE Ref)" if p in [0.05, 0.50, 0.95] else None
            )

            ax.plot(
                np.linspace(0, 100, 200),
                curve_data,
                label=label,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha_line,
                color="gray",
            )

        ax.scatter(
            self.data_table[self.x_column],
            self.data_table[self.y_column],
            alpha=0.15,
            label=(
                f"Training Data ({min(self.data_table[self.x_column])}-"
                f"{max(self.data_table[self.x_column])} yrs)"
            ),
            s=15,
            color="lightblue",
        )

        ax.scatter(
            [patient_data[self.x_column].values[0]],
            [patient_data[self.y_column].values[0]],
            color="red",
            s=80,
            zorder=5,
            label=(
                f"Patient (Age={patient_data[self.x_column].values[0]}, "
                f"Vol={patient_data[self.y_column].values[0]})"
            ),
        )

        if oos_percentile:
            ax.annotate(
                f" P {oos_percentile*100:.1f}\n (Z={oos_zscore:.2f})",
                (
                    patient_data[self.x_column].values[0],
                    patient_data[self.y_column].values[0],
                ),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                color="red",
            )
        else:
            raise Exception("No percentile found for OOS patient!")

        ax.set_xlabel("Age")
        ax.set_ylabel("Volume")
        ax.set_title(self._split_structure_name(self.y_column))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plot_min_y = min(
            np.percentile(self.data_table[self.y_column], 1),
            patient_data[self.y_column].values[0] * 0.9,
        )
        plot_max_y = max(
            np.percentile(self.data_table[self.y_column], 99),
            patient_data[self.y_column].values[0] * 1.1,
        )
        ax.set_ylim(plot_min_y, plot_max_y)
        ax.set_xlim(0, 100)

        return fig

    def generate_grids(self) -> plt.Figure:
        if self.model is None:
            self.fit()

        percentile_curves = self.calculate_percentiles()
        plot_figure = self.plot_percentiles(percentile_curves)

        return plot_figure

    def generate_grids_oos(self, patient_data: pd.DataFrame) -> plt.Figure:
        if self.model is None:
            raise Exception("Model has not been trained yet!")

        percentile_curves = self.calculate_percentiles()
        oos_zscore, oos_percentile = self.predict_patient_oos(patient_data)

        plot_figure = self.plot_oos_patient(
            patient_data, percentile_curves, oos_zscore, oos_percentile
        )

        return plot_figure
