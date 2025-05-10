import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.stats import norm

# --- 1. Setup rpy2 ---
pandas2ri.activate()
base = rpackages.importr("base")
stats = rpackages.importr("stats")

# Import grDevices for graphics output (PNG, PDF, etc.)
try:
    grDevices = rpackages.importr("grDevices")
except RRuntimeError as e:
    print(f"Error importing grDevices: {e}")
    exit()

# --- 2. Import the R GAMLSS package & Distribution ---
try:
    gamlss_r = rpackages.importr("gamlss")
    gamlss_dist = rpackages.importr("gamlss.dist")
except RRuntimeError as e:
    print(f"Error importing gamlss packages: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    exit()

# --- 3. Create Sample Data (Using User-Provided Code) ---
# --- Common Settings ---
SIZE = 3000  # Sample size
AGE_VECTOR = np.linspace(0, 100, SIZE)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def generate_conflu_data(age_vec, n_size):
    noise = np.random.randn(n_size) * np.linspace(5, 40, n_size)
    y = 1300 - 5e-3 * ((age_vec - 80) ** 2) - 0.5 * age_vec + noise

    return pd.DataFrame({"y": y, "x": age_vec})


def generate_data_late_peak(age_vec, n_size, t_df=7):
    # Define underlying mean (mu) - logistic growth + quadratic decay
    def later_peak_mu(
        age, peak_age=30, peak_val=700, rise_steepness=0.15, decline_factor=0.005
    ):
        growth = peak_val / (1 + np.exp(-rise_steepness * (age - peak_age / 2)))
        decay = -decline_factor * (np.maximum(0, age - peak_age) ** 2)
        mu = growth + decay - (peak_val / 2)  # Adjust baseline
        return np.maximum(mu, 50)  # Floor value

    mu_true = later_peak_mu(age_vec)

    # Define standard deviation (sigma) - increasing linearly
    sigma_true = 10 + 0.3 * age_vec

    # Generate noise (t-distribution) and final volume (y)
    noise = sigma_true * np.random.standard_t(t_df, n_size)
    y_volume = mu_true + noise

    # Create and return DataFrame
    df_py = pd.DataFrame({"x": age_vec, "y": y_volume})
    return df_py


def generate_data_late_increase(age_vec, n_size):
    # Define underlying mean (mu) - baseline + exponential growth after plateau
    def late_increase_mu(age, baseline=20, plateau_age=50, steepness=0.08):
        mu = baseline + np.exp(steepness * np.maximum(0, age - plateau_age))
        return mu

    mu_true = late_increase_mu(age_vec)

    # Define standard deviation (sigma) - increases with the mean
    sigma_true = 2 + 0.1 * mu_true

    # Generate noise (Normal) and final volume (y)
    noise = np.random.normal(0, sigma_true, n_size)
    y_volume = mu_true + noise

    # Create and return DataFrame
    df_py = pd.DataFrame({"x": age_vec, "y": y_volume})
    return df_py


df_py = generate_conflu_data(AGE_VECTOR, SIZE)

# Check if y is positive for BCPE
if df_py["y"].min() <= 0:
    print(
        f"Warning: Generated data contains non-positive values (min={df_py['y'].min():.2f}). Adding constant for BCPE."
    )
    min_positive = df_py[df_py["y"] > 0]["y"].min() if (df_py["y"] > 0).any() else 0.001
    adjustment = -df_py["y"].min() + min_positive if df_py["y"].min() <= 0 else 0
    df_py["y"] = df_py["y"] + adjustment
    print(f"Adjusted 'y' minimum to be positive: {df_py['y'].min():.2f}")

# --- 4. Convert Python DataFrame to R DataFrame ---
with localconverter(robjects.default_converter + pandas2ri.converter):
    df_r = robjects.conversion.py2rpy(df_py)

# --- 5. Define the GAMLSS model formulas ---
formula_mu = robjects.Formula("y ~ pb(x)")
formula_sigma = robjects.Formula("~ pb(x)")
formula_nu = robjects.Formula("~ pb(x)")
formula_tau = robjects.Formula("~ 1")

# --- 6. Fit the GAMLSS model using BCPE ---
gamlss_model = None
model_filepath_to_load = os.path.abspath("gamlss_model_BCPE.rds")

if os.path.exists(model_filepath_to_load):
    try:
        print(f"\nLoading model from: {model_filepath_to_load}")
        # Get the readRDS function
        readRDS = base.readRDS
        # Load the R object
        gamlss_model = readRDS(model_filepath_to_load)
        print("Model loaded successfully.")
        # Optional: Print summary to verify
        # print(robjects.r.summary(loaded_model))
    except Exception as e:
        print(f"Error loading model from {model_filepath_to_load}: {e}")
else:
    print(f"Error: Model file not found at {model_filepath_to_load}")

try:
    if not gamlss_model:
        gamlss_model = gamlss_r.gamlss(
            formula=formula_mu,
            sigma_formula=formula_sigma,
            nu_formula=formula_nu,
            tau_formula=formula_tau,
            family="BCPE",
            data=df_r,
            control=gamlss_r.gamlss_control(n_cyc=3000, trace=False),
        )

        try:
            model_filename = "gamlss_model_BCPE.rds"  # Use .rds extension
            model_filepath = os.path.abspath(model_filename)
            # Get the saveRDS function from R's base package
            saveRDS = base.saveRDS
            # Save the R object to the file
            saveRDS(gamlss_model, file=model_filepath)
            print(f"\nFitted GAMLSS model saved successfully to: {model_filepath}")
        except Exception as e:
            print(f"Error saving GAMLSS model: {e}")

    # --- 6b. Calculate BIC ---
    # BIC (Bayesian Information Criterion) helps compare models (lower is better)
    try:
        bic_r_object = stats.BIC(gamlss_model)
        bic_value = np.array(bic_r_object)[0]  # Extract the numeric value
    except Exception as e:
        print(f"Could not calculate BIC: {e}")

    # --- 6c. Generate and Save Worm Plot ---
    try:
        plot_filename = "worm_plot.png"
        plot_filepath = os.path.abspath(plot_filename)

        # Activate the PNG graphics device in R
        grDevices.png(file=plot_filepath, width=7, height=7, units="in", res=150)

        # Call the wp function
        gamlss_r.wp(
            gamlss_model
        )  # Checks residuals for mean, variance, skewness, kurtosis trends

        # Close the R graphics device
        grDevices.dev_off()

    except Exception as e:
        print(f"Could not generate/save worm plot: {e}")
        try:
            grDevices.dev_off()
        except RRuntimeError:
            pass

    # --- 7. Prepare for Prediction and Plotting ---
    x_pred_py = np.linspace(df_py["x"].min(), df_py["x"].max(), 200)
    df_pred_py = pd.DataFrame({"x": x_pred_py})
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df_pred_r = robjects.conversion.py2rpy(df_pred_py)

    percentiles_py = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    percentiles_r = robjects.FloatVector(percentiles_py)

    # --- 8. Predict Parameters (mu, sigma, nu, tau) ---
    pred_params = gamlss_r.predictAll(gamlss_model, newdata=df_pred_r, type="response")
    mu_pred = np.array(pred_params.rx2("mu"))
    sigma_pred = np.array(pred_params.rx2("sigma"))
    nu_pred = np.array(pred_params.rx2("nu"))
    tau_pred = np.array(pred_params.rx2("tau"))

    # --- 9. Calculate Percentile Values using qBCPE ---
    qBCPE = gamlss_dist.qBCPE
    percentile_curves = {}
    for p in percentiles_py:
        p_curve = qBCPE(p=p, mu=mu_pred, sigma=sigma_pred, nu=nu_pred, tau=tau_pred)
        percentile_curves[p] = np.array(p_curve)

    # --- 9b: Create and Save Percentile Table ---
    if percentile_curves:
        print("\nCreating percentile table...")
        try:
            # Create a dictionary for the DataFrame columns
            table_data = {"Age": x_pred_py}
            # Add each percentile curve data
            for p in percentiles_py:
                col_name = f"P{int(p*100):02d}"  # e.g., P05, P10, P50, P95
                if p in percentile_curves:
                    table_data[col_name] = percentile_curves[p]
                else:
                    print(
                        f"Warning: Percentile {p} not found in calculated curves for table."
                    )

            # Create pandas DataFrame
            percentile_df = pd.DataFrame(table_data)

            # Define filename for the table
            table_filename = f"percentiles_table_BCPE.csv"
            table_filepath = os.path.abspath(table_filename)

            # Save DataFrame to CSV
            percentile_df.to_csv(table_filepath, index=False, float_format="%.4f")
            print(f"Percentile table saved to: {table_filepath}")
            # Optionally print head of the table
            print("Percentile Table Head:")
            print(percentile_df.head())

        except Exception as e:
            print(f"Error creating or saving percentile table: {e}")

    # --- 10. Plotting using Matplotlib ---
    print("Generating final percentile plot...")
    plt.figure(figsize=(12, 7))
    plt.scatter(df_py["x"], df_py["y"], alpha=0.1, label="Data Points (User)", s=5)
    for p, curve_data in percentile_curves.items():
        linestyle = "-" if p == 0.50 else "--"
        linewidth = 1.5 if p == 0.50 else 1.0
        plt.plot(
            x_pred_py,
            curve_data,
            label=f"{int(p*100)}th Perc (BCPE)",
            linestyle=linestyle,
            linewidth=linewidth,
        )

    plt.xlabel("x")
    plt.ylabel("y (potentially adjusted)")
    plt.title("GAMLSS Model Fit (BCPE) on User-Provided Data")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plot_min_y = np.percentile(df_py["y"], 1)
    plot_max_y = np.percentile(df_py["y"], 99)
    plot_range = plot_max_y - plot_min_y
    plt.ylim(plot_min_y - 0.1 * plot_range, plot_max_y + 0.1 * plot_range)
    plt.show()

    if gamlss_model:
        # --- >>> NEW: Define Out-of-Sample Individual Data <<< ---
        oos_age = 45
        oos_volume = 1300
        print(f"\n--- Out-of-Sample Individual ---")
        print(f"Age: {oos_age}, Volume: {oos_volume}")

        # --- >>> NEW: Calculate Z-score and Percentile for OOS Individual <<< ---
        oos_zscore = None
        oos_percentile = None
        print("\nCalculating Z-score and Percentile for OOS individual...")
        print(
            "WARNING: Applying model to single individual without site calibration - interpret with caution."
        )
        try:
            # Get the centiles.pred function
            centiles_pred_func = gamlss_r.centiles_pred

            # Prepare inputs for centiles.pred
            # Needs the original predictor name used in the formula
            x_name_in_model = "x"
            x_value_r = robjects.FloatVector([oos_age])
            y_value_r = robjects.FloatVector([oos_volume])

            # Call centiles.pred to get the Z-score
            zscore_result_r = centiles_pred_func(
                gamlss_model,
                type="z-scores",
                xname=x_name_in_model,
                xvalues=x_value_r,
                yval=y_value_r,
            )

            # Extract the Z-score (it's usually the first element)
            oos_zscore = np.array(zscore_result_r)[0]

            # Convert Z-score to percentile (CDF of standard normal distribution)
            oos_percentile = norm.cdf(oos_zscore)

            print(f"Calculated Z-score: {oos_zscore:.3f}")
            print(
                f"Calculated Percentile: {oos_percentile:.4f} (or {oos_percentile*100:.2f}%)"
            )

        except RRuntimeError as r_err:
            print(f"R Error calculating Z-score: {r_err}")
        except Exception as e:
            print(f"Error calculating Z-score/Percentile: {e}")

        # --- Plot OOS Individual against Reference Curves ---
        print("\nGenerating plot for OOS individual...")
        plt.figure(figsize=(12, 7))

        # Plot background percentile curves if available
        if percentile_curves:
            for p, curve_data in percentile_curves.items():
                linestyle = "-" if p == 0.50 else "--"
                linewidth = 1.0  # Make reference lines thinner
                alpha_line = 0.5  # Make reference lines lighter
                label = (
                    f"{int(p*100)}th Perc (BCPE Ref)"
                    if p in [0.05, 0.50, 0.95]
                    else None
                )  # Label only a few
                plt.plot(
                    np.linspace(0, 100, 200),
                    curve_data,
                    label=label,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha_line,
                    color="gray",
                )

        # Scatter plot the TRAINING data points (optional, for context)
        plt.scatter(
            df_py["x"],
            df_py["y"],
            alpha=0.15,
            label=f"Training Data (0-100 yrs)",
            s=15,
            color="lightblue",
        )

        # Plot the OOS individual's point
        plt.scatter(
            [oos_age],
            [oos_volume],
            color="red",
            s=80,
            zorder=5,
            label=f"OOS Individual (Age={oos_age}, Vol={oos_volume})",
        )

        # Annotate the OOS point with its score
        if oos_percentile is not None:
            plt.annotate(
                f" P{oos_percentile*100:.1f}\n (Z={oos_zscore:.2f})",
                (oos_age, oos_volume),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                color="red",
            )
        else:
            plt.annotate(
                f"Score N/A",
                (oos_age, oos_volume),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                color="red",
            )

        plt.xlabel("Age")
        plt.ylabel("Volume")
        plt.title(f"GAMLSS Model (BCPE) Trained on 0-100 Yrs: OOS Individual Score")
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        # Adjust limits based on predicted range + OOS point
        ymin = (
            min(np.percentile(df_py["y"], 1), oos_volume * 0.9)
            if percentile_curves
            else oos_volume * 0.9
        )
        ymax = (
            max(np.percentile(df_py["y"], 99), oos_volume * 1.1)
            if percentile_curves
            else oos_volume * 1.1
        )
        plt.ylim(ymin, ymax)
        plt.xlim(0, 100)
        plt.show()

    # --- No separate table needed for a single point, info was printed ---

finally:
    # --- 11. Clean up ---
    try:
        while not base.identical(grDevices.dev_cur(), base.as_integer(1)):
            grDevices.dev_off()
    except Exception as e:
        pass
    pandas2ri.deactivate()

print(f"\n--- Model Diagnostics ---")
print(f"BIC: {bic_value:.2f}")
