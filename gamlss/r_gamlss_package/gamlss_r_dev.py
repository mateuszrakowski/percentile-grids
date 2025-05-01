import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.stats import skewnorm

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
SIZE = 1000  # Sample size
AGE_VECTOR = np.linspace(0, 100, SIZE)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def generate_conflu_data(age_vec, n_size):
    noise = np.random.randn(n_size) * np.linspace(5, 40, n_size)
    y = 1300 - 5e-3 * ((age_vec - 80) ** 2) - 0.5 * age_vec + noise

    return pd.DataFrame({"y": y, "x": age_vec})


# --- Function for Example 1: Early Peak, Steady Decline ---
def generate_data_early_peak(age_vec, n_size):
    """
    Generates data mimicking early peak and steady decline (like Grey Matter).
    Returns a pandas DataFrame with 'age' and 'volume' columns.
    """
    print("Generating Data: Example 1 (Early Peak, Steady Decline)")

    # Define underlying mean (mu) - skewed peak + linear decline
    def skewed_peak_mu(age, peak_age=6, peak_val=800, growth_scale=5, decline_rate=1.5):
        relative_age = (age - peak_age + growth_scale * 3) / growth_scale
        rise = skewnorm.pdf(relative_age, a=4)  # a=4 gives right skew for rise phase
        rise_normalized = rise / skewnorm.pdf(3, a=4)  # Approximate normalization
        mu = np.where(
            age <= peak_age,
            peak_val * rise_normalized,
            peak_val - decline_rate * (age - peak_age),
        )
        return np.maximum(mu, 50)  # Floor value

    mu_true = skewed_peak_mu(age_vec)

    # Define standard deviation (sigma) - mild non-linear change
    sigma_true = 15 + 10 * np.sin(np.pi * age_vec / 100) + 0.1 * age_vec

    # Generate noise and final volume (y)
    noise = np.random.normal(0, sigma_true, n_size)
    y_volume = mu_true + noise

    # Create and return DataFrame
    df_py = pd.DataFrame({"x": age_vec, "y": y_volume})
    print("Example 1 DataFrame head:\n", df_py.head())
    return df_py


# --- Function for Example 2: Later Peak, Slower Decline ---
def generate_data_late_peak(age_vec, n_size, t_df=7):
    """
    Generates data mimicking later peak and slower decline (like White Matter).
    Uses t-distributed noise.
    Returns a pandas DataFrame with 'age' and 'volume' columns.
    """
    print("\nGenerating Data: Example 2 (Later Peak, Slower Decline)")

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
    print("Example 2 DataFrame head:\n", df_py.head())
    return df_py


# --- Function for Example 3: Late Exponential Increase ---
def generate_data_late_increase(age_vec, n_size):
    """
    Generates data mimicking late exponential increase (like Ventricles).
    Variance increases with the mean.
    Returns a pandas DataFrame with 'age' and 'volume' columns.
    """
    print("\nGenerating Data: Example 3 (Late Exponential Increase)")

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
    print("Example 3 DataFrame head:\n", df_py.head())
    return df_py


df_py = generate_data_late_increase(AGE_VECTOR, SIZE)

# Check if y is positive for BCPE
if df_py["y"].min() <= 0:
    print(
        f"Warning: Generated data contains non-positive values (min={df_py['y'].min():.2f}). Adding constant for BCPE."
    )
    min_positive = df_py[df_py["y"] > 0]["y"].min() if (df_py["y"] > 0).any() else 0.001
    adjustment = -df_py["y"].min() + min_positive if df_py["y"].min() <= 0 else 0
    df_py["y"] = df_py["y"] + adjustment
    print(f"Adjusted 'y' minimum to be positive: {df_py['y'].min():.2f}")

print("\nSample Python DataFrame head (user data):")
print(df_py.head())


# --- 4. Convert Python DataFrame to R DataFrame ---
with localconverter(robjects.default_converter + pandas2ri.converter):
    df_r = robjects.conversion.py2rpy(df_py)

print(f"\nConverted data to R DataFrame type: {type(df_r)}")

# --- 5. Define the GAMLSS model formulas ---
# Using formulas reflecting data generation process
print("\nDefining GAMLSS formulas:")
formula_mu = robjects.Formula("y ~ pb(x)")
formula_sigma = robjects.Formula("~ poly(x)")
formula_nu = robjects.Formula("~ x")
formula_tau = robjects.Formula("~ 1")

# --- 6. Fit the GAMLSS model using BCPE ---
print("\nFitting GAMLSS model (BCPE family)...")
gamlss_model = None  # Initialize variable to None
try:
    gamlss_model = gamlss_r.gamlss(
        formula=formula_mu,
        sigma_formula=formula_sigma,
        nu_formula=formula_nu,
        tau_formula=formula_tau,
        family="BCPE",
        data=df_r,
        control=gamlss_r.gamlss_control(n_cyc=100, trace=False),
    )
    print("Model fitting complete.")

    # --- 6b. Calculate and Print BIC ---
    # BIC (Bayesian Information Criterion) helps compare models (lower is better)
    try:
        bic_r_object = stats.BIC(gamlss_model)
        # Extract the numeric value (often the first element)
        bic_value = np.array(bic_r_object)[0]
        print(f"\n--- Model Diagnostics ---")
        print(f"BIC: {bic_value:.2f}")
        print(
            "   (Lower BIC is generally better when comparing models on the same data)"
        )
    except Exception as e:
        print(f"Could not calculate BIC: {e}")

    # --- 6c. Generate and Save Worm Plot ---
    # wp() checks residuals for mean, variance, skewness, kurtosis trends
    try:
        plot_filename = "worm_plot.png"
        plot_filepath = os.path.abspath(plot_filename)  # Get full path for clarity

        # Activate the PNG graphics device in R
        grDevices.png(file=plot_filepath, width=7, height=7, units="in", res=150)

        # Call the wp function
        print(f"\nGenerating worm plot (saving to {plot_filepath})...")
        gamlss_r.wp(
            gamlss_model
        )  # Might add arguments like xvar=df_r.rx2('x') if needed

        # Close the R graphics device
        grDevices.dev_off()
        print("   Worm plot saved. Check the file.")
        print("   (Ideal: Worms lie flat within confidence bands)")

    except Exception as e:
        print(f"Could not generate/save worm plot: {e}")
        # Ensure grDevices.dev_off() is called even if wp() fails mid-plot
        try:
            grDevices.dev_off()
        except RRuntimeError:
            pass  # Device might already be off

    # --- 7. Prepare for Prediction and Plotting ---
    # (This section remains the same)
    print("\nPreparing data for prediction...")
    x_pred_py = np.linspace(df_py["x"].min(), df_py["x"].max(), 200)
    df_pred_py = pd.DataFrame({"x": x_pred_py})
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df_pred_r = robjects.conversion.py2rpy(df_pred_py)

    percentiles_py = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    percentiles_r = robjects.FloatVector(percentiles_py)

    # --- 8. Predict Parameters (mu, sigma, nu, tau) ---
    # (This section remains the same)
    print("Predicting parameters (mu, sigma, nu, tau) for plotting...")
    pred_params = gamlss_r.predictAll(gamlss_model, newdata=df_pred_r, type="response")
    mu_pred = np.array(pred_params.rx2("mu"))
    sigma_pred = np.array(pred_params.rx2("sigma"))
    nu_pred = np.array(pred_params.rx2("nu"))
    tau_pred = np.array(pred_params.rx2("tau"))

    # --- 9. Calculate Percentile Values using qBCPE ---
    # (This section remains the same)
    qBCPE = gamlss_dist.qBCPE
    print("Calculating percentiles using qBCPE...")
    percentile_curves = {}
    for p in percentiles_py:
        p_curve = qBCPE(p=p, mu=mu_pred, sigma=sigma_pred, nu=nu_pred, tau=tau_pred)
        percentile_curves[p] = np.array(p_curve)

    # --- >>> NEW Section 9b: Create and Save Percentile Table <<< ---
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
    # --- >>> END NEW Section <<< ---

    # --- 10. Plotting using Matplotlib ---
    # (This section remains the same, but now follows diagnostics)
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


# --- Error Handling for Fitting ---
except RRuntimeError as e:
    print(f"An R runtime error occurred during GAMLSS fitting: {e}")
    print(
        "Check model formulas, data suitability (y>0?), and consider simplifying the model or increasing control parameters (e.g., n.cyc)."
    )
except Exception as e:
    print(f"An unexpected error occurred during fitting or subsequent steps: {e}")

finally:
    # --- 11. Clean up ---
    # Ensure R graphics device is turned off if something failed above
    try:
        # This checks if a device is active and turns it off.
        # Useful if the script failed between png() and dev.off()
        while not base.identical(
            grDevices.dev_cur(), base.as_integer(1)
        ):  # Check if current device is not the null device
            grDevices.dev_off()
    except Exception as e:
        # print(f"Note: Error during final graphics device cleanup (might be harmless): {e}")
        pass  # Ignore cleanup errors if device handling failed earlier

    pandas2ri.deactivate()
    print("\nDeactivated pandas converter.")
