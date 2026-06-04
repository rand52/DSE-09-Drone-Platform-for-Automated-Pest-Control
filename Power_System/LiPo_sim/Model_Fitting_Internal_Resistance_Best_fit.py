import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import os
if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # Configuration  ← change these
    # ---------------------------------------------------------------------------
    TRAIN_ON    = 1          # Dataset index used for training  (1–5)
    VALIDATE_ON = 4          # Dataset index used for validation (1–5)
    MAX_DEGREE  = 1         # Highest polynomial degree to sweep

    BASE_DIR  = r"C:\Users\spash\OneDrive\Desktop\Uni\Bachelor\Year 3\DSE\DSE Code\DSE-09-Drone-Platform-for-Automated-Pest-Control\Power_System\CSV files"
    FILE_STEM = "model_fit_cycle_0_"
    NUM_FILES = 5

    # ---------------------------------------------------------------------------
    # Load all datasets
    # ---------------------------------------------------------------------------
    datasets = {}
    for i in range(1, NUM_FILES + 1):
        path = os.path.join(BASE_DIR, f"{FILE_STEM}{i}.csv")
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            datasets[i] = {"soc": data[:, 0], "R0": data[:, 1]}
        except Exception as e:
            print(f"Could not load file {i}: {e}")

    x_train = datasets[TRAIN_ON]["soc"].reshape(-1, 1)
    y_train = datasets[TRAIN_ON]["R0"]

    x_val   = datasets[VALIDATE_ON]["soc"].reshape(-1, 1)
    y_val   = datasets[VALIDATE_ON]["R0"]

    # ---------------------------------------------------------------------------
    # Sweep degrees
    # ---------------------------------------------------------------------------
    degrees        = np.arange(1, MAX_DEGREE + 1)
    train_rmse     = []
    val_rmse       = []
    best_rmse      = np.inf
    best_degree    = 1
    best_model     = None

    for deg in degrees:
        model = make_pipeline(
            PolynomialFeatures(degree=deg, include_bias=False),
            LinearRegression()
        )
        model.fit(x_train, y_train)

        t_rmse = np.sqrt(mean_squared_error(y_train, model.predict(x_train)))
        v_rmse = np.sqrt(mean_squared_error(y_val,   model.predict(x_val)))

        train_rmse.append(t_rmse)
        val_rmse.append(v_rmse)

        if v_rmse < best_rmse:
            best_rmse   = v_rmse
            best_degree = deg
            best_model  = model

        print(f"Degree {deg:2d}  |  Train RMSE: {t_rmse:.6f}  |  Val RMSE: {v_rmse:.6f}")

    print(f"\n→ Best degree by validation RMSE: {best_degree}  (RMSE = {best_rmse:.6f})")

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    try:
        import scienceplots
        plt.style.use(["science", "no-latex", "grid"])
    except ImportError:
        pass

    soc_smooth = np.linspace(0, 1, 300).reshape(-1, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: RMSE vs degree ---
    ax = axes[0]
    ax.plot(degrees, train_rmse, "o-", color="steelblue",  linewidth=2, label="Train RMSE")
    ax.plot(degrees, val_rmse,   "s-", color="tomato",     linewidth=2, label="Validation RMSE")
    ax.axvline(best_degree, color="green", linestyle="--", linewidth=1.5,
            label=f"Best degree = {best_degree}")
    ax.set_xlabel("Polynomial Degree (-)")
    ax.set_ylabel("RMSE (Ω)")
    ax.set_title(f"RMSE vs Degree  |  Train: dataset {TRAIN_ON}  |  Val: dataset {VALIDATE_ON}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Right: best fit overlaid on train + val data ---
    ax2 = axes[1]
    ax2.scatter(datasets[TRAIN_ON]["soc"],    datasets[TRAIN_ON]["R0"],
                color="steelblue", s=12, alpha=0.6, label=f"Train data (cycle {TRAIN_ON})")
    ax2.scatter(datasets[VALIDATE_ON]["soc"], datasets[VALIDATE_ON]["R0"],
                color="tomato",    s=12, alpha=0.6, label=f"Val data   (cycle {VALIDATE_ON})")
    ax2.plot(soc_smooth, best_model.predict(soc_smooth),
            color="black", linewidth=2, linestyle="--",
            label=f"Best fit (degree {best_degree})")
    ax2.set_xlabel("SOC (-)")
    ax2.set_ylabel("R0 (Ω)")
    ax2.set_title(f"Best polynomial fit (degree {best_degree})")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------------
    # Train final model on ALL available data using best degree
    # ---------------------------------------------------------------------------

    # Combine all datasets

    # Rebuild and retrain best model
    final_model = make_pipeline(
        PolynomialFeatures(degree=best_degree, include_bias=False),
        LinearRegression()
    )

    final_model.fit(x_train, y_train)
    linreg = final_model.named_steps["linearregression"]

    intercept_resistance = linreg.intercept_
    coef_resistance = linreg.coef_

    print("\nFinal model trained on ALL data")
    print(f"Degree: {best_degree}")
    print(f"Coefficients: {final_model.named_steps['linearregression'].coef_}")
    print(f"Intercept: {final_model.named_steps['linearregression'].intercept_}")