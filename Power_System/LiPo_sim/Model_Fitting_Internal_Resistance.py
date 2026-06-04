import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POLY_DEGREE  = 1                   # Polynomial degree
FIT_ON       = 1                   # Which dataset index (1–5) to fit on
BASE_DIR     = r"C:\Users\spash\OneDrive\Desktop\Uni\Bachelor\Year 3\DSE\DSE Code\DSE-09-Drone-Platform-for-Automated-Pest-Control\Power_System\CSV files"
FILE_STEM    = "model_fit_cycle_0_"
NUM_FILES    = 5

# ---------------------------------------------------------------------------
# Load all datasets
# ---------------------------------------------------------------------------
datasets = {}
for i in range(1, NUM_FILES + 1):
    path = os.path.join(BASE_DIR, f"{FILE_STEM}{i}.csv")
    try:
        data = np.loadtxt(path, delimiter=",",skiprows=1)
        datasets[i] = {"soc": data[:, 0], "R0": data[:, 1]}
    except Exception as e:
        print(f"Could not load file {i}: {e}")

# ---------------------------------------------------------------------------
# Fit polynomial on chosen dataset
# ---------------------------------------------------------------------------
x_train = datasets[FIT_ON]["soc"]
y_train = datasets[FIT_ON]["R0"]

model = make_pipeline(
    PolynomialFeatures(degree=POLY_DEGREE, include_bias=False),
    LinearRegression()
)
model.fit(x_train.reshape(-1, 1), y_train)

lin = model.named_steps["linearregression"]

# Build coefficient vector: coeffs[k] = coefficient for SOC^k
coeffs = np.concatenate([[lin.intercept_], lin.coef_])

# ---------------------------------------------------------------------------
# Exported variables (mirroring Final_Model_Fit_Discharge.py)
# ---------------------------------------------------------------------------
intercept = coeffs[0]
coef      = coeffs        # coef[1] … coef[POLY_DEGREE] used in R0_poly

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print(f"--- Polynomial fit (degree {POLY_DEGREE}) trained on dataset {FIT_ON} ---")
for k in range(POLY_DEGREE + 1):
    print(f"  coef[{k}] (SOC^{k}): {coeffs[k]:.6e}")

print(f"\nFitted equation:")
terms = [f"{coeffs[0]:.4e}"]
for k in range(1, POLY_DEGREE + 1):
    terms.append(f"({coeffs[k]:.4e})*SOC^{k}")
print("  R0 = " + " + ".join(terms))

# ---------------------------------------------------------------------------
# Plot — 2x3 grid, fit overlaid on all 5 datasets
# ---------------------------------------------------------------------------
try:
    import scienceplots
    plt.style.use(["science", "no-latex", "grid"])
except ImportError:
    pass

soc_smooth = np.linspace(0, 1, 300)
R0_fit     = np.polyval(coeffs[::-1], soc_smooth)   # polyval needs high-degree first

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (num, data) in enumerate(datasets.items()):
    ax = axes[idx]

    soc = data["soc"]
    R0  = data["R0"]

    R0_pred = model.predict(soc.reshape(-1, 1))
    r2 = r2_score(R0, R0_pred)

    ax.scatter(soc, R0,
               color="darkblue", alpha=0.6, s=12,
               label=f"Actual Cycle {num}")
    ax.plot(soc_smooth, R0_fit,
            color="red", linestyle="--", linewidth=2,
            label=f"Poly fit (trained on {FIT_ON})  R²={r2:.4f}")

    ax.set_title(f"R0 vs SOC — Cycle {num}")
    ax.set_xlabel("SOC (-)")
    ax.set_ylabel("R0 (Ω)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

axes[5].axis("off")

plt.suptitle(f"Degree-{POLY_DEGREE} polynomial fit (trained on dataset {FIT_ON}) vs all cycles",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.show()