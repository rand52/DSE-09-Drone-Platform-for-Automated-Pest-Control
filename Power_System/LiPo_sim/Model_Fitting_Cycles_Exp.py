import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# =========================================================
# 1. DEFINE YOUR 3-PARAMETER AGING KNEE MODEL
# =========================================================
def aging_knee_model(n, a, b, c):
    return a - b * np.exp(c * n)


# Dictionary mapping dataset numbers to their file paths
base_dir = r"C:\Users\spash\OneDrive\Desktop\Uni\Bachelor\Year 3\DSE\DSE Code\DSE-09-Drone-Platform-for-Automated-Pest-Control\Power_System\CSV files"

file_paths = {
    1: f"{base_dir}\\Capacity_stress_1.csv",
    2: f"{base_dir}\\Capacity_stress_2.csv",
    3: f"{base_dir}\\Capacity_stress_3.csv",
    4: f"{base_dir}\\Capacity_stress_4.csv",
    5: f"{base_dir}\\Capacity_stress_5.csv",
}

datasets = {}

# =========================================================
# 2. LOAD AND PROCESS (RAW CYCLES, NORMALISED SOH)
# =========================================================
for num, path in file_paths.items():
    try:
        raw_data = np.loadtxt(path)

        # KEEP CYCLES AS RAW NUMBERS
        raw_cycles = np.arange(len(raw_data)).astype(float)

        # Normalize capacity to State of Health (SOH) starting at 1.0
        y_norm = raw_data / raw_data[0]

        datasets[num] = {"x": raw_cycles, "y": y_norm, "raw_len": len(raw_data)}
    except Exception as e:
        print(f"Could not load Stress_{num}: {e}")

# =========================================================
# 3. FIT THE 3-PARAMETER MODEL ON STRESS 5
# =========================================================
x_train = datasets[4]["x"]
y_train = datasets[4]["y"]

# TARGETED GUESSES FOR RAW CYCLES:
# a = 1.02 (starts right near the top)
# b = 0.005 (keeps the initial drop tiny)
# c = 0.008 (a small positive fraction so it doesn't overflow over 500 cycles)
initial_guess = [1.02, 0.005, 0.008]

popt, _ = curve_fit(
    aging_knee_model, x_train, y_train, p0=initial_guess, maxfev=100000
)

print("--- Optimized Parameters (3-Parameter Model on Stress 5) ---")
print(f"a (Baseline Ceiling): {popt[0]:.6f}")
print(f"b (Scaling Factor) : {popt[1]:.6f}")
print(f"c (Growth Rate)    : {popt[2]:.6f}")
print(f"\nFitted Equation: SOH = {popt[0]:.3f} - {popt[1]:.5f} * e^({popt[2]:.5f} * n)")

# =========================================================
# 4. VISUALIZE ALL 5 PLOTS
# =========================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (num, data) in enumerate(datasets.items()):
    ax = axes[idx]

    # Predict using the 3-parameter model parameters
    y_pred = aging_knee_model(data["x"], *popt)

    ax.plot(
        data["x"],
        data["y"],
        color="darkblue",
        alpha=0.8,
        label=f"Actual Stress {num}",
    )
    ax.plot(
        data["x"],
        y_pred,
        color="red",
        linestyle="--",
        linewidth=2,
        label="3-Param Model Prediction",
    )

    ax.set_title(f"Capacity Stress {num} ({data['raw_len']} Cycles)")
    ax.set_xlabel("Raw Cycles")
    ax.set_ylabel("SOH (0 to 1)")
    ax.set_ylim(0.55, 1.10)

    # Set universal x-limit to visually compare their lifetimes
    ax.set_xlim(-20, 560)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

axes[5].axis("off")
plt.tight_layout()
plt.show()