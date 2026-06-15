import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


# =========================================================
# 1. MODEL
# =========================================================
def aging_knee_model(n, a, b, c):
    return a - b * np.exp(c * n)


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
# 2. LOAD DATA
# =========================================================
for num, path in file_paths.items():
    try:
        raw_data = np.loadtxt(path)
        raw_cycles = np.arange(len(raw_data)).astype(float)
        y_norm = raw_data / raw_data[0]
        datasets[num] = {"x": raw_cycles, "y": y_norm, "raw_len": len(raw_data)}
    except Exception as e:
        print(f"Could not load Stress_{num}: {e}")

dataset_list = list(datasets.values())
n_datasets = len(dataset_list)

# =========================================================
# 3. JOINT FIT: shared a, b — individual c per dataset
#    params layout: [a, b, c_1, c_2, c_3, c_4, c_5]
# =========================================================
def joint_residuals(params):
    a, b = params[0], params[1]
    c_vals = params[2:]
    residuals = []
    for i, data in enumerate(dataset_list):
        y_pred = aging_knee_model(data["x"], a, b, c_vals[i])
        residuals.append(y_pred - data["y"])
    return np.concatenate(residuals)

x0 = [1.02, 0.005] + [0.008] * n_datasets

result = least_squares(joint_residuals, x0, max_nfev=100_000)

a_opt, b_opt = result.x[0], result.x[1]
c_opts = result.x[2:]

print("--- Globally Optimized Parameters ---")
print(f"a (shared) = {a_opt:.6f}")
print(f"b (shared) = {b_opt:.6f}")
print()
for num, c in zip(datasets.keys(), c_opts):
    print(f"Stress {num}: SOH = {a_opt:.4f} - {b_opt:.5f} * exp({c:.6f} * n)  |  c = {c:.6f}")

# =========================================================
# 4. PLOT
# =========================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (num, data) in enumerate(datasets.items()):
    ax = axes[idx]
    c_val = c_opts[idx]
    y_pred = aging_knee_model(data["x"], a_opt, b_opt, c_val)

    ax.plot(data["x"], data["y"], color="darkblue", alpha=0.8, label=f"Actual Stress {num}")
    ax.plot(data["x"], y_pred, color="red", linestyle="--", linewidth=2,
            label=f"c = {c_val:.5f}")

    ax.set_title(f"Capacity Stress {num} ({data['raw_len']} Cycles)")
    ax.set_xlabel("Raw Cycles")
    ax.set_ylabel("SOH (0 to 1)")
    ax.set_ylim(0.55, 1.10)
    ax.set_xlim(-20, 560)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

axes[5].axis("off")
plt.tight_layout()
plt.show()