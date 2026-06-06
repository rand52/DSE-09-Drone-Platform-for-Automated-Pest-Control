import os
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR     = r"C:\Users\spash\OneDrive\Desktop\Uni\Bachelor\Year 3\DSE\DSE Code\DSE-09-Drone-Platform-for-Automated-Pest-Control\Power_System\CSV files"
FILE_PATTERN = "model_fit_cycle_{}"
FILE_EXT     = ".csv"
CYCLES       = list(range(0, 316, 45))  # 0, 45, 90, 135, 180, 225, 270, 315

# --- Extract value from each file ---
cycles_found = []
r0_values    = []

for cyc in CYCLES:
    filepath = os.path.join(DATA_DIR, FILE_PATTERN.format(cyc) + FILE_EXT)
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()

        # Skip title row (index 0), read first data row (index 1), second column (index 1)
        data_line = lines[1].strip()
        value = float(data_line.split(",")[1])

        cycles_found.append(cyc)
        r0_values.append(value)
        

    except FileNotFoundError:
        print(f"[WARN] File not found: {filepath}, skipping.")
    except Exception as e:
        print(f"[ERROR] {filepath}: {e}")

if len(cycles_found) < 2:
    raise RuntimeError("Need at least 2 data points to fit a model.")

cycles_arr = np.array(cycles_found, dtype=float)
r0_arr     = np.array(r0_values,    dtype=float)

# --- Linear model: R0(n) = R0_0 * (1 + alpha * n) ---
# Fit fractional growth y = (R0(n) - R0_0) / R0_0 = alpha * n  (through origin)
R0_0  = r0_arr[0]
y     = r0_arr / R0_0 - 1.0
alpha = np.dot(cycles_arr, y) / np.dot(cycles_arr, cycles_arr)


# --- Prediction function ---
def r0_growth_factor(cycle_number: float) -> float:
    """Returns the factor by which R0 has grown at a given cycle number.
    Usage: R0_at_cycle_n = R0_initial * r0_growth_factor(n)
    """
    return 1.0 + alpha * cycle_number
if __name__ == '__main__':
# --- Plot ---
    n_plot = np.linspace(0, max(cycles_arr) * 1.05, 300)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("R0 Cycle Degradation Model", fontsize=14, fontweight="bold")

    ax1.scatter(cycles_arr, r0_arr, color="steelblue", s=60, zorder=5, label="Measured R0")
    ax1.plot(n_plot, R0_0 * (1 + alpha * n_plot), color="tomato", linewidth=2,
            label=f"Linear fit\nR0(n) = R0₀·(1 + {alpha:.3e}·n)")
    ax1.set_xlabel("Cycle number")
    ax1.set_ylabel("R0 (Ω)")
    ax1.set_title("Absolute R0 vs Cycle")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.scatter(cycles_arr, y, color="steelblue", s=60, zorder=5, label="Measured fractional increase")
    ax2.plot(n_plot, alpha * n_plot, color="tomato", linewidth=2,
            label=f"Fit: α = {alpha:.3e} per cycle")
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Cycle number")
    ax2.set_ylabel("(R0(n) − R0₀) / R0₀")
    ax2.set_title("Fractional R0 Growth vs Cycle")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- Usage demo ---
    print("\n--- Usage example ---")
    for test_cycle in [0, 45, 90, 180, 315]:
        gf = r0_growth_factor(test_cycle)
        print(f"  Cycle {test_cycle:>3d}: growth factor = {gf:.4f}  →  R0 = R0_initial × {gf:.4f}")