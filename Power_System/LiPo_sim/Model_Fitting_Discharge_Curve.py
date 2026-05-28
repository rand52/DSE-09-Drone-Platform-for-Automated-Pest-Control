import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error

# ── Paths ────────────────────────────────────────────────────────────────────
battery_csv_path = r"C:\Users\spash\OneDrive\Desktop\Uni\Bachelor\Year 3\DSE\DSE Code\DSE-09-Drone-Platform-for-Automated-Pest-Control\Power_System\CSV files\0_Discharge_std_3.csv"
validation_battery_data_path = r"C:\Users\spash\OneDrive\Desktop\Uni\Bachelor\Year 3\DSE\DSE Code\DSE-09-Drone-Platform-for-Automated-Pest-Control\Power_System\CSV files\0_Discharge_std_4.csv"

# ── Load data ─────────────────────────────────────────────────────────────────
battery_data = np.loadtxt(battery_csv_path, delimiter="\t")
extracted_charge_train = battery_data[:, 0]
voltage_train = battery_data[:, 1]
capacity_train = extracted_charge_train[-1]
soc_train = (capacity_train - extracted_charge_train) / capacity_train

validation_data = np.loadtxt(validation_battery_data_path, delimiter="\t")
extracted_charge_val = validation_data[:, 0]
voltage_val = validation_data[:, 1]
capacity_val = extracted_charge_val[-1]
soc_val = (capacity_val - extracted_charge_val) / capacity_val

X_train = soc_train.reshape(-1, 1)
X_val = soc_val.reshape(-1, 1)

# ── Degree sweep ──────────────────────────────────────────────────────────────
degrees = range(1, 16)           # test degrees 1 – 15; adjust as needed

train_rmses = []
val_rmses = []
models = {}

for deg in degrees:
    poly = PolynomialFeatures(degree=deg)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    model = LinearRegression()
    model.fit(X_train_poly, voltage_train)

    train_pred = model.predict(X_train_poly)
    val_pred = model.predict(X_val_poly)

    train_rmse = root_mean_squared_error(voltage_train, train_pred)
    val_rmse = root_mean_squared_error(voltage_val, val_pred)

    train_rmses.append(train_rmse)
    val_rmses.append(val_rmse)
    models[deg] = (poly, model)

    print(f"Degree {deg:2d}  |  Train RMSE: {train_rmse:.6f}  |  Val RMSE: {val_rmse:.6f}")

best_degree = degrees[int(np.argmin(val_rmses))]
print(f"\nBest degree by validation RMSE: {best_degree}")

# ── Plot 1: RMSE vs Degree ────────────────────────────────────────────────────
plt.style.use(['science', 'no-latex', 'grid'])

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(list(degrees), train_rmses, marker="o", label="Training RMSE", linewidth=1.5)
ax.plot(list(degrees), val_rmses,   marker="s", label="Validation RMSE", linewidth=1.5)
ax.axvline(best_degree, color="red", linestyle="--", linewidth=1,
           label=f"Best degree = {best_degree}")

ax.set_xlabel("Polynomial Degree")
ax.set_ylabel("RMSE (V)")
ax.set_title("Polynomial Degree Selection — RMSE vs Degree")
ax.legend()
ax.set_xticks(list(degrees))
plt.tight_layout()
plt.show()
'''
# ── Plot 2: Best-fit curve overlaid on data ───────────────────────────────────
best_poly, best_model = models[best_degree]

soc_fine = np.linspace(0, 1, 500).reshape(-1, 1)
voltage_pred_fine = best_model.predict(best_poly.transform(soc_fine))

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(soc_train, voltage_train, s=4, alpha=0.5, label="Training data")
ax.scatter(soc_val,   voltage_val,   s=4, alpha=0.5, label="Validation data")
ax.plot(soc_fine, voltage_pred_fine, color="red", linewidth=1.5,
        label=f"Degree-{best_degree} fit")

ax.set_xlabel("State of Charge (–)")
ax.set_ylabel("Voltage (V)")
ax.set_title(f"OCV–SoC Model — Degree {best_degree} Polynomial")
ax.legend()
plt.tight_layout()
plt.show()
'''