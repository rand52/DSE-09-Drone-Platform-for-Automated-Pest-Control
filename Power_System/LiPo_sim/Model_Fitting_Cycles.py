import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error

# =========================================================
# LOAD TRAINING DATA (capacity only)
# =========================================================
train_path = r"C:\Users\spash\OneDrive\Desktop\Uni\Bachelor\Year 3\DSE\DSE Code\DSE-09-Drone-Platform-for-Automated-Pest-Control\Power_System\CSV files\Capacity_stress_3.csv"

train_data = np.loadtxt(train_path)

capacity_train = train_data

cycle_train = np.arange(len(capacity_train))  # cycle index = row number

X_train = cycle_train.reshape(-1, 1)
y_train = capacity_train


# =========================================================
# LOAD VALIDATION DATA (capacity only)
# =========================================================
val_path = r"C:\Users\spash\OneDrive\Desktop\Uni\Bachelor\Year 3\DSE\DSE Code\DSE-09-Drone-Platform-for-Automated-Pest-Control\Power_System\CSV files\Capacity_stress_4.csv"

val_data = np.loadtxt(val_path)

capacity_val = val_data

cycle_val = np.arange(len(capacity_val))  # cycle index = row number

X_val = cycle_val.reshape(-1, 1)
y_val = capacity_val


# =========================================================
# POLYNOMIAL DEGREE SWEEP
# =========================================================
degrees = range(1, 16)

train_rmse_list = []
val_rmse_list = []

best_degree = None
best_val_rmse = np.inf

for degree in degrees:

    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # predictions
    y_train_pred = model.predict(X_train_poly)
    y_val_pred = model.predict(X_val_poly)

    # RMSE
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    val_rmse = root_mean_squared_error(y_val, y_val_pred)

    train_rmse_list.append(train_rmse)
    val_rmse_list.append(val_rmse)

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_degree = degree

    print(
        f"Degree {degree:2d} | "
        f"Train RMSE = {train_rmse:.6f} | "
        f"Val RMSE = {val_rmse:.6f}"
    )


# =========================================================
# BEST MODEL
# =========================================================
print("\n====================================")
print(f"Best polynomial degree: {best_degree}")
print(f"Best validation RMSE: {best_val_rmse:.6f}")
print("====================================")


# =========================================================
# RMSE PLOT
# =========================================================
plt.figure(figsize=(8, 5))

plt.plot(degrees, train_rmse_list, marker='o', label='Training RMSE')
plt.plot(degrees, val_rmse_list, marker='o', label='Validation RMSE')

plt.axvline(best_degree, linestyle='--', label=f'Best Degree = {best_degree}')

plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")
plt.title("Capacity Fade Model Selection")
plt.grid(True)
plt.legend()

plt.show()


# =========================================================
# FINAL MODEL FIT (BEST DEGREE)
# =========================================================
poly_best = PolynomialFeatures(degree=best_degree)

X_train_best = poly_best.fit_transform(X_train)
X_val_best = poly_best.transform(X_val)

final_model = LinearRegression()
final_model.fit(X_train_best, y_train)

y_val_pred = final_model.predict(X_val_best)


# =========================================================
# PLOT 2: VALIDATION VS PREDICTION
# =========================================================
plt.figure(figsize=(8, 5))

plt.plot(X_val, y_val, 'o', label="Validation Data")
plt.plot(X_val, y_val_pred, '-', label=f"Polynomial Prediction (deg {best_degree})")

plt.xlabel("Cycle Number")
plt.ylabel("Capacity")
plt.title("Validation vs Model Prediction")
plt.grid(True)
plt.legend()

plt.show()



