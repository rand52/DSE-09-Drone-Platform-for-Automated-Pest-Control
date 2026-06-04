import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# ---------------------------------------------------------------------------
# Configuration  ← change these
# ---------------------------------------------------------------------------
TRAIN_ON    = 1          # Dataset index used for training  (1–5)
VALIDATE_ON = 4          # Dataset index used for validation (1–5)

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

# ---------------------------------------------------------------------------
# Normalise R0 to percentage of the initial (first) R0 in the training set
# ---------------------------------------------------------------------------
R0_initial = datasets[TRAIN_ON]["R0"][0]   # reference value (Ω at first sample)


x_train = datasets[TRAIN_ON]["soc"].reshape(-1, 1)
y_train = datasets[TRAIN_ON]["R0"] / R0_initial # now in frac of initial R0

# ---------------------------------------------------------------------------
# Train linear model on the percentage-normalised target
# ---------------------------------------------------------------------------
final_model = LinearRegression()
final_model.fit(x_train, y_train)

intercept_res = final_model.intercept_   # frac of R0_initial when SoC = 0
coef_res      = final_model.coef_        # frac change per unit SoC

