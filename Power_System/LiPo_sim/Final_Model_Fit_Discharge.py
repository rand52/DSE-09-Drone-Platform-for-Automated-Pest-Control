import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error


# Loading Data
battery_csv_path = r"C:\Users\spash\OneDrive\Desktop\Uni\Bachelor\Year 3\DSE\DSE Code\DSE-09-Drone-Platform-for-Automated-Pest-Control\Power_System\CSV files\0_Discharge_std_3.csv"
battery_data = np.loadtxt(battery_csv_path, delimiter="\t")
extracted_charge_linreg = battery_data[:,0]
voltage_linreg = battery_data[:,1]
capacity_linreg = extracted_charge_linreg[-1] #C apacity of tested battery at current cycle, used only to derive mathematical model [Ah]

    # Calculations
state_of_charge_linreg = (capacity_linreg - extracted_charge_linreg) / capacity_linreg # State of charge vector as franction a.k.a % / 100

    # Model fitting
polyfit = PolynomialFeatures(degree = 4)
train_soc = polyfit.fit_transform(state_of_charge_linreg.reshape(-1,1))
voltage_curve_model = LinearRegression()
voltage_curve_model.fit(train_soc,voltage_linreg)
intercept = voltage_curve_model.intercept_
coef = voltage_curve_model.coef_

