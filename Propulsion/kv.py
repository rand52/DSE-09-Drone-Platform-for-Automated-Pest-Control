import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# Load the cleaned data
df = pd.read_csv('drone_motor_specs_scraped.csv')

# Drop rows where 'weight' or 'kv' are missing to ensure a clean plot
mask = df['kv'] < 10000
plot_data = df[mask].dropna(subset=['weight', 'kv'])

def regressionfunc(x, a, b):
    return a + b*x

# Fit the exponential curve
popt, pcov = curve_fit(regressionfunc, np.log(plot_data['kv']), np.log(plot_data['weight']))

# Create the scatter plot
plt.scatter(np.log(plot_data['kv']), np.log(plot_data['weight']), alpha=0.6, color='blue', edgecolors='k')

# Plot the fitted curve
x_fit = np.linspace(np.log(plot_data['kv'].min()), np.log(plot_data['kv'].max()), 100)
y_fit = regressionfunc(x_fit, *popt)
plt.plot(x_fit, y_fit, 'r--', label=f'Fitted Curve: y = {popt[0]:.2f} + {popt[1]:.2f} * x')

# Add labels and title
plt.title('Drone Motor Weight vs. KV Rating')
plt.xlabel('KV (rpm/V)')
plt.ylabel('Weight ($g$)')
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and save the figure
plt.legend()
plt.tight_layout()
plt.show()

x_exp_fit = np.linspace(plot_data['kv'].min(), plot_data['kv'].max(), 100)
y_exp_fit = np.exp(regressionfunc(np.log(x_exp_fit), *popt))
plt.plot(x_exp_fit, y_exp_fit, 'r--', label=f'Exponential Fit: y = {np.exp(popt[0]):.2f} * e^({popt[1]:.2f} * x)')
plt.plot(plot_data['kv'], plot_data['weight'], 'bo', alpha=0.6, label='Data Points')

# Add labels and title
plt.title('Drone Motor Weight vs. KV Rating')
plt.xlabel('KV (rpm/V)')
plt.ylabel('Weight ($g$)')
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and save the figure
plt.legend()
plt.tight_layout()
plt.show()