import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import scienceplots
plt.style.use(['science','no-latex','grid'])
# Load the cleaned data
df = pd.read_csv('cleaned_battery_specs_jens.csv')

df['current (A)'] = df['capacity_mAh'] / 1000 * df['discharge_C']
df['calculated power (W)'] = df['current (A)'] * df['voltage_V']

def regressionfunc(x, a, b):
    return a + b*x

mask = df['weight_g'].notna() & df['calculated power (W)'].notna() & (20 < df['weight_g']) & (df['weight_g'] < 90) & (df['calculated power (W)'] < 1500)
x = df.loc[mask, 'calculated power (W)'].to_numpy(dtype=float)
y = df.loc[mask, 'weight_g'].to_numpy(dtype=float)

popt, pcov = curve_fit(regressionfunc, x, y)

# Create the scatter plot
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(x, y, alpha=0.6, color='blue', edgecolors='k', s=30)
# for i in range(len(x)):
#     plt.text(x[i], y[i], df.loc[mask].iloc[i]['product_name'])
# Plot the fitted curve
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = regressionfunc(x_fit, *popt)
ax.plot(x_fit, y_fit, 'r--', label=f'Fitted Curve: $y = {popt[0]:.2f} + {popt[1]:.2f}x$')
# Add labels and title
ax.set_xlabel('Calculated Power (W)')
ax.set_ylabel('Weight (g)')
# Adjust layout and save the figure
ax.legend()
fig.tight_layout()
fig.savefig('battery_weight_vs_power.png', dpi=300, bbox_inches='tight')
plt.show()