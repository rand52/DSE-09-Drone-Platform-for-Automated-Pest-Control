import numpy as np
import matplotlib.pyplot as plt
import Comparison_Params as cp

# General Flight parameters

total_flight_time = cp.total_flight_time_short #s
t_peak = cp.t_peak #s time peak power is required
avg_to_peak_ratio = cp.avg_to_peak_ratio

# Values for possible combinations

combination = np.array([[
                        [cp.E_density_li_ion_classI,cp.E_density_lipo_classI,cp.E_density_supercap_classI],
                        [cp.P_density_li_ion_classI,cp.P_density_lipo_classI,cp.P_density_supercap_classI]
                        ],
                        [
                        [cp.E_density_li_ion_classII,cp.E_density_lipo_classII,cp.E_density_supercap_classII],
                        [cp.P_density_li_ion_classII,cp.P_density_lipo_classII,cp.P_density_supercap_classII]
                        ]])

# Input for class

clas = int(input("Select 1 for classI and 2 for Class2"))

# Extracting P_peak and Getting P_avg

P_peak_arr = np.array([[cp.P_peak_ClassI,cp.P_peak_ClassII]])
P_peak = P_peak_arr[0,clas-1]
P_avg = avg_to_peak_ratio * P_peak # W

# Input for which combination
different = False

while not different:
    comb1 = int(input("""Select first power source:
    1: Li-ion
    2: Lipo
    3: Supercap
    """))

    comb2 = int(input("""Select second power source:
    1: Li-ion
    2: Lipo
    3: Supercap
    """))

    if comb1 != comb2:
        different=True

# Setting specific power and energy for the two selected sources

P_density_source1 = combination[clas-1,1,comb1-1]
E_density_source1 = combination[clas-1,0,comb1-1]

P_density_source2 = combination[clas-1,1,comb2-1]
E_density_source2 = combination[clas-1,0,comb2-1]

# Total energy needed

E_total = P_avg * (total_flight_time/3600)

percentage_first_source = np.arange(0,1,1/100)

m_energy_requirement = E_total / (percentage_first_source * E_density_source1 + (1-percentage_first_source) * E_density_source2)
m_power_requirement = P_peak / (percentage_first_source * P_density_source1 + (1-percentage_first_source) * P_density_source2)

total_mass = np.maximum(m_energy_requirement,m_power_requirement)

minimum_mass = np.min(total_mass)
minimum_mass_index = np.argmin(total_mass)
minimum_mass_fraction = percentage_first_source[minimum_mass_index]

# For plotting 

source_names = {1: 'Li-ion', 2: 'LiPo', 3: 'Supercap'}
name1 = source_names[comb1]
name2 = source_names[comb2]

fig,ax = plt.subplots(2,1,figsize=(9,5))

ax[0].plot(percentage_first_source, m_energy_requirement, label='Energy requirement', color='blue', linewidth=2)
ax[0].plot(percentage_first_source, m_power_requirement, label='Power requirement', color='orange', linewidth=2)
ax[1].plot(percentage_first_source, total_mass, label='Total mass', color='red', linestyle='--')

ax[0].set_xlabel(f'Fraction of {name1} (vs {name2})')
ax[0].set_ylabel('Mass (kg)')
ax[0].set_title(f'Energy & Power Requirements vs Battery Ratio — ({name1} / {name2})')

ax[1].set_xlabel(f'Fraction of {name1} (vs {name2})')
ax[1].set_ylabel('Mass (kg)')
ax[1].set_title(f'Total System Mass vs Battery Ratio —  ({name1} / {name2})')

ax[0].legend()
ax[1].legend()

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Minimum Mass: {minimum_mass:.3f} kg")
print(f"Optimal fraction of {name1}: {minimum_mass_fraction:.2f}  |  {name2}: {1 - minimum_mass_fraction:.2f}")