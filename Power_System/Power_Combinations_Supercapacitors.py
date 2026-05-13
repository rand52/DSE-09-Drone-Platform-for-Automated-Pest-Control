import numpy as np
import matplotlib.pyplot as plt
import Comparison_Params as cp

# General Flight parameters

t_total = cp.total_flight_time_short #s
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

clas = int(input("Select 1 for Class_1 and 2 for Class_2"))

# Extracting P_peak and Getting P_avg

P_peak_arr = np.array([[cp.P_peak_ClassI,cp.P_peak_ClassII]])
P_peak = P_peak_arr[0,clas-1]
P_avg = avg_to_peak_ratio * P_peak # W

# Input for which combination

comb1 = int(input("""Select power source to combine with supercapacitor:
1: Li-ion
2: Lipo
"""))

source_names = ['Li-ion','LiPo']
# Setting specific power and energy for the two selected sources

P_density_supercap = combination[clas-1,1,2]
E_density_supercap = combination[clas-1,0,2]

P_density_source2 = combination[clas-1,1,comb1-1]
E_density_source2 = combination[clas-1,0,comb1-1]

# Mass of supercapacitor

mass_for_power_supercap = (P_peak - P_avg) / P_density_supercap
E_supercap = (P_peak-P_avg) * t_peak/3600
mass_for_energy_supercap = E_supercap / E_density_supercap

mass_supercap = max(mass_for_power_supercap,mass_for_energy_supercap)

# Mass of other power source

mass_for_power_source2 = P_avg / P_density_source2
E_source2 = P_avg * t_total/3600
mass_for_energy_source2 = E_source2 / E_density_source2

mass_soucre2 = max(mass_for_power_source2,mass_for_energy_source2)

total_mass = mass_supercap + mass_soucre2


print(f"\n------Results-------")
print(f"\nSupercapacitor mass is: {round(mass_supercap*1000,2)}g")
print(f"\n{source_names[comb1-1]} mass is: {round(mass_soucre2*1000,2)}g")
print(f"\nTotal mass is: {round(total_mass*1000,2)}g")

