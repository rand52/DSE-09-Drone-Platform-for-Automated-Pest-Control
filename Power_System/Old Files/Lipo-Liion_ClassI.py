import numpy as np
import matplotlib.pyplot as plt
import Comparison_Params as cp
#------------------------------------------------=-----

#flight parameters
total_flight_time = cp.total_flight_time_long #s
t_peak = cp.t_peak #s time peak power is required
P_peak = cp.P_peak_ClassI #W
P_avg = cp.avg_to_peak_ratio * P_peak # W

#battery and supercapcitor parameters
P_density_Li_ion = cp.P_density_li_ion_classI #W/kg
E_density_Li_ion = cp.E_density_li_ion_classI # Wh/kg

P_density_Lipo = cp.P_density_lipo_classI # W/kg
E_density_Lipo = cp.E_density_lipo_classI # Wh/kg


E_total = P_avg * (total_flight_time/3600)

m_energy_req_list = np.zeros((101))
m_maxpower_req_list = np.zeros((101))
# minimum battery fraction
"""
P_bat = P_avg
P_supercap = P_peak - P_avg

m_battery_min = P_avg / P_density_Li_ion
m_supercap_min = P_supercap / P_density_s
m_frac_min = m_battery_min / (m_battery_min+m_supercap_min)
"""
battery_ratio_list = np.arange(0,1,0.01)
m_energy_req_list = E_total / ( battery_ratio_list * E_density_Li_ion + (1-battery_ratio_list)*E_density_Lipo)
m_maxpower_req_list = P_peak / (battery_ratio_list * P_density_Li_ion + (1- battery_ratio_list) *P_density_Lipo)


difference = np.abs(m_energy_req_list-m_maxpower_req_list)
optimal_ratio_index = np.argmin(difference)
print(battery_ratio_list[optimal_ratio_index])
print(m_maxpower_req_list[optimal_ratio_index]) # weight due to energy and power constraint approximately the same

plt.figure(figsize=(9, 5))
plt.plot(battery_ratio_list, m_energy_req_list, label='Energy requirement', color='blue', linewidth=2)
plt.plot(battery_ratio_list, m_maxpower_req_list, label='Power requirement', color='orange', linewidth=2, linestyle='--')

plt.xlabel('Battery ratio (Li-ion fraction)')
plt.ylabel('Mass Li-ion Lipo battery')
plt.title('Energy and power requirements vs battery ratio (t=200s)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------------
# to visualize different flight times, make for example 10 lines with legend that show the requirements for different flight times