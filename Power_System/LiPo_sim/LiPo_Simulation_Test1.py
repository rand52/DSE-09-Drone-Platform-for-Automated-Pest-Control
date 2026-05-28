import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error
from Final_Model_Fit_Discharge import voltage_curve_model,polyfit

plt.style.use(['science', 'no-latex', 'grid'])

def LiPo_sim (P_max_mot=320,P_avg_mot=128,t_flight=10):

    max_cycles = 200

    cycle_vec = np.zeros(max_cycles)
    C_rate_max_vec = np.zeros(max_cycles)

    '''------------Battery Parameters----------------'''
    for idc, cycle_number in enumerate(np.arange(0,max_cycles,1)):

        cycle_vec[idc] = cycle_number

        R_i = 0.03 # Internal resistance of battery [Ohms]
        num_cells = 4 # Number of cells in the batetry
        nominal_battery_capacity_Ah = 0.5 # Assumed capacity of battery [Ah]
        avg_DoD = 0.3 # Average DoD that the battery will have as a fraction
    
        t_p_max = 4 # Time at max power [s]
        t_t_p_max_frac = 0 # Temporal location of the start of peak power as % of total fight

        initial_charge_soc = 0.65 # SOC to which we maximum charge
        charging_rate = 2 # C-Rate for charging

        '''---------------Additional Time calculations/parameters-------------------'''

            # Params
        dt = 0.01  # Timestep size [s]

            # Calculations
        t = np.arange(0,t_flight+dt,dt) # Time vector 
        t_start_max_power = t_t_p_max_frac * t_flight # Temporal location of max power
        t_max_power = np.arange(t_start_max_power, t_start_max_power + t_p_max + dt, dt) # Time Vector of Max Power

        '''-------------Power and Battery Parameters and Calculations----------------'''
            # Calculations
        P_max = P_max_mot / 0.8 # Assuming that propellers draw 80% of the power 
        P_avg = P_avg_mot / 0.8 # Assuming that propellers draw 80% of the power
        cycle_number_actual = cycle_number * avg_DoD # Equivalent actual full cycles the battery has gone through

                # Degradation Model
        degradation_frac = 1.085 - 0.07961 * np.exp(0.00563*cycle_number_actual)
        battery_capacity = nominal_battery_capacity_Ah * degradation_frac * 3600 # Battery Capacity in Amp_secs for calculation
        battery_capacity_Ah = battery_capacity / 3600
        initial_charge = battery_capacity * initial_charge_soc

        '''-------------LiPo Parameter Calculations---------------------'''


        '''--------------Battery Performance Calculations------------------'''

        used_charge = 0 # Initialise used charge for calculation [As]
        P_actual = P_avg # Initialise P_actual delivered [W]
        N = len(t) # Next lines, initialising vectors to be plotted afterwards
        C_rate_vec = np.zeros(N)
        I_vec = np.zeros(N)
        V_vec = np.zeros(N)
        soc_vec = np.zeros(N)

        # For loop for calculating C-rate, etc. of battery during flight
        for idx, time in enumerate(t):

            battery_soc = (initial_charge - used_charge) / battery_capacity
            Vocv_per_cell = voltage_curve_model.predict(polyfit.transform([[battery_soc]]))[0]

            tau_up = 0.1   # spool up faster
            tau_down = 0.6 # spool down slower

            t_ramp = 0.05

            def smoothstep(x):
                return 1 / (1 + np.exp(-x))

            transition = smoothstep((time - t_max_power[0]) / t_ramp) - smoothstep((time - t_max_power[-1]) / t_ramp)

            P_target = P_avg + (P_max - P_avg) * transition

            if P_target > P_actual:
                tau_use = tau_up
            else:
                tau_use = tau_down
            
            P_actual += (P_target - P_actual) * (dt / tau_use)
            P_delivered = P_actual

            V_pack = num_cells * Vocv_per_cell
            R_pack = num_cells * R_i

            disc = V_pack**2 - 4 * R_pack * P_delivered

            I_drawn  = (V_pack - np.sqrt(disc)) / (2*R_pack)

            V_delivered = V_pack - I_drawn*R_pack

            time_interval_charge_used = I_drawn * dt
            used_charge += time_interval_charge_used
            
            current_c_rate = I_drawn / (battery_capacity_Ah)

            C_rate_vec[idx] = current_c_rate
            soc_vec[idx] = battery_soc
            V_vec[idx] = V_delivered
            I_vec[idx] = I_drawn

        C_rate_max_vec[idc] = np.max(C_rate_vec)
        DoD = initial_charge_soc - soc_vec[-1]
        recharge_time = (DoD) * 3600 / charging_rate
        minutes_recharge = int(recharge_time // 60)
        seconds_recharge = int(recharge_time % 60)

        '''---------------Plotting------------------'''

    fig,ax = plt.subplots(figsize=(9,5))

    ax.plot(cycle_vec, C_rate_max_vec, color='Steelblue', linewidth=2)


    ax.set_xlabel('Cycle Number(-)')
    ax.set_ylabel('Max C_rate(-)')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

   

'''---------------Running Simulation------------------'''

if __name__ == "__main__":
    LiPo_sim()

   




    
