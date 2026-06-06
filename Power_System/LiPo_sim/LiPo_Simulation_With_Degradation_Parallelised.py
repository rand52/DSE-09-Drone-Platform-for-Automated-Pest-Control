import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from Final_Model_Fit_Discharge import intercept,coef
from Final_Model_Fit_Internal_Resistance import intercept_res,coef_res
import time
from Resistance_degradation import r0_growth_factor

plt.style.use(['science', 'no-latex', 'grid'])
 
def smoothstep(x):
        return 1 / (1 + np.exp(-x))

def Vocv_poly(x):
    x2 = x*x
    x3 = x2*x
    x4 = x3*x

    return (
        intercept
        + coef[1]*x
        + coef[2]*x2
        + coef[3]*x3
        + coef[4]*x4
    )

def R_eff_poly(x):
     return intercept_res + coef_res[0] * x

def LiPo_sim (P_max_mot=326,P_avg_mot=130.4,t_flight=10):

    '''------------Battery Parameters To Enter For Each Battery----------------'''
    Cycle_to_display = 0 # Cycle to display in detail
    number_of_cycles_to_simulate = 500
    R_eff_initial_cycle_0 = 0.018 # Initial condition for internal resistance
    num_cells = 4 # Number of cells in the batetry
    nominal_battery_capacity_Ah = 0.32 # Assumed capacity of battery [Ah]
    avg_DoD = 0.3 # Average DoD that the battery will have as a fraction

    t_p_max = 3 # Time at max power [s]
    t_t_p_max_frac = 0 # Temporal location of the start of peak power as % of total fight

    initial_charge_soc = 0.8 # SOC to which we maximum charge
    charging_rate = 2 # C-Rate for charging

    '''------------Additional Battery Parameters----------------'''

    P_max = P_max_mot / 0.95 # Assuming that propellers draw 80% of the power 
    P_avg = P_avg_mot / 0.95 # Assuming that propellers draw 80% of the power

    '''---------------Additional Time calculations/parameters-------------------'''

        # Params
    dt = 0.01  # Timestep size [s]
    tau_up = 0.1   # spool up faster, how fast motors spool up
    tau_down = 0.6 # spool down slower, how fast motors spool down
    t_ramp = 0.05 # How fast demanded power raises

        # Calculations
    t = np.arange(0,t_flight+dt,dt) # Time vector
    t_start_max_power = t_t_p_max_frac * t_flight # Temporal location of max power
    t_max_power = np.arange(t_start_max_power, t_start_max_power + t_p_max + dt, dt) # Time Vector of Max Power

    '''-------------Degradation Calculations----------------'''
    # Calculations
    number_of_cycles = np.arange(0,number_of_cycles_to_simulate,1) # VECTOR with cycles
    number_of_equiv_cycles = number_of_cycles * avg_DoD * 2 # Equivalent actual full cycles the battery has gone through
    
   # Degradation Model
    degradation_frac = 1.085 - 0.07961 * np.exp(0.00563*number_of_equiv_cycles) # VECTOR for each cycle
    battery_capacity = nominal_battery_capacity_Ah * degradation_frac * 3600 # VECTOR Battery Capacity in Amp_secs for calculation
    battery_capacity_Ah = battery_capacity / 3600 # VECTOR
    initial_charge = battery_capacity * initial_charge_soc # VECTOR

    # Resistance degradation
    R_eff_initial = r0_growth_factor(number_of_equiv_cycles) * R_eff_initial_cycle_0
    '''------------Initializing Battery Performance Arrays----------------'''

    C_rate_arr = np.empty((number_of_cycles_to_simulate,len(t)))
    I_arr = np.empty((number_of_cycles_to_simulate,len(t)))
    V_arr = np.empty((number_of_cycles_to_simulate,len(t)))
    soc_arr = np.empty((number_of_cycles_to_simulate,len(t)))

    '''--------------Battery Performance Calculations------------------'''

    used_charge = 0.0 # Initialise used charge for calculation [As]
    P_actual = P_avg # Initialise P_actual delivered [W]
    

    # For loop for calculating C-rate, etc. of battery during flight
    for idx, time in enumerate(t):

        battery_soc = (initial_charge - used_charge) / battery_capacity
        Vocv_per_cell = Vocv_poly(battery_soc)
        R_eff = R_eff_initial * R_eff_poly(battery_soc)
        
        transition = smoothstep((time - t_max_power[0]) / t_ramp) - smoothstep((time - t_max_power[-1]) / t_ramp)

        P_target = P_avg + (P_max - P_avg) * transition
        
        if P_target > P_actual:
            tau_use = tau_up
        else:
            tau_use = tau_down
        
        P_actual += (P_target - P_actual) * (dt / tau_use)
        P_delivered = P_actual

        V_ocv_pack = num_cells * Vocv_per_cell
        R_pack = num_cells * R_eff

        #---------Quick Check--------------------
        max_power_possible = (V_ocv_pack ** 2) / (4*R_pack)
        fail_mask = P_target > max_power_possible
        if fail_mask.any():
            print("Demanded Power: ", P_target, "Max Power Possible: ", max_power_possible)
            idx = np.where(fail_mask)[0][0]  # first failing index (or choose logic)
            print(f"Power demand not feasible at t = {time}, cycle = {idx}")
            break
        #------------End------------------------
        disc = V_ocv_pack**2 - 4 * R_pack * P_delivered
        I_drawn  = (V_ocv_pack - np.sqrt(disc)) / (2*R_pack)
        V_delivered = V_ocv_pack - I_drawn*R_pack

        time_interval_charge_used = I_drawn * dt
        used_charge += time_interval_charge_used
        
        current_c_rate = I_drawn / (battery_capacity_Ah)

        C_rate_arr[:,idx] = current_c_rate
        soc_arr[:,idx] = battery_soc
        V_arr[:,idx] = V_delivered
        I_arr[:,idx] = I_drawn

    I_max_vec = np.max(I_arr,axis=1)
    soc_min_vec = np.min(soc_arr,axis=1)   
    V_min_vec = np.min(V_arr,axis=1)
    C_rate_max_vec = np.max(C_rate_arr,axis=1)
    DoD = initial_charge_soc - soc_arr[:,-1]
    recharge_time = (DoD) * 3600 / charging_rate

    return number_of_cycles,C_rate_max_vec,recharge_time, V_min_vec, soc_min_vec, I_max_vec

   

'''---------------Running Simulation------------------'''

if __name__ == "__main__":

    '''---------------Main Function------------------'''
    start = time.perf_counter()
    
    cycle_vec,C_rate_max_vec,_, V_min_vec, soc_min_vec, I_max_vec = LiPo_sim()
    
    end = time.perf_counter()

    print("Runtime:", end - start, "seconds")

    '''---------------Plotting------------------'''

    fig,ax = plt.subplots(2,2,figsize=(9,5))

    ax[0,0].plot(cycle_vec, C_rate_max_vec, color='Steelblue', linewidth=2)
    ax[1,0].plot(cycle_vec, V_min_vec, color='Red', linewidth=2)
    ax[0,1].plot(cycle_vec, soc_min_vec, color='Green', linewidth=2)
    ax[1,1].plot(cycle_vec, I_max_vec, color='Green', linewidth=2)

    ax[0,0].set_xlabel('Cycle Number(-)')
    ax[0,0].set_ylabel('Max C_rate(-)')

    ax[1,0].set_xlabel('Cycle Number(-)')
    ax[1,0].set_ylabel('Min Voltage')

    ax[0,1].set_xlabel('Cycle Number(-)')
    ax[0,1].set_ylabel('Min SOC')

    ax[1,1].set_xlabel('Cycle Number(-)')
    ax[1,1].set_ylabel('Maximum current')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


   




    
