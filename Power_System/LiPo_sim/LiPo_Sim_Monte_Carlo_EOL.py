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

def LiPo_sim (P_max_mot=320,P_avg_mot=130,t_flight=10):
    '''--------Monte Carlo-----------'''
    Number_Monte_carlo_runs = 100
    fail_cycles = np.full(Number_Monte_carlo_runs,np.nan)
    V_min = []
    Monte_Carlo_to_display = Number_Monte_carlo_runs/2 # Cycle to display in detail
    '''------------Battery Parameters To Enter For Each Battery----------------'''
    number_of_cycles_to_simulate = 1500
    R_eff_initial_cycle_0 = 0.015 # Initial condition for internal resistance
    num_cells = 4 # Number of cells in the batetry
    nominal_battery_capacity_Ah = 0.32 # Assumed capacity of battery [Ah]
    avg_DoD = 0.17 # Average DoD that the battery will have as a fraction
    max_c_rate = 95 # Max C rate of battery

    t_p_max = 3 # Time at max power [s]
    t_t_p_max_frac = 0 # Temporal location of the start of peak power as % of total fight

    initial_charge_soc = 0.75 # SOC to which we maximum charge
    charging_rate = 2 # C-Rate for charging

    '''------------Additional Battery Parameters----------------'''

    P_max = P_max_mot / 0.95 # Assuming that propellers draw 95% of the power 
    P_avg = P_avg_mot / 0.80 # Assuming that propellers draw 95% of the power

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

    for run in range(Number_Monte_carlo_runs):

        '''------------Monte Carlo Simulation---------------'''
        mean = 0.003225333
        c = np.random.normal(mean,0.1*mean)
        
        
        '''-------------Degradation Calculations----------------'''
        # Calculations
        number_of_cycles = np.arange(0,number_of_cycles_to_simulate,1) # VECTOR with cycles
        number_of_equiv_cycles = number_of_cycles * avg_DoD * 1.5 # Equivalent actual full cycles the battery has gone through
        
    # Degradation Model
        degradation_frac = 1.185202 - 0.203385 * np.exp(c*number_of_equiv_cycles) # VECTOR for each cycle
        battery_capacity = nominal_battery_capacity_Ah * degradation_frac * 3600 # VECTOR Battery Capacity in Amp_secs for calculation
        battery_capacity_Ah = battery_capacity / 3600 # VECTOR
        initial_charge = battery_capacity * initial_charge_soc # VECTOR
        
        # Resistance degradation
        R_eff_initial = r0_growth_factor(number_of_equiv_cycles) * R_eff_initial_cycle_0 # Aging
        '''------------Initializing Battery Performance Arrays----------------'''

        C_rate_arr = np.zeros((number_of_cycles_to_simulate,len(t)))
        I_arr = np.zeros((number_of_cycles_to_simulate,len(t)))
        V_arr = np.empty((number_of_cycles_to_simulate,len(t)))
        soc_arr = np.zeros((number_of_cycles_to_simulate,len(t)))
        V_ocv_arr = np.empty((number_of_cycles_to_simulate,len(t)))
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
            fail_mask = P_delivered > max_power_possible
            if fail_mask.any():
                failing_cycles = np.where(fail_mask)[0]
                idx_fail = failing_cycles.min()
                print(f"Power demand not feasible at t = {time}, cycle = {idx_fail}")
                if idx_fail <= fail_cycles[run] or np.isnan(fail_cycles[run]):
                    fail_cycles[run] = idx_fail
            #------------End------------------------
            disc = V_ocv_pack**2 - 4 * R_pack * P_delivered
            disc = np.maximum(disc, 0.0)
            I_drawn  = (V_ocv_pack - np.sqrt(disc)) / (2*R_pack)
            V_delivered = V_ocv_pack - I_drawn*R_pack

            time_interval_charge_used = I_drawn * dt
            used_charge += time_interval_charge_used
            
            current_c_rate = I_drawn / (battery_capacity_Ah)
            
            C_rate_arr[:,idx] = current_c_rate
            soc_arr[:,idx] = battery_soc
            V_arr[:,idx] = V_delivered
            I_arr[:,idx] = I_drawn
            V_ocv_arr[:,idx] = V_ocv_pack
        V_min.append(np.min(V_arr))
        '''--------Fail Cycle detection---------'''
        fail_mask_2 = V_ocv_arr < 2.75 * 4
        fail_per_cycle = fail_mask_2.any(axis=1)
        fail_cycle_indices = np.where(fail_per_cycle)[0]

        if len(fail_cycle_indices) > 0:
            fail_cycle = fail_cycle_indices[0]
        else:
            fail_cycle = number_of_cycles_to_simulate - 1
        
        if (np.isnan(fail_cycles[run]) or fail_cycle < fail_cycles[run]):
            fail_cycles[run] = fail_cycle
                
        if Monte_Carlo_to_display == run:
            plot_idx = max(fail_cycle - 1, 0)
            I_plotting = I_arr[plot_idx,:]
            Soc_plotting = soc_arr[plot_idx,:]
            V_plotting = V_arr[plot_idx,:]
            C_rate_plotting = C_rate_arr[plot_idx,:]
            DoD = initial_charge_soc - soc_arr[plot_idx,-1]
            recharge_time = (DoD) * 3600 / charging_rate
    V_min_min = np.min(V_min)
    print(fail_cycles)
    return I_plotting, Soc_plotting, V_plotting, C_rate_plotting, recharge_time, fail_cycles, t, V_min_min

   

'''---------------Running Simulation------------------'''

if __name__ == "__main__":

    '''---------------Main Function------------------'''
    start = time.perf_counter()
    
    I_plot, Soc_plot, V_plot, C_rate_plot, recharge_time, fail_cycles, time_vec, Minimum_Voltage = LiPo_sim()
    
    end = time.perf_counter()

    print("Runtime:", end - start, "seconds")

    print("Absolute Minimum Voltage [V]:", Minimum_Voltage)

    mean_cycles = np.average(fail_cycles)
    std_cycles = np.std(fail_cycles)

    var_mean = (std_cycles ** 2) / len(fail_cycles)
    std_mean = np.sqrt(var_mean)


    print("Cycle Distribution N(", mean_cycles,",",std_cycles ** 2,")")
    print("Standard Error of the mean: ", std_mean)

    lower_lim_95_percent = mean_cycles - 2.57 * std_cycles

    print(round(0.95 ** 4 *100,1),"% Chance of more than: ", lower_lim_95_percent, "cycles")

    recharge_mins = recharge_time // 60
    recharge_secs = recharge_time % 60
    print("Recharge Time: ", int(recharge_mins),"minutes and ", int(recharge_secs), "seconds")

    '''---------------Plotting------------------'''

    # Figure 1: Voltage
    plt.figure()
    plt.plot(time_vec, V_plot, color='Steelblue', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Figure 2: Current
    plt.figure()
    plt.plot(time_vec, I_plot, color='Steelblue', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Current (A)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Figure 3: C rate
    plt.figure()
    plt.plot(time_vec, C_rate_plot, color='Steelblue', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('C rate (1/h)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Figure 4: State of Charge
    plt.figure()
    plt.plot(time_vec, Soc_plot, color='Steelblue', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('State of Charge (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


   




    
