import numpy as np
import matplotlib.pyplot as plt
import Comparison_Params as cp
import scienceplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error

plt.style.use(['science', 'no-latex', 'grid'])

def LiPo_sim (P_max=320,P_avg=128,t_flight=10):

    ''' Battery Parameters '''

    R_i = 0.03 # Internal resistance of battery [Ohms]
    num_cells = 4 # Number of cells in the batetry
    battery_capacity_Ah = 0.5 # Assumed capacity of battery [Ah]

    t_p_max = 4 # Time at max power [s]
    t_t_p_max_frac = 0 # Temporal location of the start of peak power as % of total fight

    initial_charge_soc = 0.65 # SOC to which we maximum charge
    charging_rate = 2 # C-Rate for charging

    ''' Additional Time calculations/parameters '''

        # Params
    dt = 0.0005  # Timestep size [s]

        # Calculations
    t = np.arange(0,t_flight+dt,dt) # Time vector 
    t_start_max_power = t_t_p_max_frac * t_flight # Temporal location of max power
    t_max_power = np.arange(t_start_max_power, t_start_max_power + t_p_max + dt, dt) # Time Vector of Max Power

    ''' Power and Battery Parameters and Calculations '''

        # Calculations
    P_max = P_max / 0.8 # Assuming that propellers draw 80% of the power 
    P_avg = P_avg / 0.8 # Assuming that propellers draw 80% of the power
    battery_capacity = battery_capacity_Ah * 3600 # Battery Capacity in Amp_secs for calculation
    initial_charge = battery_capacity * initial_charge_soc

    ''' LiPo Parameter Calculations '''

        # Loading Data
    battery_csv_path = r"C:\Users\spash\OneDrive\Desktop\Uni\Bachelor\Year 3\DSE\DSE Code\DSE-09-Drone-Platform-for-Automated-Pest-Control\Power_System\CSV files\0_Discharge_std_3.csv"
    battery_data = np.loadtxt(battery_csv_path, delimiter="\t")
    extracted_charge_linreg = battery_data[:,0]
    voltage_linreg = battery_data[:,1]
    capacity_linreg = extracted_charge_linreg[-1] #C apacity of tested battery at current cycle, used only to derive mathematical model [Ah]

        # Calculations
    state_of_charge_linreg = (capacity_linreg - extracted_charge_linreg) / capacity_linreg # State of charge vector as franction a.k.a % / 100

        # Model fitting
    polyfit = PolynomialFeatures(degree = 8)
    train_soc = polyfit.fit_transform(state_of_charge_linreg.reshape(-1,1))
    voltage_curve_model = LinearRegression()
    voltage_curve_model.fit(train_soc,voltage_linreg)

    '''Battery Performance Calculations'''

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

        tau_up = 0.2   # spool up faster
        tau_down = 0.6 # spool down slower

        t_ramp = 0.1

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

    
    recharge_time = (initial_charge_soc - soc_vec[-1]) * 3600 / charging_rate
    minutes_recharge = int(recharge_time // 60)
    seconds_recharge = int(recharge_time % 60)
    
    fig,ax = plt.subplots(2,2,figsize=(9,5))

    ax[0,0].plot(t, C_rate_vec, color='Steelblue', linewidth=2)
    ax[0,1].plot(t, soc_vec*100, color='Steelblue', linewidth=2)
    ax[1,0].plot(t, V_vec/4, color='Tomato', linewidth=2, label = 'Voltage Per Cell')
    ax[1,0].plot(t, V_vec, color='Steelblue', linewidth=2, label = 'Total Voltage')
    ax[1,0].plot(t, np.full_like(t, 2.75), color='red', linewidth=1, label = 'Cutoff Voltage', linestyle = '--')
    ax[1,1].plot(t, I_vec, color='Steelblue', linewidth=2)

    ax[0,0].set_xlabel('Time(s)')
    ax[0,0].set_ylabel('C_rate(-)')

    ax[0,1].set_xlabel('Time(s)')
    ax[0,1].set_ylabel('State Of Charge(%)')

    ax[1,0].set_xlabel('Time(s)')
    ax[1,0].set_ylabel('Voltage(V)')

    ax[1,1].set_xlabel('Time(s)')
    ax[1,1].set_ylabel('Current(A)')

    ax[1,0].legend()

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Recharge Time: ",f"{minutes_recharge} min {seconds_recharge} s")

LiPo_sim()

   




    
