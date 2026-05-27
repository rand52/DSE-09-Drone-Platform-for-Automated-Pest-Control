import numpy as np
import matplotlib.pyplot as plt
import Comparison_Params as cp
import scienceplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error

def LiPo_sim (P_max=100,P_avg=40,motor_eff=0.4,t_flight=10):

    ''' Battery Parameters '''

    R_i = 0.03 # Internal resistance of battery [Ohms]
    Num_cells = 4 # Number of cells in the batetry
    Battery_capacity = 1 # Assumed capacity of battery [Ah]

    ''' Additional Time calculations/parameters '''

        # Params
    dt = 0.01  # Timestep size [s]
    t_p_max = 2 # Time at max power [s]
    t_t_p_max_frac = 0.2 # Temporal location of the start of peak power as % of total fight

        # Calculations
    t = np.arange(0,t_flight+dt,dt) # Time vector 
    t_start_max_power = t_t_p_max_frac * t_flight # Temporal location of max power
    t_max_power = np.arange(t_start_max_power, t_start_max_power + t_p_max + dt, dt) # Time Vector of Max Power

    ''' Power Parameters and Calculations '''

        # Calculations
    P_max = P_max / 0.8 # Assuming that propellers draw 80% of the power 
    P_avg = P_avg / 0.8 # Assuming that propellers draw 80% of the power

    ''' LiPo Parameter Calculations '''

        # Loading Data
    battery_csv_path = r"C:\Users\spash\OneDrive\Desktop\Uni\Bachelor\Year 3\DSE\DSE Code\DSE-09-Drone-Platform-for-Automated-Pest-Control\Power_System\CSV files\0_Discharge_std.csv"
    battery_data = np.loadtxt(battery_csv_path, delimiter="\t")
    extracted_charge = battery_data[:,0]
    voltage = battery_data[:,1]
    capacity = extracted_charge[-1] #C apacity of tested battery at current cycle, used only to derive mathematical model [Ah]

        # Calculations
    state_of_charge = (capacity - extracted_charge) / capacity # State of charge vector as franction a.k.a % / 100

        # Model fitting
    poly9 = PolynomialFeatures(degree=9)
    train_soc = poly9.fit_transform(state_of_charge.reshape(-1,1))
    voltage_curve_model = LinearRegression()
    voltage_curve_model.fit(train_soc,voltage)

    '''Battery Performance Calculations'''

    
    for time in t:

        if time <= t_max_power[0] or time >= t_max_power[-1]:
            P_delivered = P_avg
        else: 
            P_delivered = P_max


LiPo_sim()

   




    
