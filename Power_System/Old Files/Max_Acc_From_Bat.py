import numpy as np
import matplotlib.pyplot as plt

def peak_accel(etha,m,Cd,A,P_max,Ft_cap):
    """
    Inputs: 
        etha : Efficiency factor [-]
        m : drone mass [kg]
        Cd : drone Cd [-]
        A : drone area [m^2]
        P_max : Maximum Power delivered by the battery [W]
        Ft_cap : Maximum thrust provided by the propellors [N]

    Returns:
        Plot acceleration vs velocity
    """
    # Constants
    g = 9.80665 # [m/s^2]
    rho = 1.225 # [kg/m^3]
    v = np.arange(0,15,0.1) # Velocity [m/s]
    a = []
    # Calculations
    for velocity in v:
        P_mechanical = P_max * etha # Mechanical Power
        Fg = m * g  # Gravity
        Fd = 0.5 * rho * velocity**2 * A * Cd # Drag Force
        Ft = np.min([P_mechanical/velocity , Ft_cap])
        acc = ((Ft - Fd - Fg)/m)/g # Acceleration 
        a.append(acc)

    fig,ax = plt.subplots()
    ax.set_title('Max acceleration at different velocities')
    ax.set_ylabel('Max Acceleration')
    ax.set_xlabel('Velocity')
    ax.plot(v,a)
    plt.show()

peak_accel(0.6,0.1,1.28,0.005625,100,15)