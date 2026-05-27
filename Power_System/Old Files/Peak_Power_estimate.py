import numpy as np


def peak_power (etha,m,Cd,A,a,v):
    """
    Inputs: 
        etha : Efficiency factor [-]
        m : drone mass [kg]
        Cd : drone Cd [-]
        A : drone area [m^2]
        a : Acceleration wanted as x*g [x*g m/s^2]
        v : Velocity at which acc is achieved [m/s]

    Returns:
        Pmech : Estimate for mechanical power needed
        Pelec : Estimate for electrical power needed from battery
    """
    # Constants
    g = 9.80665 # [m/s^2]
    rho = 1.225 # [kg/m^3]


    # Calculations
    Fg = m * g  # Gravity
    Fa = m * a * g # ma from F=ma
    Fd = 0.5 * rho * v**2 * A * Cd # Drag Force

    Pmech = (Fg + Fa + Fd) * v  # Mechanical power needed to achieve acceleration
    Pelec = Pmech / etha # Electrical power needed from battery to achieve acceleration
    Ft = Pmech/v # Thrust needed from propellors
    return Pmech,Pelec,Ft

Pmechanical,Pelectrical,Ft = peak_power(0.6,0.1,1.28,0.005625,17.5,3.5)

print(Ft)

