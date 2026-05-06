import numpy as np

# geometry definition: a=width (x global), b=height (y global), forms the xy-plane, z is out of plane.
a = 75e-3 # m, True-X
b = 75e-3 # m, True-X
phi_list = np.linspace(0,np.pi/2,90) # all angles of the arm (in-plane as well)
phi = phi_list[len(phi_list)//2] # m, angle of the arm with respect to x-axis, so 45 degrees for True-X

# Beam definition
t = 3e-3 # m, thickness of the arm (for stress calculations)
w = 20e-3 # m, width of the arm (for stress calculations)

L = np.sqrt((a/2)**2 + (b/2)**2) # m, length of each arm'
A = w * t # m^2, cross-sectional area of the arm
# Rectangular beam:
I_yy = (w * t**3) / 12 # m^4, second moment of area about beam y-axis, so not global
I_zz = (t * w**3) / 12 # m^4, second moment of area about beam z-axis, so not global
J = I_yy + I_zz # m^4, polar moment of inertia

# Material properties
E = 70e9 # Pa, Young's modulus of aluminum
sigma_yield = 200e6 # Pa, yield strength of aluminum

#Force definition
psi = np.linspace(-np.pi/2 - phi,np.pi-phi,270) # in-plane angle of applied force for every degree in radians
chi = np.linspace(-np.pi/4,np.pi/4,180) # out-of-plane angle of applied force for every degree in radians
F_mag = 0.5 # N

def force_components(F_mag, psi, chi):
    # Every combination of every angle decomposed into parallel and perpendicular components
    F_par  = F_mag * np.cos(psi)[:, np.newaxis] * np.cos(chi)  
    F_perp = F_mag * np.sin(psi)[:, np.newaxis] * np.cos(chi)  
    F_z    = F_mag * np.sin(chi)                                
    
    return F_par, F_perp, F_z

# In-plane bending around center of beam:
def stress(F_par, F_perp, F_z):
    # Bending from perpendicular component and out of plane component can be superimposed.
    s = np.linspace(0, L, 100) # m, position along the beam from the fixed end to the tip
    M_z = F_perp[:, :, np.newaxis]  # Bending moment at the center of the beam
    c1 = w / 2       # Distance from neutral axis to outer fiber
    M_y = F_z[np.newaxis, :, np.newaxis]     # Bending moment from out of plane force
    c2 = t / 2       # Distance from neutral axis to outer fiber
    sig_bend = M_z * c1 / I_zz + M_y * c2 / I_yy #computes bending stress at the outer fiber from both components, superimposed

    # Normal
    sig_norm = F_par / A

    return sig_bend, sig_norm