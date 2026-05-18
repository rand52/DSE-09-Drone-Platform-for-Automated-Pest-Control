import numpy as np
import matplotlib.pyplot as plt

# Material properties
E = 70e9                # [Pa] Young's modulus of aluminum
sigma_yield = 200e6     # [Pa] yield strength of aluminum

# Drone definition
L = 200 / 1000          # [m]
phi = 60                # [deg]

t = 50 / 1000            # [m] arm height in y-direction
w = 20 / 1000           # [m] arm width in x-direction
a = 200 / 1000          # [m] drone box side a
b = 150 / 1000          # [m] drone box side b

# Force definition
F = -100                # [N]
l = 50 / 1000           # [m]

# Discretization
disc_x = 50            # mesh point in x direction
disc_z = 50            # mesh point in y direction



def drone_geometry(t, w, a = None, b= None, L=None, phi = None):
    """
    Define X-frame geometry using either:
        - a + b (size of box in [mm] which drone frame fits)
        - L + phi (drone arm length [mm] + drone arm angle [rad] )
    """

    if a is None and b is None:
        a = 2 * L * np.cos(phi/2)
        b = 2 * L * np.sin(phi/2)
    else:
        L = np.sqrt(a**2 + b**2) / 2
        phi = 2 * np.asin(b/2 / L)

    A = w * t                   # [m^2], cross-sectional area of the arm
    Ixx = (w * t ** 3) / 12     # [m^4], second moment of area about beam x-axis
    Izz = (t * w ** 3) / 12     # [m^4], second moment of area about beam z-axis

    return {
        "a": a,         # [m] drone box side a
        "b": b,         # [m] drone box side b
        "phi": phi,     # [rad] angle between arms
        "L": L,         # [m] arm length
        "w": w,         # [m] arm width in x-direction
        "t": t,         # [m] arm height in y-direction
        "A": A,         # [m^2], cross-section area
        "Ixx": Ixx,     # [m^4] moment of area about beam x-axis
        "Izz": Izz      # [m^4] moment of area about beam z-axis
    }

geometry = drone_geometry(t=t,w=w,L=L, phi=phi)
#print(geometry)

def internal_loads(l, F, psi, chi):
    """Returns the forces at position l[mm] from the tip of the arm"""

    if l < 0 or l > geometry["L"]:
        raise ValueError(f'Value outside arm length (0,{geometry["L"]})')

    Fz = F * np.sin(chi)
    Fx = F * np.sin(psi) * np.cos(chi)
    Fy = F * np.cos(psi) * np.cos(chi)
    Mx = - Fz * l   # negative sign, coz Mx + Fz*l =0
    Mz = - Fx * l   # negative sign, coz Mz + Fx*l =0
    # print(Fz,Fx,Fy,Mx,Mz)
    # print("Fx:", Fx)

    return {
        "Fx": Fx,       # [N] force in x-direction
        "Fy": Fy,       # [N] force in y-direction
        "Fz": Fz,       # [N] force in z-direction
        "Mx": Mx,       # [Nm] moment about x-axis
        "Mz": Mz        # [Nm] moment about z-axis
    }

def find_max_stress(l):

    if l < 0 or l > geometry["L"]:
        raise ValueError(f'Value outside arm length (0,{geometry["L"]})')

    # Keep track of the angle combination that gives the maximum stress
    max_global_stress_von_misses = 0
    max_global_sigma_yy = []
    max_global_sigma_misses = []
    psi_global_max_stress = 0
    chi_global_mac_stress = 0


    # Coordinate grid of the cross-section
    x = np.linspace(-geometry["w"]/2, geometry["w"]/2, disc_x)      # width w along x-coord
    z = np.linspace(-geometry["t"]/2, geometry["t"]/2, disc_z)      # height t along y-coord

    X, Z = np.meshgrid(x, z)
    # Create mesh like
    # x = [1,0,-1]
    # z = [-a,0,a]
    # X = [[1,0,-1]    Z = [[-a,0,a]]
    #      [1,0,-1]         [-a,0,a]
    #      [1,0,-1]]        [-a,0,a]]


    # All available impact angles for this phi
    chi_arr = np.radians(np.arange(-90, 90))
    psi_arr = np.radians(np.arange(np.round(-90-phi/2), np.round(180-phi/2)))
    print(np.degrees(chi_arr[0]),np.degrees(chi_arr[-1]))
    print(np.degrees(psi_arr[0]),np.degrees(psi_arr[-1]))

    for chi in chi_arr:
        for psi in psi_arr:
            print(chi, psi)
            loads = internal_loads(l=l, F=F, psi=psi, chi=chi)

            # Normal stress superposition
            # sigma= Fz/A + Mx*z/Ixx + Mz*x/Izz
            sigma_axial = loads["Fy"] / geometry["A"]
            sigma_bending_on_x = loads["Mx"] * Z / geometry["Ixx"]
            sigma_bending_on_z = loads["Mz"] * X / geometry["Izz"]

            sigma_yy = sigma_axial + sigma_bending_on_x + sigma_bending_on_z

            # Shear stresses
            tau_yx = (2 * loads["Fz"] / geometry["Ixx"]) * (geometry["t"]**2 / 4 - Z**2)
            tau_yz = (2 * loads["Fx"] / geometry["Izz"]) * (geometry["w"]**2 / 4 - X**2)

            # von Misses stress
            sigma_misses = np.sqrt(sigma_yy**2 + 3*tau_yx**2+ 3*tau_yz**2)
            max_sigma_misses = np.max(sigma_misses)
            #print("Psi=", np.round(np.degrees(psi)), "[deg]; Chi=", np.round(np.degrees(chi)), "[deg]; Fx=", np.round(loads["Fx"], 2), "[N]; Fy=", np.round(loads["Fy"], 2), "[N]; Fz=", np.round(loads["Fz"], 2), "[N]; sigma_misses_max=", np.round(max_sigma_misses / 1e6, 2), "[Mpa]")

            # Max stress
            if max_sigma_misses > max_global_stress_von_misses:
                max_global_stress_von_misses = max_sigma_misses
                max_global_sigma_yy = sigma_yy
                max_global_sigma_misses = sigma_misses
                psi_global_max_stress = psi
                chi_global_max_stress = chi
                Fx_max = loads["Fx"]
                Fy_max = loads["Fy"]

    print(f'Max normal stress of, {np.round(max_global_stress_von_misses / 1e6, 2)}, MPa, at psi = {np.round(np.degrees(psi_global_max_stress), 2)} deg, chi = {np.round(np.degrees(chi_global_max_stress), 2)} deg,')

    ### Plot the global combination of Psi and Chi angles that gives the most stress: ###

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # Plot the normal stresses
    # cmap='bwr_r' gives +=blue (tension) and -=red (compression)
    color_bar_1_ref = ax1.pcolormesh(X * 1000, Z * 1000, max_global_sigma_yy / 1e6, cmap='bwr_r', vmin=-max_global_stress_von_misses / 1e6, vmax=max_global_stress_von_misses / 1e6)

    # Invert x-axis to be consistent with our coord definition
    ax1.invert_xaxis()

    # Plot axis and legend
    fig.colorbar(color_bar_1_ref, label='Stress [MPa]', ax=ax1)
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('z [mm]')
    ax1.set_title(f'Normal_yy at l={l * 1000} mm')
    ax1.axis('equal')

    # Plot the von Misses stress
    color_bar_2_ref = ax2.pcolormesh(X * 1000, Z * 1000, max_global_sigma_misses / 1e6, cmap='bwr_r', vmin=-max_global_stress_von_misses / 1e6, vmax=max_global_stress_von_misses / 1e6)
    ax2.invert_xaxis()
    # Add the max stress
    fig.colorbar(color_bar_2_ref, label='Stress [MPa]', ax=ax2)
    ax2.set_xlabel('x [mm]')
    ax2.set_ylabel('z [mm]')
    ax2.set_title(f'von Misses stress at l={l * 1000} mm')
    ax2.axis('equal')

    plt.show()


l = 50 / 1000      # [m] from tip of arm
find_max_stress(l)
