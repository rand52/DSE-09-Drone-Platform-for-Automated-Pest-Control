import numpy as np
import matplotlib.pyplot as plt

# Material properties
E = 70e9                # [Pa] Young's modulus of aluminum
sigma_yield = 200e6     # [Pa] yield strength of aluminum

# Force definition
F = 100                 # [N]
psi = np.radians(0)    # [rad]
chi = np.radians(0)     # [rad]

# Discretization
disc_x = 100            # mesh point in x direction
disc_z = 100            # mesh point in y direction

t = 5 / 1000            # [m] arm height in y-direction
w = 20 / 1000           # [m] arm width in x-direction
a = 200 / 1000          # [m] drone box side a
b = 150 / 1000          # [m] drone box side b


def drone_geometry(t, w, a = None, b= None, L=None, phi = None):
    """
    Define X-frame geometry using either:
        - a + b (size of box in [mm] which drone frame fits)
        - L + phi (drone arm length [mm] + drone arm angle [deg] )
    """

    if a is None and b is None:
        a = 2 * L * np.cos(phi/2)
        b = 2 * L * np.sin(phi/2)
    else:
        L = np.sqrt(a**2 + b**2) / 2
        phi = 2 * np.atan2(b/2, a/2)

    A = w * t                   # [m^2], cross-sectional area of the arm
    Ixx = (w * t ** 3) / 12     # [m^4], second moment of area about beam x-axis
    Izz = (t * w ** 3) / 12     # [m^4], second moment of area about beam z-axis

    return {
        "a": a,         # [m] drone box side a
        "b": b,         # [m] drone box side b
        "L": L,         # [m] arm length
        "phi": phi,     # [rad] angle between arms
        "w": w,         # [m] arm width in x-direction
        "t": t,         # [m] arm height in y-direction
        "A": A,         # [m^2], cross-section area
        "Ixx": Ixx,     # [m^4] moment of area about beam x-axis
        "Izz": Izz      # [m^4] moment of area about beam z-axis
    }

geometry = drone_geometry(t=t,w=w,a=a,b=b)
#print(geometry)

def internal_loads(l):
    """Returns the forces at position l[mm] from the tip of the arm"""

    if l < 0 or l > geometry["L"]:
        raise ValueError(f'Value outside arm length (0,{geometry["L"]})')

    Fz = F * np.sin(chi)
    Fx = F * np.cos(chi) * np.sin(psi)
    Fy = F * np.cos(chi) * np.cos(psi)
    Mx = Fz * l
    Mz = Fx * l
    #print(Fz,Fx,Fy,Mx,Mz)

    return {
        "l": l,         # [m] distance from arm tip
        "Fx": Fx,       # [N] force in x-direction
        "Fy": Fy,       # [N] force in y-direction
        "Fz": Fz,       # [N] force in z-direction
        "Mx": Mx,       # [Nm] moment about x-axis
        "Mz": Mz        # [Nm] moment about z-axis
    }

def plot_stresses(l):

# sigma= Fz/A + Mx*z/Ixx + Mz*x/Izz

    if l < 0 or l > geometry["L"]:
        raise ValueError(f'Value outside arm length (0,{geometry["L"]})')

    loads = internal_loads(l)

    w = geometry["w"]
    t = geometry["t"]
    Ixx = geometry["Ixx"]
    Izz = geometry["Izz"]
    A = geometry["A"]

    # Coordinate grid
    x = np.linspace(-w/2, w/2, disc_x)      # width w along x-coord
    z = np.linspace(-t/2, t/2, disc_z)      # height t along y-coord

    X, Z = np.meshgrid(x, z)
    # Create mesh like
    # x = [1,2,3]
    # z = [a,b,c]
    # X = [[1,2,3]    Z = [[a,a,a]]
    #      [1,2,3]         [b,b,b]
    #      [1,2,3]]        [c,c,c]]

    # Stress superposition
    # sigma= Fz/A + Mx*z/Ixx + Mz*x/Izz
    sigma_axial = loads["Fy"] / A
    sigma_x_bending = loads["Mx"] * Z / Ixx
    sigma_z_bending = loads["Mz"] * X / Izz
    sigma = sigma_axial + sigma_x_bending + sigma_z_bending

    # Max stress
    sigma_abs_max = np.max(np.abs(sigma / 1e6)) # [Mpa]

    # Plot
    plt.figure()
    # Grid now back in mm(*1000), stress in Mpa (/10^6)
    # cmap='bwr_r' gives +=blue (tension) and -=red (compression)
    plt.pcolormesh(X * 1000, Z * 1000, sigma / 1e6, cmap='bwr_r', vmin=-sigma_abs_max, vmax=sigma_abs_max)

    # Invert x-axis to be consistent with our coord definition
    plt.gca().invert_xaxis()

    # Add the max stress
    plt.text(10, 8.5, f'σ_max = {sigma_abs_max} MPa', fontsize=10)

    # Plot axis and legend
    plt.colorbar(label='Stress [MPa]')
    plt.xlabel('x [mm]')
    plt.ylabel('z [mm]')
    plt.title(f'Stress Distribution at l={l * 1000} mm')
    plt.axis('equal')
    plt.show()


    # return stresses for good measure
    return {
        "x": x,             # [m] x-coord of stress mesh point
        "z": z,             # [m] z-coord of stress mesh point
        "sigma": sigma      # [Pa] stress at mesh points [x,y]
    }

l = 50 / 1000      # [m] from tip of arm
plot_stresses(l)
