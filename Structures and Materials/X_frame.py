import numpy as np
import matplotlib.pyplot as plt

# Material properties
E = 70e9                # [Pa] Young's modulus of aluminum
sigma_yield = 200e6     # [Pa] yield strength of aluminum

# Force definition
F = 100                 # [N]
psi = np.radians(np.linspace(-180, 90, 271))  # [rad], 270,
chi = np.radians(np.linspace(-90, 90, 181))    # [rad]

# Discretization
disc_x = 50            # mesh point in x direction
disc_z = 50            # mesh point in y direction

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
        phi = 2 * np.asin(b/2 / L)

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

    psi_eff = psi - geometry["phi"] / 2  # Now it takes the arm angle into account
    Fz = F * np.sin(chi)
    Fx = F * np.sin(psi_eff)[:, np.newaxis] * np.cos(chi) # Creates a new column for every chi angle, so that we have every combination of psi and chi. AI came up with this function when i asked for a more elegant way instead of using a bunch of for loops
    Fy = F * np.cos(psi_eff)[:, np.newaxis] * np.cos(chi) # See comment above
    Mx = - Fz * l   # negative sign, coz Mx + Fz*l =0
    Mz = - Fx * l   # negative sign, coz Mz + Fx*l =0
    #print(Fz,Fx,Fy,Mx,Mz)
    # print("Fx:", Fx)
    
    return {
        "l": l,         # [m] distance from arm tip
        "Fx": Fx,       # [N] force in x-direction, shape = (271, 181), rows correspond to psi, columns correspond to chi
        "Fy": Fy,       # [N] force in y-direction, see above
        "Fz": Fz,       # [N] force in z-direction, shape = (181,), rows correspond to chi
        "Mx": Mx,       # [Nm] moment about x-axis, see above
        "Mz": Mz        # [Nm] moment about z-axis, see see Fx
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
    # x = [1,0,-1]
    # z = [-a,0,a]
    # X = [[1,0,-1]    Z = [[-a,0,a]]
    #      [1,0,-1]         [-a,0,a]
    #      [1,0,-1]]        [-a,0,a]]

    # Normal stress superposition
    # sigma= Fz/A + Mx*z/Ixx + Mz*x/Izz
    sigma_axial = loads["Fy"] / A
    sigma_bending_on_x = loads["Mx"][np.newaxis, :, np.newaxis, np.newaxis] * Z[np.newaxis, np.newaxis, :, :] / geometry["Ixx"]
    print("Mx:", loads["Mx"])
    print("Z:", Z)
    print("Bending Stress x size:", sigma_bending_on_x.shape)
    sigma_bending_on_y = loads["Mz"][:, :, np.newaxis, np.newaxis] * X[np.newaxis, np.newaxis, :, :] / geometry["Izz"]
    # print("Bending Stress y:",sigma_bending_on_y)
    
    sigma_axial = sigma_axial[:, :, np.newaxis, np.newaxis] # REshape to (271, 181, 1\, 1)
    sigma_yy = sigma_axial + sigma_bending_on_x + sigma_bending_on_y

    # Shear stresses
    tau_yx = (2 * loads["Fz"] / geometry["Ixx"])[np.newaxis, :, np.newaxis, np.newaxis] * (geometry["t"]**2 / 4 - Z**2)[np.newaxis, np.newaxis, :, :]
    tau_yz = (2 * loads["Fx"] / geometry["Izz"])[:, :, np.newaxis, np.newaxis] * (geometry["w"]**2 / 4 - X**2)[np.newaxis, np.newaxis, :, :]

    # von Misses stress
    sigma_misses = np.sqrt(sigma_yy**2 + 3*tau_yx**2+ 3*tau_yz**2)

    # Max stress
    sigma_normal_max = np.max(np.abs(sigma_yy))
    ind_norm = np.unravel_index(np.argmax(np.abs(sigma_yy)), sigma_yy.shape)
    ind_norm_sign = np.sign(sigma_yy[ind_norm]) # 1 for tension, -1 for compression
    sigma_misses_max = np.max(np.abs(sigma_misses))
    ind_misses = np.unravel_index(np.argmax(np.abs(sigma_misses)), sigma_misses.shape)
    ind_misses_sign = np.sign(sigma_misses[ind_misses]) # 1 for tension, -1 for compression
    # AI autocompleted this for me :). I did check it. Argmax turns sigma_yy into a 1d array and then finds the index of the maximum value, then adding sigma_misses.shape turns the index from the 1d variant into the 4d variant

    print(f'Max normal stress of, {np.round(sigma_normal_max / 1e6, 2)}, MPa, at psi = {np.round(np.degrees(psi[ind_norm[0]]), 2)} deg, chi = {np.round(np.degrees(chi[ind_norm[1]]), 2)} deg, sign: {"tension" if ind_norm_sign == 1 else "compression"}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the normal stresses
    # cmap='bwr_r' gives +=blue (tension) and -=red (compression)
    color_bar_1_ref = ax1.pcolormesh(X * 1000, Z * 1000, sigma_yy[ind_norm[0], ind_norm[1], :, :] / 1e6, cmap='bwr_r', vmin=-np.abs(sigma_normal_max / 1e6), vmax=np.abs(sigma_normal_max / 1e6))

    # Invert x-axis to be consistent with our coord definition
    ax1.invert_xaxis()

    # Add the max stress
    ax1.text(10, 12, f'stress_normal_max = {np.round(sigma_normal_max / 1e6, 2)} MPa', fontsize=10)

    # Plot axis and legend
    fig.colorbar(color_bar_1_ref, label='Stress [MPa]', ax=ax1)
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('z [mm]')
    ax1.set_title(f'Normal_yy at l={l * 1000} mm')
    ax1.axis('equal')

    # Plot the von Misses stress
    color_bar_2_ref = ax2.pcolormesh(X * 1000, Z * 1000, sigma_misses[ind_misses[0], ind_misses[1], :, :] / 1e6, cmap='bwr_r', vmin=-np.abs(sigma_misses_max / 1e6), vmax=np.abs(sigma_misses_max / 1e6))
    ax2.invert_xaxis()
    # Add the max stress
    ax2.text(10, 12, f'stress_von_Misses_max = {np.round(sigma_misses_max / 1e6, 2)} MPa', fontsize=10)
    fig.colorbar(color_bar_2_ref, label='Stress [MPa]', ax=ax2)
    ax2.set_xlabel('x [mm]')
    ax2.set_ylabel('z [mm]')
    ax2.set_title(f'von Misses stress at l={l * 1000} mm')
    ax2.axis('equal')

    plt.show()



l = 50 / 1000      # [m] from tip of arm
plot_stresses(l)
