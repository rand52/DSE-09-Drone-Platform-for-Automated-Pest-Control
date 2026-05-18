import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
 
plt.style.use(['science', 'no-latex', 'grid'])
 
# ── Rotor parameters from XROTOR fw2 file ────────────────────────────────────
R        = 35.0           # tip radius [mm]
R_hub    = 0.16 * R       # hub radius [mm]  (XI0 = 0.20)
n_blades = 3
 
r_R = np.array([
    0.0,      # blade root at hub center
    0.20323, 0.21558, 0.23834, 0.26873, 0.30422, 0.34292,
    0.38348, 0.42499, 0.46680, 0.50843, 0.54952, 0.58977,
    0.62894, 0.66683, 0.70327, 0.73810, 0.77119, 0.80239,
    0.83161, 0.85875, 0.88370, 0.90639, 0.92674, 0.94470,
    0.96019, 0.97317, 0.98361, 0.99146, 0.99671, 0.99934])

C_R = np.array([
    0.0,      # chord at root 
    0.57029, 0.55198, 0.52872, 0.50927, 0.49542, 0.48603,
    0.47949, 0.47454, 0.47033, 0.46620, 0.46165, 0.45627,
    0.44970, 0.44163, 0.43178, 0.41988, 0.40574, 0.38918,
    0.37012, 0.34852, 0.32441, 0.29787, 0.26907, 0.23824,
    0.20562, 0.17154, 0.13647, 0.10101, 0.066829, 0.040434])

Beta_deg = np.array([
    0.0,      # blade root at hub center
    54.715, 53.037, 50.165, 46.735, 43.245, 39.976,
    37.045, 34.476, 32.247, 30.321, 28.658, 27.219,
    25.971, 24.886, 23.939, 23.113, 22.390, 21.757,
    21.203, 20.720, 20.299, 19.935, 19.622, 19.356,
    19.134, 18.952, 18.810, 18.705, 18.635, 18.600])
 
r    = r_R * R
C    = C_R * R
Beta = np.radians(Beta_deg)
 
# ── Axial projection ──────────────────────────────────────────────────────────
# Only the tangential (in-plane) component of the chord is visible from the front
C_tang = C * np.cos(Beta)
 
# Pitch axis at 0.25 C from LE (XROTOR convention)
# Blade extends along +y; rotation is CCW viewed from front (+x direction)
x_LE = -0.25 * C_tang   # leading edge
x_TE = +0.75 * C_tang   # trailing edge
 
# Build closed blade outline: LE root→tip, rounded tip point, TE tip→root, close at root
outline_x = np.concatenate([x_LE,  [0.0],    x_TE[::-1]])
outline_y = np.concatenate([r,     [R],       r[::-1]])
 
# ── Helper: 2-D rotation ──────────────────────────────────────────────────────
def rotate_2d(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return x * c - y * s, x * s + y * c
 
# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 3.5))
 
for i in range(n_blades):
    theta = i * 2.0 * np.pi / n_blades
    xr, yr = rotate_2d(outline_x, outline_y, theta)
    ax.fill(xr, yr, color='#1a1a1a', zorder=2, linewidth=0)
    ax.plot(np.append(xr, xr[0]), np.append(yr, yr[0]),
            color='#555555', linewidth=0.6, zorder=3)
 
# Hub circle
theta_hub = np.linspace(0, 2.0 * np.pi, 360)
hx = R_hub * np.cos(theta_hub)
hy = R_hub * np.sin(theta_hub)
ax.fill(hx, hy, color='grey', zorder=4)
ax.plot(hx, hy, color='black', linewidth=0.8, zorder=5)
 
# Hub crosshairs
ax.plot([-R_hub, R_hub], [0.0,   0.0  ], color='black', linewidth=0.5, zorder=6)
ax.plot([0.0,   0.0  ], [-R_hub, R_hub], color='black', linewidth=0.5, zorder=6)
 
# ── Axes formatting ───────────────────────────────────────────────────────────
margin = 1.12 * R
ax.set_aspect('equal')
ax.set_xlim(-margin, margin)
ax.set_ylim(-margin, margin)
ax.set_xlabel(r'$x$ (mm)')
ax.set_ylabel(r'$y$ (mm)')
 
fig.tight_layout()
fig.savefig('propeller_axial.png', dpi=300, bbox_inches='tight')
plt.show()
