import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

# ── Parameters (edit here) ────────────────────────────────────────────────────
la    = 0.075
lb    = 0.100
F_mag = 20
phi   = 30
shi   = 20
E     = 70e9
n_sec = 200

# ── Cross-section ─────────────────────────────────────────────────────────────
CS_TYPE = 'rect'      # 'rect' | 'circle' | 'hollow_circle' | 'ibeam'

# rect:          a=width,          b=height
# circle:        a=outer diam,     b=ignored
# hollow_circle: a=outer diam,     b=inner diam
# ibeam:         a=flange width,   b=total height,  tw=web thickness, tf=flange thickness
a_cs  = 20e-3    # [m]
b_cs  = 30e-3    # [m]
tw    = 2e-3     # web thickness   [m]  (ibeam only)
tf    = 3e-3     # flange thickness[m]  (ibeam only)

DEFAULT_ELEV, DEFAULT_AZIM = 25, 45

# ── Geometry ──────────────────────────────────────────────────────────────────
arms = [np.array([la, lb, 0]), np.array([-la, lb, 0]),
        np.array([-la,-lb, 0]), np.array([la, -lb, 0])]
tip = arms[0]; L = np.linalg.norm(tip)
e_x = tip / L
e_z = np.array([0., 0., 1.])
e_y = np.cross(e_z, e_x); e_y /= np.linalg.norm(e_y)

# ── Section properties ────────────────────────────────────────────────────────
def section_props(cs_type, a, b, tw=0, tf=0):
    if cs_type == 'rect':
        A  = a * b
        Iy = a * b**3 / 12
        Iz = b * a**3 / 12
        r_o = r_i = None
    elif cs_type == 'circle':
        r_o = a / 2; r_i = None
        A  = np.pi * r_o**2
        Iy = Iz = np.pi * r_o**4 / 4
    elif cs_type == 'hollow_circle':
        r_o, r_i = a / 2, b / 2
        A  = np.pi * (r_o**2 - r_i**2)
        Iy = Iz = np.pi * (r_o**4 - r_i**4) / 4
    elif cs_type == 'ibeam':
        hw = b - 2*tf                        # web height
        A  = 2*(a*tf) + hw*tw
        Iy = (a*b**3 - (a-tw)*hw**3) / 12   # strong axis (bending about y)
        Iz = (2*tf*a**3 + hw*tw**3) / 12    # weak axis
        r_o = r_i = None
    return A, Iy, Iz, r_o, r_i

A, Iy, Iz, r_o, r_i = section_props(CS_TYPE, a_cs, b_cs, tw, tf)

# ── Force decomposition ───────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

phi_r, shi_r = np.radians(phi), np.radians(shi)
F_vec = F_mag * (np.cos(phi_r)*np.cos(shi_r)*e_x +
                 np.sin(phi_r)*np.cos(shi_r)*e_y +
                 np.sin(shi_r)*e_z)
Fx = F_vec @ e_x; Fy = F_vec @ e_y; Fz = F_vec @ e_z

s  = np.linspace(0, L, n_sec)
N  = Fx * np.ones(n_sec)
My = Fz * (L - s)
Mz = -Fy * (L - s)

u = (Fx / (E*A)) * s
v = (Fy / (6*E*Iz)) * (3*L*s**2 - s**3)
w = (Fz / (6*E*Iy)) * (3*L*s**2 - s**3)
pts_def = np.outer(s+u, e_x) + np.outer(v, e_y) + np.outer(w, e_z)
tip_def = pts_def[-1]

# ── Stress grid & mask ────────────────────────────────────────────────────────
if CS_TYPE == 'ibeam':
    half_y, half_z = a_cs/2, b_cs/2
elif r_o:
    half_y = half_z = r_o
else:
    half_y, half_z = a_cs/2, b_cs/2

ny = nz = 80
yy = np.linspace(-half_y, half_y, ny)
zz = np.linspace(-half_z, half_z, nz)
YY, ZZ = np.meshgrid(yy, zz)
R2 = YY**2 + ZZ**2

if CS_TYPE == 'rect':
    mask = (np.abs(YY) <= a_cs/2) & (np.abs(ZZ) <= b_cs/2)
elif CS_TYPE == 'circle':
    mask = R2 <= r_o**2
elif CS_TYPE == 'hollow_circle':
    mask = (R2 <= r_o**2) & (R2 >= r_i**2)
elif CS_TYPE == 'ibeam':
    hw = b_cs - 2*tf
    top_flange    = (np.abs(YY) <= a_cs/2) & (ZZ >=  b_cs/2 - tf)
    bot_flange    = (np.abs(YY) <= a_cs/2) & (ZZ <= -b_cs/2 + tf)
    web           = (np.abs(YY) <= tw/2)   & (np.abs(ZZ) <= hw/2)
    mask = top_flange | bot_flange | web

sigma_all = ((N[:,None,None] / A) +
             (My[:,None,None] * ZZ[None,:,:] / Iy) +
             (-Mz[:,None,None] * YY[None,:,:] / Iz))
sigma_all[:, ~mask] = np.nan
sig_abs_max = np.nanmax(np.abs(sigma_all))

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 6), facecolor='#1a1a2e')
ax3d = fig.add_subplot(121, projection='3d'); ax3d.set_facecolor('#1a1a2e')
ax_cs = fig.add_subplot(122);                 ax_cs.set_facecolor('#1a1a2e')
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.18, top=0.95, wspace=0.3)

# ── 3-D ───────────────────────────────────────────────────────────────────────
ax3d.view_init(elev=DEFAULT_ELEV, azim=DEFAULT_AZIM)
for i, arm in enumerate(arms):
    col = '#ff6b6b' if i == 0 else '#4ecdc4'
    ax3d.plot([0,arm[0]],[0,arm[1]],[0,arm[2]], color=col, lw=1.5, alpha=0.2, ls='--')
ax3d.plot(pts_def[:,0], pts_def[:,1], pts_def[:,2], color='#ff6b6b', lw=2.5)
for arm in arms[1:]:
    ax3d.plot([0,arm[0]],[0,arm[1]],[0,arm[2]], color='#4ecdc4', lw=2.5)
    ax3d.scatter(*arm, color='#4ecdc4', s=30)
ax3d.scatter(*arms[0], color='#ff6b6b', s=30, alpha=0.4)
ax3d.scatter(*tip_def, color='#ffe66d', s=80, zorder=6)
ax3d.text(*tip_def,
          f" ({tip_def[0]*1e3:.2f}, {tip_def[1]*1e3:.2f}, {tip_def[2]*1e3:.2f}) mm",
          fontsize=7, color='#ffe66d')
fv = F_vec * 0.004
ax3d.quiver(*arms[0], *fv, color='#a8ff78', lw=2, arrow_length_ratio=0.25)
ax3d.text(*(arms[0]+fv*1.3), f'F={F_mag}N\nφ={phi}°,ψ={shi}°', color='#a8ff78', fontsize=7)
sec_line, = ax3d.plot([], [], [], color='white', lw=1.5)
lim = lb*1.5
ax3d.set_xlim(-lim,lim); ax3d.set_ylim(-lim,lim); ax3d.set_zlim(-lim*.6,lim*.6)
ax3d.set_box_aspect([1,1,0.6])
for lbl,fn in [('X',ax3d.set_xlabel),('Y',ax3d.set_ylabel),('Z',ax3d.set_zlabel)]:
    fn(lbl, color='white', fontsize=8)
ax3d.tick_params(colors='white', labelsize=6)
ax3d.set_title('Drone frame — deformed', color='white', fontsize=9)

# ── Cross-section ─────────────────────────────────────────────────────────────
h_y, h_z = half_y*1e3, half_z*1e3
cs_img = ax_cs.imshow(sigma_all[0]/1e6, origin='lower', aspect='equal',
                       cmap='RdBu_r', vmin=-sig_abs_max/1e6, vmax=sig_abs_max/1e6,
                       extent=[-h_y, h_y, -h_z, h_z])
cbar = fig.colorbar(cs_img, ax=ax_cs, label='σ [MPa]', shrink=0.8)
cbar.ax.yaxis.label.set_color('white'); cbar.ax.tick_params(colors='white')

# Outlines
if CS_TYPE == 'rect':
    ax_cs.add_patch(plt.Rectangle((-a_cs/2*1e3,-b_cs/2*1e3), a_cs*1e3, b_cs*1e3,
                                   edgecolor='white', facecolor='none', lw=1.5))
elif CS_TYPE == 'circle':
    ax_cs.add_patch(plt.Circle((0,0), r_o*1e3, edgecolor='white', facecolor='none', lw=1.5))
elif CS_TYPE == 'hollow_circle':
    ax_cs.add_patch(plt.Circle((0,0), r_o*1e3, edgecolor='white', facecolor='none', lw=1.5))
    ax_cs.add_patch(plt.Circle((0,0), r_i*1e3, edgecolor='white', facecolor='none', lw=1.5, ls='--'))


# clean redraw for ibeam outlines (simpler unpacked loop)
if CS_TYPE == 'ibeam':
    hw = b_cs - 2*tf
    patches = [
        plt.Rectangle((-a_cs/2*1e3,  (b_cs/2-tf)*1e3), a_cs*1e3,  tf*1e3),   # top flange
        plt.Rectangle((-a_cs/2*1e3, -b_cs/2*1e3),       a_cs*1e3,  tf*1e3),   # bot flange
        plt.Rectangle((-tw/2*1e3,   -hw/2*1e3),           tw*1e3,  hw*1e3),   # web
    ]
    for p in patches:
        p.set_edgecolor('white'); p.set_facecolor('none'); p.set_linewidth(1.5)
        ax_cs.add_patch(p)

ax_cs.axhline(0, color='w', lw=0.8, alpha=0.4, ls='--')
ax_cs.axvline(0, color='w', lw=0.8, alpha=0.4, ls='--')
ax_cs.set_xlabel('y [mm]', color='white', fontsize=8)
ax_cs.set_ylabel('z [mm]', color='white', fontsize=8)
ax_cs.tick_params(colors='white', labelsize=7)
cs_title = ax_cs.set_title('', color='white', fontsize=8)

def update_cs(idx):
    cs_img.set_data(sigma_all[idx] / 1e6)
    cs_title.set_text(
        f"s = {s[idx]*1e3:.1f} mm  |  "
        f"σ_ax={N[idx]/A/1e6:.2f}  "
        f"σ_My={My[idx]*half_z*1e-3/Iy/1e6:.2f}  "
        f"σ_Mz={-Mz[idx]*half_y*1e-3/Iz/1e6:.2f}  MPa"
    )
    # section marker on 3D
    theta = np.linspace(0, 2*np.pi, 40)
    pt = pts_def[idx]
    ring = (pt[:,None]
            + np.outer(e_y, np.cos(theta)) * half_y
            + np.outer(e_z, np.sin(theta)) * half_z)
    sec_line.set_data_3d(ring[0], ring[1], ring[2])

update_cs(0)

# ── Slider ────────────────────────────────────────────────────────────────────
ax_sl = fig.add_axes([0.15, 0.07, 0.55, 0.03], facecolor='#2a2a4a')
slider = Slider(ax_sl, 's/L', 0.0, 1.0, valinit=0.0,
                color='#ff6b6b', track_color='#2a2a4a')
slider.label.set_color('white'); slider.valtext.set_color('white')
slider.on_changed(lambda val: [update_cs(int(val*(n_sec-1))), fig.canvas.draw_idle()])

# ── Buttons ───────────────────────────────────────────────────────────────────
def home(e):   ax3d.view_init(DEFAULT_ELEV, DEFAULT_AZIM); fig.canvas.draw_idle()
def vtop(e):   ax3d.view_init(90, 0);  fig.canvas.draw_idle()
def vfront(e): ax3d.view_init(0,  0);  fig.canvas.draw_idle()
def vside(e):  ax3d.view_init(0, 90);  fig.canvas.draw_idle()

btns = []
for label, pos, cb in [('Home',[0.75,0.05,0.05,0.04],home),
                        ('Top', [0.81,0.05,0.05,0.04],vtop),
                        ('Front',[0.87,0.05,0.05,0.04],vfront),
                        ('Side',[0.93,0.05,0.05,0.04],vside)]:
    bt = plt.Button(fig.add_axes(pos), label, color='#2a2a4a', hovercolor='#4a4a6a')
    bt.label.set_color('white'); bt.label.set_fontsize(8)
    bt.on_clicked(cb); btns.append(bt)

plt.show()