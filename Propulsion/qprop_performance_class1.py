import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- QMIL and QPROP Interface Classes ---

class QMILInterface:
    def __init__(self, name="Propeller"):
        self.name = name
        self.n_blades = 3
        self.cl_params = {"cl0": 0.4171, "cl_a": 5.19, "cl_min": -0.35, "cl_max": 1.4}
        self.cd_params = {"cd0": 0.0485, "cd2u": 0.018, "cd2l": 0.06, "clcd0": 0.49}
        self.re_params = {"re_ref": 50000, "re_exp": -0.27}
        self.dist_r_R = [0.0, 0.5, 1.0]
        self.dist_cl = [0.0, 0.5, 0.4]
        self.radii = {"hub": 0.0044, "tip": 0.044}
        self.op_point = {"vel": 5.0, "rpm": 30000.0}
        self.targets = {"thrust": 4.5, "power": 0}
        self.ldes = 0 
        self.kqdes = 0
        self.n_out = 30

    def write_input_file(self, filename="qmil.inp"):
        with open(filename, 'w') as f:
            f.write(f"{self.name}\n\n")
            f.write(f" {self.n_blades} ! Nblades\n\n")
            f.write(f" {self.cl_params['cl0']} {self.cl_params['cl_a']} ! CL0 CLa\n")
            f.write(f" {self.cl_params['cl_min']} {self.cl_params['cl_max']} ! CLmin CLmax\n\n")
            f.write(f" {self.cd_params['cd0']} {self.cd_params['cd2u']} {self.cd_params['cd2l']} {self.cd_params['clcd0']} ! CD0 CD2u CD2l CLCD0\n")
            f.write(f" {self.re_params['re_ref']} {self.re_params['re_exp']} ! REref REexp\n\n")
            f.write(f" {' '.join(map(str, self.dist_r_R))} ! XIdes\n")
            f.write(f" {' '.join(map(str, self.dist_cl))} ! CLdes\n\n")
            f.write(f" {self.radii['hub']} ! hub radius\n")
            f.write(f" {self.radii['tip']} ! tip radius\n")
            f.write(f" {self.op_point['vel']} ! speed\n")
            f.write(f" {self.op_point['rpm']} ! rpm\n\n")
            f.write(f" {self.targets['thrust']} ! Thrust\n")
            f.write(f" {self.targets['power']} ! Power\n\n")
            f.write(f" {self.ldes} {self.kqdes} ! Ldes KQdes\n\n")
            f.write(f" {self.n_out} ! Nout\n")
        return filename

class QPROPInterface:
    def __init__(self, motor_name="Motor"):
        self.motor_name = motor_name
        self.motor_type = 1
        self.params = [0.141, 0.62, 2800] # R, Io, Kv

    def write_motor_file(self, filename="motor.dat"):
        with open(filename, 'w') as f:
            f.write(f"{self.motor_name}\n\n")
            f.write(f" {self.motor_type} ! Motor type\n\n")
            for p in self.params:
                f.write(f" {p}\n")
        return filename

def weight_from_kv_and_battery_power(kv, battery_power_w):
    m0 = 68.41
    a1 = 1441526.73
    b1 = -1.41
    a2 = 21.81
    b2 = 0.04
    weight_from_motor = a1 * kv**b1
    weight_from_battery = a2 + b2 * battery_power_w
    print(f"Debug: For KV={kv} rpm/V and Battery Power={battery_power_w} W, "
          f"Motor Weight={weight_from_motor:.2f} g, Battery Weight={weight_from_battery:.2f} g")
    return (m0 + weight_from_motor*4 + weight_from_battery)/1000


def thrust_from_tw(tw_ratio, kv, battery_power_w):
    weight_n = (weight_from_kv_and_battery_power(kv, battery_power_w)) * 9.81
    return tw_ratio * weight_n/4

def run_and_parse(qmil_obj, qprop_obj, vel="5.0", rpm_sweep="20000,35000/8"):
    """Runs software workflow and returns performance data as a numpy array"""
    qmil_in = qmil_obj.write_input_file("design.inp")
    prop_file = "generated.prop"
    subprocess.run(["qmil", qmil_in, prop_file], capture_output=True)
    
    motor_file = qprop_obj.write_motor_file("motor.dat")
    
    result = subprocess.run(
        ["qprop", prop_file, motor_file, str(vel), rpm_sweep, "0"],
        capture_output=True, text=True
    )
    
    data = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = [float(x) for x in line.split()]
            if len(parts) >= 16:
                data.append(parts)
        except ValueError:
            continue
            
    return np.array(data)

# --- GUI Logic ---

design = QMILInterface()
motor = QPROPInterface()

# Setup Figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
plt.subplots_adjust(left=0.1, bottom=0.32, hspace=0.35, right=0.95)

def update_plots(data, battery_power_w):
    ax1.clear(); ax2.clear(); ax3.clear()
    if data.size == 0: return
    
    rpms = data[:, 1]
    thrust = data[:, 3]
    eff = data[:, 14]
    pelec = data[:, 15]

    ax1.plot(rpms, thrust, 'r-o', markersize=4)
    ax1.set_ylabel("Thrust [N]")
    ax1.grid(True, linestyle='--')

    ax2.plot(rpms, pelec, 'b-o', markersize=4)
    ax2.set_ylabel("$P_{elec}$ [W]")
    ax2.grid(True, linestyle='--')

    POWER_LIMIT_W = battery_power_w # set your limit
    ax2.axhline(y=POWER_LIMIT_W, color='r', linestyle='--', linewidth=1.2, label=f'Limit ({POWER_LIMIT_W} W)')
    ax2.axvline(x=s_rpmdes.val, color='g', linestyle='--', linewidth=1.2, label=f'Design RPM ({s_rpmdes.val:.0f} rpm)')
    ax2.legend(fontsize=8)

    ax3.plot(rpms, eff, 'g-o', markersize=4)
    ax3.set_ylabel("Total Efficiency")
    ax3.set_xlabel("RPM")
    ax3.grid(True, linestyle='--')
    
    fig.canvas.draw_idle()

# --- Sliders Configuration ---
ax_color = 'aliceblue'

# Column 1: Propeller Design
s_vdes = Slider(plt.axes([0.15, 0.20, 0.25, 0.025], facecolor=ax_color), 'Des Vel [m/s]', 1.0, 25.0, valinit=5.0)
s_rpmdes = Slider(plt.axes([0.15, 0.16, 0.25, 0.025], facecolor=ax_color), 'Des RPM', 5000, 50000, valinit=30000, valstep=500)
s_hub = Slider(plt.axes([0.15, 0.12, 0.25, 0.025], facecolor=ax_color), 'Hub Rad [m]', 0.001, 0.02, valinit=0.0044)
s_tip = Slider(plt.axes([0.15, 0.08, 0.25, 0.025], facecolor=ax_color), 'Tip Rad [m]', 0.01, 0.2, valinit=0.044)

# Column 2: Motor Parameters
s_r = Slider(plt.axes([0.65, 0.20, 0.25, 0.025], facecolor=ax_color), 'Motor R [$\Omega$]', 0.01, 1.0, valinit=0.141)
s_io = Slider(plt.axes([0.65, 0.16, 0.25, 0.025], facecolor=ax_color), 'Motor $I_0$ [A]', 0.1, 5.0, valinit=0.62)
s_kv = Slider(plt.axes([0.65, 0.12, 0.25, 0.025], facecolor=ax_color), 'Motor $K_v$ [rpm/V]', 500, 5000, valinit=2800, valstep=50)
s_tw = Slider(plt.axes([0.65, 0.08, 0.25, 0.025], facecolor=ax_color), 'Thrust/Weight', 2.0, 15.0, valinit=5.0, valstep=0.1)

# Annotation showing derived weight and thrust
info_ax = plt.axes([0.38, 0.06, 0.22, 0.16])
info_ax.axis('off')
info_text = info_ax.text(0.5, 0.5, '', transform=info_ax.transAxes,
                          ha='center', va='center', fontsize=9,
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

def _update_info_label(kv, tw, battery_power_w):
    t_n  = tw * weight_from_kv_and_battery_power(kv, battery_power_w) * 9.81 / 4
    info_text.set_text(
        f"Estimated Design Weight: {weight_from_kv_and_battery_power(kv, battery_power_w)*1000:.1f} g\n"
        f"Design thrust\n"
        f"  {t_n:.3f} N  (T/W = {tw:.1f})"
    )

def update(val):
    kv = s_kv.val
    tw = s_tw.val

    # Update Propeller Object
    design.op_point['vel'] = s_vdes.val
    design.op_point['rpm'] = s_rpmdes.val
    design.radii['hub'] = s_hub.val
    design.radii['tip'] = s_tip.val

    # Update Motor Object
    motor.params = [s_r.val, s_io.val, kv]

    design.targets['thrust'] = thrust_from_tw(tw, kv, battery_power_w=0)  # using the same power limit as before

    data = run_and_parse(design, motor)
    battery_power_w = float(np.interp(s_rpmdes.val, data[:, 1], data[:, 15]))

    design.targets['thrust'] = thrust_from_tw(tw, kv, battery_power_w*4)

    # Update info label
    _update_info_label(kv, tw, battery_power_w*4)

    update_plots(data, battery_power_w)

# Register callbacks
sliders = [s_vdes, s_rpmdes, s_hub, s_tip, s_r, s_io, s_kv, s_tw]
for s in sliders:
    s.on_changed(update)

# Initial execution — set thrust from default T/W and KV before first run
design.targets['thrust'] = thrust_from_tw(s_tw.val, s_kv.val, battery_power_w=127.11/4)
_update_info_label(s_kv.val, s_tw.val, battery_power_w=127.11/4)
initial_data = run_and_parse(design, motor)
update_plots(initial_data, battery_power_w=127.11/4)

plt.suptitle("Interactive QMIL/QPROP Analysis: Propeller Geometry & Motor Performance", fontsize=14)
plt.show()