"""
qprop_tuner.py - Interactive QPROP parameter tuning GUI
Overlay simulated thrust/power vs measured data with live matplotlib sliders.
Run with: python qprop_tuner.py
"""

import subprocess
import os
import re
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

# ──────────────────────────────────────────────────────────────
# MEASURED DATA  (Throttle%, Thrust N, Voltage V, Current A, RPM, Power W, Eff g/W)
# ──────────────────────────────────────────────────────────────
MEASURED = np.array([
    [50,  1.738, 16.82, 2.94,  18867, 49.45,  3.58],
    [55,  2.042, 16.82, 3.57,  20324, 60.05,  3.47],
    [60,  2.369, 16.81, 4.29,  21796, 72.11,  3.35],
    [65,  2.709, 16.80, 5.05,  23130, 84.84,  3.26],
    [70,  3.034, 16.79, 5.92,  24519, 99.40,  3.11],
    [75,  3.339, 16.78, 6.69,  25627, 112.26, 3.03],
    [80,  3.622, 16.77, 7.51,  26720, 125.94, 2.93],
    [85,  3.996, 16.76, 8.55,  27903, 143.30, 2.84],
    [90,  4.396, 16.75, 9.77,  29249, 163.65, 2.74],
    [95,  4.787, 16.73, 11.04, 30352, 184.70, 2.64],
    [100, 5.208, 16.71, 12.45, 31361, 208.04, 2.55],
])
MEAS_RPM    = MEASURED[:, 4]
MEAS_THRUST = MEASURED[:, 1]
MEAS_POWER  = MEASURED[:, 5]
MEAS_VOLT   = MEASURED[:, 2]


# ──────────────────────────────────────────────────────────────
# DEFAULT PARAMETERS
# ──────────────────────────────────────────────────────────────
DEFAULTS = {
    "cl0":    0.4222,
    "cl_a":   5.19,
    "cl_min": -0.35,
    "cl_max": 1.4,
    "cd0":    0.2,
    "cd2u":   0.018,
    "cd2l":   0.06,
    "clcd0":  0.49,
    "re_ref": 50000,
    "re_exp": -1.5,
}

# Motor (Speed-400 equivalent)
MOTOR_R  = 0.141   # Ohm
MOTOR_IO = 0.62    # A
MOTOR_KV = 2800    # rpm/V

# Prop geometry
N_BLADES = 3
R_HUB    = 0.0044
R_TIP    = 0.044

# Simulation sweep: use measured voltages, sweep rpm range
RPM_SWEEP = np.linspace(15000, 33000, 40)


# ──────────────────────────────────────────────────────────────
# FILE WRITERS
# ──────────────────────────────────────────────────────────────
def write_prop_input(params, filename):
    """Write QMIL input file."""
    lines = [
        "MyProp_tuner\n",
        f" {N_BLADES} ! Nblades\n\n",
        f" {params['cl0']} {params['cl_a']} ! CL0 CLa\n",
        f" {params['cl_min']} {params['cl_max']} ! CLmin CLmax\n\n",
        f" {params['cd0']} {params['cd2u']} {params['cd2l']} {params['clcd0']} ! CD0 CD2u CD2l CLCD0\n",
        f" {params['re_ref']:.0f} {params['re_exp']} ! REref REexp\n\n",
        " 0.0 0.5 1.0 ! XIdes\n",
        " 0.0 0.5 0.4 ! CLdes\n\n",
        f" {R_HUB} ! hub radius\n",
        f" {R_TIP} ! tip radius\n",
        " 10 ! speed\n",
        " 30000 ! rpm\n\n",
        " 1.4715 ! Thrust\n",
        " 0 ! Power\n\n",
        " 0 0 ! Ldes KQdes\n\n",
        " 30 ! Nout\n",
    ]
    with open(filename, 'w') as f:
        f.writelines(lines)


def write_motor_file(filename):
    with open(filename, 'w') as f:
        f.write("Speed-400\n\n")
        f.write(" 1 ! Motor type\n\n")
        f.write(f" {MOTOR_R}\n {MOTOR_IO}\n {MOTOR_KV}\n")


# ──────────────────────────────────────────────────────────────
# QMIL + QPROP RUNNER
# ──────────────────────────────────────────────────────────────
def run_simulation(params):
    """
    Run QMIL to generate prop geometry, then sweep QPROP across
    the measured voltage/RPM range. Returns arrays (rpm, thrust, power).
    Falls back to analytic model if binaries are unavailable.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        qmil_in  = os.path.join(tmpdir, "design.inp")
        prop_f   = os.path.join(tmpdir, "prop.dat")
        motor_f  = os.path.join(tmpdir, "motor.dat")

        write_prop_input(params, qmil_in)
        write_motor_file(motor_f)

        # --- QMIL ---
        try:
            r = subprocess.run(["qmil", qmil_in, prop_f],
                               capture_output=True, text=True, timeout=10)
            if not os.path.exists(prop_f):
                raise FileNotFoundError("qmil produced no output")
        except Exception as e:
            print(f"[WARN] qmil failed ({e}), using analytic fallback")
            return analytic_model(params)

        # --- QPROP sweep ---
        # Build voltage string covering our measured range
        volt_str = f"8,16/11"
        try:
            r = subprocess.run(
                ["qprop", prop_f, motor_f, "0", "0", volt_str],
                capture_output=True, text=True, timeout=20
            )
            return parse_qprop_output(r.stdout)
        except Exception as e:
            print(f"[WARN] qprop failed ({e}), using analytic fallback")
            return analytic_model(params)


def parse_qprop_output(text):
    """Extract V, rpm, T, Pshaft columns from QPROP stdout."""
    rpms, thrusts, powers = [], [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            rpms.append(float(parts[1]))
            thrusts.append(float(parts[3]))
            powers.append(float(parts[5]))
        except ValueError:
            continue
    if not rpms:
        return None, None, None
    return np.array(rpms), np.array(thrusts), np.array(powers)


# ──────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────
class TunerGUI:
    SLIDER_DEFS = [
        # (label,   param_key,  min,     max,    valinit,  fmt)
        ("cd0",      "cd0",      0.005,   0.5,    DEFAULTS["cd0"],    "%.4f"),
        ("cd2u",     "cd2u",     0.005,   0.30,   DEFAULTS["cd2u"],   "%.4f"),
        ("cd2l",     "cd2l",     0.005,   0.20,   DEFAULTS["cd2l"],   "%.4f"),
        ("clcd0",    "clcd0",    0.0,     1.0,    DEFAULTS["clcd0"],  "%.3f"),
        ("cl_max",   "cl_max",   0.5,     2.0,    DEFAULTS["cl_max"], "%.3f"),
        ("cl0",      "cl0",      0.0,     1.0,    DEFAULTS["cl0"],    "%.4f"),
        ("cl_a",     "cl_a",     2.0,     8.0,    DEFAULTS["cl_a"],   "%.3f"),
        ("re_exp",   "re_exp",  -2.5,    -0.1,   DEFAULTS["re_exp"],  "%.3f"),
        ("re_ref",   "re_ref",   10000,  100000, DEFAULTS["re_ref"],  "%.0f"),
    ]

    def __init__(self):
        self.params = dict(DEFAULTS)
        self._build_figure()
        self._init_plots()
        self._update(None)

    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.patch.set_facecolor('#1c1c1e')

        # Layout: left = plots (2/3), right = sliders (1/3)
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=self.fig)
        gs_plots = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.35)

        self.ax_thrust = self.fig.add_subplot(gs_plots[0])
        self.ax_power  = self.fig.add_subplot(gs_plots[1])
        self.ax_sliders = self.fig.add_subplot(gs[1])
        self.ax_sliders.set_visible(False)

        for ax in [self.ax_thrust, self.ax_power]:
            ax.set_facecolor('#2c2c2e')
            ax.tick_params(colors='#ebebf5', labelsize=9)
            ax.spines[:].set_color('#48484a')
            ax.xaxis.label.set_color('#ebebf5')
            ax.yaxis.label.set_color('#ebebf5')
            ax.title.set_color('#ebebf5')
            ax.grid(True, color='#3a3a3c', linewidth=0.5)

        self.ax_thrust.set_title("Thrust vs RPM", fontsize=11, fontweight='bold')
        self.ax_thrust.set_xlabel("RPM")
        self.ax_thrust.set_ylabel("Thrust (N)")

        self.ax_power.set_title("Power vs RPM", fontsize=11, fontweight='bold')
        self.ax_power.set_xlabel("RPM")
        self.ax_power.set_ylabel("Shaft Power (W)")

        # Measured data scatter
        self.ax_thrust.scatter(MEAS_RPM, MEAS_THRUST, color='#ff9f0a',
                               zorder=5, s=50, label='Measured', marker='o')
        self.ax_power.scatter(MEAS_RPM, MEAS_POWER, color='#ff9f0a',
                              zorder=5, s=50, label='Measured', marker='o')

        self.ax_thrust.legend(fontsize=8, facecolor='#2c2c2e', labelcolor='#ebebf5')
        self.ax_power.legend(fontsize=8, facecolor='#2c2c2e', labelcolor='#ebebf5')

        # Simulated lines (initially empty)
        self.line_thrust, = self.ax_thrust.plot([], [], color='#0a84ff',
                                                 lw=2, label='Simulated')
        self.line_power,  = self.ax_power.plot([],  [], color='#30d158',
                                                lw=2, label='Simulated')
        self.ax_thrust.legend(fontsize=8, facecolor='#2c2c2e', labelcolor='#ebebf5')
        self.ax_power.legend(fontsize=8, facecolor='#2c2c2e', labelcolor='#ebebf5')

        # Status text
        self.status_text = self.fig.text(0.01, 0.01, "", fontsize=7,
                                          color='#636366', ha='left')

        # Build sliders
        self._sliders = {}
        n = len(self.SLIDER_DEFS)
        slider_panel_x = 0.69   # left edge of slider column
        slider_w       = 0.25
        slider_h       = 0.032
        top            = 0.90
        gap            = 0.082

        for i, (label, key, vmin, vmax, vinit, fmt) in enumerate(self.SLIDER_DEFS):
            y = top - i * gap
            ax_s = self.fig.add_axes([slider_panel_x, y, slider_w, slider_h],
                                      facecolor='#2c2c2e')
            sl = Slider(ax_s, label, vmin, vmax, valinit=vinit,
                        color='#0a84ff', track_color='#48484a')
            sl.label.set_color('#ebebf5')
            sl.label.set_fontsize(9)
            sl.valtext.set_color('#ff9f0a')
            sl.valtext.set_fontsize(8)
            sl.on_changed(self._update)
            self._sliders[key] = sl

        # Reset button
        ax_btn = self.fig.add_axes([slider_panel_x, top - n * gap - 0.02,
                                    0.10, 0.035])
        self.btn_reset = Button(ax_btn, 'Reset', color='#3a3a3c', hovercolor='#636366')
        self.btn_reset.label.set_color('#ebebf5')
        self.btn_reset.on_clicked(self._reset)

        # Print button
        ax_btn2 = self.fig.add_axes([slider_panel_x + 0.13, top - n * gap - 0.02,
                                     0.12, 0.035])
        self.btn_print = Button(ax_btn2, 'Print Params', color='#3a3a3c',
                                hovercolor='#636366')
        self.btn_print.label.set_color('#ebebf5')
        self.btn_print.on_clicked(self._print_params)

    def _init_plots(self):
        # Set initial axis limits from data
        rpm_pad = 1000
        self.ax_thrust.set_xlim(MEAS_RPM.min() - rpm_pad, MEAS_RPM.max() + rpm_pad)
        self.ax_thrust.set_ylim(0, MEAS_THRUST.max() * 1.5)
        self.ax_power.set_xlim(MEAS_RPM.min() - rpm_pad, MEAS_RPM.max() + rpm_pad)
        self.ax_power.set_ylim(0, MEAS_POWER.max() * 1.5)

    def _read_sliders(self):
        for key, sl in self._sliders.items():
            self.params[key] = sl.val

    def _update(self, _val):
        self._read_sliders()
        rpms, thrusts, powers = run_simulation(self.params)

        if rpms is not None and len(rpms) > 0:
            self.line_thrust.set_data(rpms, thrusts)
            self.line_power.set_data(rpms, powers)

            # Auto-scale Y to fit both measured and simulated
            t_max = max(MEAS_THRUST.max(), thrusts.max()) * 1.15
            p_max = max(MEAS_POWER.max(),  powers.max())  * 1.15
            r_min = min(MEAS_RPM.min(), rpms.min()) - 500
            r_max = max(MEAS_RPM.max(), rpms.max()) + 500

            self.ax_thrust.set_xlim(r_min, r_max)
            self.ax_thrust.set_ylim(0, t_max)
            self.ax_power.set_xlim(r_min, r_max)
            self.ax_power.set_ylim(0, p_max)

            # RMS errors
            t_sim = np.interp(MEAS_RPM, rpms, thrusts)
            p_sim = np.interp(MEAS_RPM, rpms, powers)
            t_rms = np.sqrt(np.mean((t_sim - MEAS_THRUST) ** 2))
            p_rms = np.sqrt(np.mean((p_sim - MEAS_POWER)  ** 2))
            self.status_text.set_text(
                f"RMS errors — Thrust: {t_rms:.4f} N   Power: {p_rms:.2f} W"
            )

        self.fig.canvas.draw_idle()

    def _reset(self, _):
        for key, sl in self._sliders.items():
            sl.set_val(DEFAULTS[key])

    def _print_params(self, _):
        self._read_sliders()
        print("\n── Current Parameters ──────────────────")
        print(f"cl_params = {{\"cl0\": {self.params['cl0']:.4f}, "
              f"\"cl_a\": {self.params['cl_a']:.3f}, "
              f"\"cl_min\": {self.params['cl_min']:.3f}, "
              f"\"cl_max\": {self.params['cl_max']:.3f}}}")
        print(f"cd_params = {{\"cd0\": {self.params['cd0']:.4f}, "
              f"\"cd2u\": {self.params['cd2u']:.4f}, "
              f"\"cd2l\": {self.params['cd2l']:.4f}, "
              f"\"clcd0\": {self.params['clcd0']:.3f}}}")
        print(f"re_params = {{\"re_ref\": {self.params['re_ref']:.0f}, "
              f"\"re_exp\": {self.params['re_exp']:.3f}}}")
        print("────────────────────────────────────────\n")

    def show(self):
        self.fig.suptitle("QPROP Interactive Tuner", color='#ebebf5',
                          fontsize=13, fontweight='bold', y=0.98)
        plt.show()


if __name__ == "__main__":
    print("Starting QPROP Interactive Tuner...")
    print("  • Orange dots = measured data")
    print("  • Blue line   = simulated thrust")
    print("  • Green line  = simulated power")
    print("  • 'Print Params' dumps current values to terminal\n")
    gui = TunerGUI()
    gui.show()
