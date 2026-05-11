"""
qprop_tuner.py - Interactive QPROP prop & motor performance viewer
Adjust prop design point and motor parameters via sliders.
Plots show Thrust, Power, and Efficiency vs RPM for the designed prop.

Run with: python qprop_tuner.py

QPROP output columns (0-based index):
  0:V(m/s)  1:rpm  2:Dbeta  3:T(N)  4:Q(N-m)  5:Pshaft(W)  6:Volts  7:Amps
  8:effmot  9:effprop  10:adv  11:CT  12:CP  13:DV(m/s)  14:eff
  15:Pelec(W)  16:Pprop(W)  17:cl_avg  18:cd_avg
"""

import subprocess
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

# ──────────────────────────────────────────────────────────────
# FIXED AIRFOIL PARAMETERS
# ──────────────────────────────────────────────────────────────
AIRFOIL = {
    "cl0":    0.4222,
    "cl_a":   5.19,
    "cl_min": -0.35,
    "cl_max": 1.4,
    "cd0":    0.013,
    "cd2u":   0.018,
    "cd2l":   0.06,
    "clcd0":  0.49,
    "re_ref": 50000,
    "re_exp": -0.5,
}

N_BLADES = 3
R_HUB    = 0.0044   # m

# ──────────────────────────────────────────────────────────────
# TUNABLE DEFAULTS
# ──────────────────────────────────────────────────────────────
DEFAULTS = {
    "kv":             2800.0,
    "motor_r":        0.141,
    "motor_io":       0.62,
    "voltage":        16.8,
    "design_thrust":  1.4715,
    "design_speed":   10.0,
    "design_rpm":     30000.0,
    "r_tip":          0.044,
}


# ──────────────────────────────────────────────────────────────
# FILE WRITERS
# ──────────────────────────────────────────────────────────────
def write_prop_input(params, filename):
    af = AIRFOIL
    lines = [
        "MyProp\n",
        f" {N_BLADES} ! Nblades\n\n",
        f" {af['cl0']} {af['cl_a']} ! CL0 CLa\n",
        f" {af['cl_min']} {af['cl_max']} ! CLmin CLmax\n\n",
        f" {af['cd0']} {af['cd2u']} {af['cd2l']} {af['clcd0']} ! CD0 CD2u CD2l CLCD0\n",
        f" {af['re_ref']:.0f} {af['re_exp']} ! REref REexp\n\n",
        " 0.0 0.5 1.0 ! XIdes\n",
        " 0.0 0.5 0.4 ! CLdes\n\n",
        f" {R_HUB} ! hub radius\n",
        f" {params['r_tip']:.5f} ! tip radius\n",
        f" {params['design_speed']:.2f} ! design airspeed (m/s)\n",
        f" {params['design_rpm']:.0f} ! design RPM\n\n",
        f" {params['design_thrust']:.4f} ! design Thrust (N)\n",
        " 0 ! Power (0 = use Thrust)\n\n",
        " 0 0 ! Ldes KQdes\n\n",
        " 30 ! Nout\n",
    ]
    with open(filename, 'w') as f:
        f.writelines(lines)


def write_motor_file(params, filename):
    with open(filename, 'w') as f:
        f.write("Motor\n\n")
        f.write(" 1 ! Motor type (Kv model)\n\n")
        f.write(f" {params['motor_r']:.4f}\n")
        f.write(f" {params['motor_io']:.4f}\n")
        f.write(f" {params['kv']:.1f}\n")


# ──────────────────────────────────────────────────────────────
# SIMULATION
# ──────────────────────────────────────────────────────────────
def run_simulation(params):
    """
    Returns (rpm, thrust, power, efficiency) arrays, or (None,...) with a
    human-readable error string as a 5th element on failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        qmil_in = os.path.join(tmpdir, "design.inp")
        prop_f  = os.path.join(tmpdir, "prop.dat")
        motor_f = os.path.join(tmpdir, "motor.dat")

        write_prop_input(params, qmil_in)
        write_motor_file(params, motor_f)

        # ── QMIL ──────────────────────────────────────────────
        try:
            r = subprocess.run(
                ["qmil", qmil_in, prop_f],
                capture_output=True, text=True, timeout=10
            )
        except FileNotFoundError:
            return None, None, None, None, "qmil binary not found on PATH"
        except Exception as e:
            return None, None, None, None, f"qmil exception: {e}"

        if r.returncode != 0:
            msg = (r.stderr or r.stdout or "no output").strip()
            return None, None, None, None, f"qmil exit {r.returncode}: {msg[:200]}"

        if not os.path.exists(prop_f):
            msg = (r.stdout or "no stdout").strip()
            return None, None, None, None, f"qmil gave no prop file. stdout: {msg[:200]}"

        # ── QPROP ─────────────────────────────────────────────
        # QPROP maps one voltage → one RPM operating point.
        # Sweep from ~20% up to the set voltage in 20 steps for a full curve.
        volt_max = params["voltage"]
        volt_min = max(1.0, volt_max * 0.20)
        volt_str = f"{volt_min:.2f},{volt_max:.2f}/20"

        try:
            r = subprocess.run(
                ["qprop", prop_f, motor_f, "10", "0", volt_str],
                capture_output=True, text=True, timeout=20
            )
        except FileNotFoundError:
            return None, None, None, None, "qprop binary not found on PATH"
        except Exception as e:
            return None, None, None, None, f"qprop exception: {e}"

        if r.returncode != 0:
            msg = (r.stderr or r.stdout or "no output").strip()
            return None, None, None, None, f"qprop exit {r.returncode}: {msg[:200]}"

        rpms, thrusts, powers, effs = parse_qprop_output(r.stdout)

        if rpms is None:
            preview = "\n".join(r.stdout.splitlines()[:30])
            return None, None, None, None, f"Parser found no data rows. qprop stdout:\n{preview}"

        return rpms, thrusts, powers, effs, None


def parse_qprop_output(text):
    """
    Column indices (0-based after splitting each data row):
      0:V  1:rpm  2:Dbeta  3:T(N)  4:Q  5:Pshaft  6:Volts  7:Amps
      8:effmot  9:effprop  10:adv  11:CT  12:CP  13:DV  14:eff
      15:Pelec  16:Pprop  17:cl_avg  18:cd_avg
    """
    rpms, thrusts, powers, effs = [], [], [], []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Strip leading # — QPROP sometimes marks the operating-point summary
        # line with a # even though it contains numeric data
        if stripped.startswith('#'):
            stripped = stripped[1:].strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 16:
            continue
        try:
            rpm    = float(parts[1])
            thrust = float(parts[3])
            pshaft = float(parts[16])
            eff    = float(parts[14])
            rpms.append(rpm)
            thrusts.append(thrust)
            powers.append(pshaft)
            effs.append(eff)
        except (ValueError, IndexError):
            continue

    if not rpms:
        return None, None, None, None

    order = np.argsort(rpms)
    return (np.array(rpms)[order],
            np.array(thrusts)[order],
            np.array(powers)[order],
            np.array(effs)[order])


# ──────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────
class TunerGUI:
    SLIDER_DEFS = [
        # (label,           key,              min,     max,    fmt)
        ("Kv  (rpm/V)",    "kv",            1500,   4500,   "%.0f"),
        ("R   (Ω)",        "motor_r",        0.05,   0.50,  "%.4f"),
        ("I₀  (A)",        "motor_io",       0.10,   2.00,  "%.3f"),
        ("Voltage  (V)",   "voltage",        6.0,   25.2,   "%.1f"),
        ("T_des  (N)",     "design_thrust",  0.50,   8.00,  "%.3f"),
        ("V_des  (m/s)",   "design_speed",   2.00,  25.0,   "%.1f"),
        ("RPM_des",        "design_rpm",    15000, 40000,   "%.0f"),
        ("R_tip  (m)",     "r_tip",          0.030,  0.090, "%.4f"),
    ]

    def __init__(self):
        self.params = dict(DEFAULTS)
        self._build_figure()
        self._update(None)

    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.patch.set_facecolor('#1c1c1e')

        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2], figure=self.fig,
                               left=0.05, right=0.98, top=0.92, bottom=0.08)
        gs_plots = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0], hspace=0.50)

        self.ax_thrust = self.fig.add_subplot(gs_plots[0])
        self.ax_power  = self.fig.add_subplot(gs_plots[1])
        self.ax_eff    = self.fig.add_subplot(gs_plots[2])

        ax_bg = self.fig.add_subplot(gs[1])
        ax_bg.set_visible(False)

        for ax, title, ylabel in [
            (self.ax_thrust, "Thrust vs RPM",      "Thrust (N)"),
            (self.ax_power,  "Shaft Power vs RPM", "Power (W)"),
            (self.ax_eff,    "Efficiency vs RPM",  "Efficiency (g/W)"),
        ]:
            ax.set_facecolor('#2c2c2e')
            ax.tick_params(colors='#ebebf5', labelsize=8)
            ax.spines[:].set_color('#48484a')
            ax.xaxis.label.set_color('#ebebf5')
            ax.yaxis.label.set_color('#ebebf5')
            ax.title.set_color('#ebebf5')
            ax.grid(True, color='#3a3a3c', linewidth=0.5)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel("RPM", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)

        self.line_thrust, = self.ax_thrust.plot([], [], color='#0a84ff', lw=2)
        self.line_power,  = self.ax_power.plot( [], [], color='#30d158', lw=2)
        self.line_eff,    = self.ax_eff.plot(   [], [], color='#bf5af2', lw=2)

        vline_kw = dict(color='#ff9f0a', lw=1, linestyle='--', alpha=0.7, label='Design RPM')
        self.vline_t = self.ax_thrust.axvline(DEFAULTS["design_rpm"], **vline_kw)
        self.vline_p = self.ax_power.axvline( DEFAULTS["design_rpm"], **vline_kw)
        self.vline_e = self.ax_eff.axvline(   DEFAULTS["design_rpm"], **vline_kw)

        for ax in [self.ax_thrust, self.ax_power, self.ax_eff]:
            ax.legend(fontsize=7, facecolor='#2c2c2e', labelcolor='#ebebf5')

        # Two-line status: error detail on second line
        self.status_text = self.fig.text(
            0.01, 0.012, "", fontsize=7, color='#ebebf5', ha='left'
        )
        self.error_text = self.fig.text(
            0.01, 0.002, "", fontsize=6, color='#ff453a', ha='left'
        )

        # Sliders
        self._sliders = {}
        sl_left   = 0.625
        sl_width  = 0.32
        sl_height = 0.032
        top_y     = 0.88
        row_gap   = 0.096

        for i, (label, key, vmin, vmax, fmt) in enumerate(self.SLIDER_DEFS):
            y    = top_y - i * row_gap
            ax_s = self.fig.add_axes([sl_left, y, sl_width, sl_height],
                                     facecolor='#2c2c2e')
            sl   = Slider(ax_s, label, vmin, vmax, valinit=DEFAULTS[key],
                          color='#0a84ff', track_color='#48484a')
            sl.label.set_color('#ebebf5')
            sl.label.set_fontsize(8)
            sl.valtext.set_color('#ff9f0a')
            sl.valtext.set_fontsize(7)
            sl.on_changed(self._update)
            self._sliders[key] = sl

        btn_y = top_y - len(self.SLIDER_DEFS) * row_gap

        ax_btn = self.fig.add_axes([sl_left, btn_y, 0.13, 0.038])
        self.btn_reset = Button(ax_btn, 'Reset', color='#3a3a3c', hovercolor='#636366')
        self.btn_reset.label.set_color('#ebebf5')
        self.btn_reset.label.set_fontsize(8)
        self.btn_reset.on_clicked(self._reset)

        ax_btn2 = self.fig.add_axes([sl_left + 0.15, btn_y, 0.17, 0.038])
        self.btn_print = Button(ax_btn2, 'Print Params', color='#3a3a3c', hovercolor='#636366')
        self.btn_print.label.set_color('#ebebf5')
        self.btn_print.label.set_fontsize(8)
        self.btn_print.on_clicked(self._print_params)

    def _read_sliders(self):
        for key, sl in self._sliders.items():
            self.params[key] = sl.val

    def _update(self, _val):
        self._read_sliders()
        rpms, thrusts, powers, effs, err = run_simulation(self.params)

        design_rpm = self.params["design_rpm"]
        for vline in [self.vline_t, self.vline_p, self.vline_e]:
            vline.set_xdata([design_rpm, design_rpm])

        if err:
            # Print full detail to terminal, show condensed in GUI
            print(f"\n[ERROR] {err}\n")
            first_line = err.splitlines()[0]
            self.status_text.set_text("Simulation failed — see terminal for details")
            self.error_text.set_text(first_line[:120])
        else:
            self.error_text.set_text("")

        if rpms is not None and len(rpms) > 0:
            self.line_thrust.set_data(rpms, thrusts)
            self.line_power.set_data( rpms, powers)
            self.line_eff.set_data(   rpms, effs)

            pad = (rpms.max() - rpms.min()) * 0.05 or 500
            r_min, r_max = rpms.min() - pad, rpms.max() + pad

            self.ax_thrust.set_xlim(r_min, r_max)
            self.ax_thrust.set_ylim(0, thrusts.max() * 1.20)
            self.ax_power.set_xlim( r_min, r_max)
            self.ax_power.set_ylim( 0, powers.max()  * 1.20)
            self.ax_eff.set_xlim(   r_min, r_max)
            self.ax_eff.set_ylim(   0, effs.max()    * 1.20)

            t_at = np.interp(design_rpm, rpms, thrusts)
            p_at = np.interp(design_rpm, rpms, powers)
            e_at = np.interp(design_rpm, rpms, effs)
            self.status_text.set_text(
                f"At design RPM ({design_rpm:.0f}):  "
                f"Thrust = {t_at:.3f} N    "
                f"Power = {p_at:.1f} W    "
                f"Eff = {e_at:.2f} g/W"
            )

        self.fig.canvas.draw_idle()

    def _reset(self, _):
        for key, sl in self._sliders.items():
            sl.set_val(DEFAULTS[key])

    def _print_params(self, _):
        self._read_sliders()
        p = self.params
        print("\n── Parameters ───────────────────────────────────────")
        print(f"motor        = {{kv: {p['kv']:.1f} rpm/V,  R: {p['motor_r']:.4f} Ω,  I0: {p['motor_io']:.4f} A}}")
        print(f"supply       = {{voltage: {p['voltage']:.1f} V}}")
        print(f"design_point = {{thrust: {p['design_thrust']:.3f} N,  speed: {p['design_speed']:.1f} m/s,  rpm: {p['design_rpm']:.0f}}}")
        print(f"geometry     = {{r_tip: {p['r_tip']:.4f} m,  r_hub: {R_HUB} m,  blades: {N_BLADES}}}")
        print("─────────────────────────────────────────────────────\n")

    def show(self):
        self.fig.suptitle("QPROP Prop Performance Viewer",
                          color='#ebebf5', fontsize=13, fontweight='bold', y=0.98)
        plt.show()


if __name__ == "__main__":
    print("QPROP Prop Performance Viewer")
    print("  Blue   = Thrust (N)")
    print("  Green  = Shaft Power (W)")
    print("  Purple = Efficiency (g/W)")
    print("  Orange dashed = design RPM")
    print("  Errors are printed in full to this terminal\n")
    gui = TunerGUI()
    gui.show()