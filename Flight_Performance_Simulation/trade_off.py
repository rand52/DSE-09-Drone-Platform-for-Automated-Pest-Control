import math
import mujoco
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

from moth import MothTrajectory
from aero import AeroEngine
from controller import FlightController

# --- Parameters — kept identical to main.py ---
Model_path = r"Flight_Performance_Simulation\chameleon.xml"
Moth_Log   = r"Flight_Performance_Simulation\files\log_itrk3.csv"

Drone_Mass   = 0.143
TW_ratio     = 4
Max_Thrust   = Drone_Mass * TW_ratio * 9.81
diameter     = 96.72e-3
Drone_area   = (math.pi * diameter**2) / 4   # ~0.00735 m²

SLACK_MARGIN    = 0.03
Capture_Radius  = 0.18
R_SPOOL         = 0.032
I_SPOOL         = 1e-5
M_EFF           = I_SPOOL / R_SPOOL**2
ZETA            = 0.3
INTERCEPT, BRAKE = 0, 1

MAX_G    = 26.0   # g
MAX_DIST = 0.6    # m


def integrate_spool(P, Pdot, T, F_brake_max, F_reel, dt):
    F_drive   = T - F_reel
    Pdot_tent = Pdot + (F_drive / M_EFF) * dt
    dv_fric   = (F_brake_max / M_EFF) * dt
    if Pdot_tent > 0.0:
        Pdot_new = max(0.0, Pdot_tent - dv_fric)
    elif Pdot_tent < 0.0:
        Pdot_new = min(0.0, Pdot_tent + dv_fric)
    else:
        Pdot_new = 0.0
    return P + Pdot_new * dt, Pdot_new


def run_headless_sim(f_brake_test, wire_e_test, brake_ramp_test, wire_d_test):
    """Headless simulation — logic mirrors main.py step_logic() exactly."""
    model = mujoco.MjModel.from_xml_path(Model_path)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    wire_a = math.pi * (wire_d_test / 2.0) ** 2

    moth     = MothTrajectory(Moth_Log)
    bid      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,   "drone")
    tid      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "tether")
    sid_mount = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,  "drone_mount")
    moth_mid = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moth")]

    L0       = float(data.ten_length[tid]) + SLACK_MARGIN
    k_spring = (wire_a * wire_e_test) / max(L0, 1e-3)
    model.tendon_stiffness[tid] = k_spring

    aero = AeroEngine(model, data,
                      cd_csv_path=r"Flight_Performance_Simulation\files\Cd_values.csv",
                      body_name="drone", area=Drone_area)
    ctrl = FlightController(model, data, body_name="drone",
                            max_thrust=Max_Thrust, mass=Drone_Mass, gravity=9.81)

    data.mocap_pos[moth_mid] = moth.start_pos
    mujoco.mj_forward(model, data)

    P    = float(data.ten_length[tid]) + SLACK_MARGIN
    Pdot = 0.0
    model.tendon_lengthspring[tid] = [0.0, P]

    drone_pos0 = data.xpos[bid].copy()

    # Initial thrust direction — match main.py (includes +z bias)
    moth_start        = moth.start_pos.copy()
    moth_start[2]    += 1.5
    to_moth           = moth_start - drone_pos0
    ctrl.thrust_dir   = to_moth / np.linalg.norm(to_moth) + np.array([0.0, 0.0, 0.3])

    state         = INTERCEPT
    t_state_enter = 0.0
    p_brake_start = np.zeros(3)
    max_g_brake   = 0.0
    max_tension   = 0.0
    prev_vel      = np.zeros(3)
    tension       = 0.0
    t             = 0.0
    timeout       = 10.0

    while t < timeout:
        pos  = data.xpos[bid].copy()
        vel, _ = aero.body_velocity()
        L    = float(data.ten_length[tid])
        Ldot = float(data.ten_velocity[tid])

        k_spring = (wire_a * wire_e_test) / max(L, 1e-3)
        model.tendon_stiffness[tid] = k_spring
        C_TAUT = 2.0 * ZETA * math.sqrt(k_spring * Drone_Mass)

        moth_p     = moth.position(t)
        moth_p[2] += 1.5
        data.mocap_pos[moth_mid] = moth_p

        if state == INTERCEPT:
            aim  = moth_p - pos
            dist = float(np.linalg.norm(aim))
            ctrl.rotate_thrust_toward(aim / dist, dt)
            thrust_cmd = Max_Thrust
            ctrl.release_spool()
            P = max(P, L + SLACK_MARGIN)

            overshot = np.linalg.norm(pos) > np.linalg.norm(moth_p) + Capture_Radius
            if dist < Capture_Radius or pos[2] < -3 or overshot:
                state         = BRAKE
                P             = L
                Pdot          = Ldot
                t_state_enter = t
                p_brake_start = pos.copy()

        elif state == BRAKE:
            # Record g-force every tick — matches main.py (no first-tick exclusion)
            g_force     = float(np.linalg.norm(vel - prev_vel) / dt / 9.81)
            max_g_brake = max(max_g_brake, g_force)

            desired_dir = drone_pos0 / np.linalg.norm(drone_pos0)
            ctrl.rotate_thrust_toward(desired_dir, dt)
            thrust_cmd = Max_Thrust * 0.3   # matches main.py

            ramp = min((t - t_state_enter) / brake_ramp_test, 1.0)
            P, Pdot = integrate_spool(P, Pdot, tension, ramp * f_brake_test, 0.0, dt)

            u_away   = (pos - np.zeros(3)) / L if L > 1e-6 else np.zeros(3)
            v_radial = float(np.dot(vel, u_away))

            if v_radial <= 0.0 and (t - t_state_enter) > 0.5 * brake_ramp_test:
                braking_dist = float(np.linalg.norm(pos - p_brake_start))
                return max_g_brake, braking_dist, max_tension

        P = max(P, 0.02)
        model.tendon_lengthspring[tid] = [0.0, P]

        stretch = L - P
        if stretch > 0.0:
            spring  = k_spring * stretch
            tension = max(0.0, spring + C_TAUT * (Ldot - Pdot))
            max_tension = max(max_tension, tension)
            u   = np.zeros(3) - data.site_xpos[sid_mount]
            nrm = np.linalg.norm(u)
            damp_force = ((tension - spring) / nrm) * u if nrm > 1e-9 else np.zeros(3)
        else:
            tension    = 0.0
            damp_force = np.zeros(3)

        ctrl.apply_drone_wrench(thrust_cmd, aero.compute_drag(),
                                attitude_hold=(state == INTERCEPT))
        data.xfrc_applied[bid, 0:3] += damp_force
        prev_vel = vel.copy()

        try:
            mujoco.mj_step(model, data)
        except Exception:
            return None, None, max_tension

        t += dt

    return None, None, max_tension   # timeout — no clean stop detected


def plot_results(all_results, moduli_gpa, diameters_mm):
    """
    Two figures:
      1. 3D surfaces: stopping distance and peak G vs (E, D) — best brake force at each node.
      2. Brake-force sweep at the optimal (E, D) showing both metrics on dual axes.
    """
    n_E = len(moduli_gpa)
    n_D = len(diameters_mm)

    grid_dist = np.full((n_E, n_D), np.nan)
    grid_g    = np.full((n_E, n_D), np.nan)
    grid_f    = np.full((n_E, n_D), np.nan)

    # Index lookup (avoids float-comparison issues)
    e_idx = {v: i for i, v in enumerate(moduli_gpa)}
    d_idx = {v: i for i, v in enumerate(diameters_mm)}

    by_ed = defaultdict(list)
    for r in all_results:
        if r['Peak_G'] is not None and r['Dist'] is not None:
            if r['Peak_G'] <= MAX_G and r['Dist'] <= MAX_DIST:
                by_ed[(r['E_gpa'], r['D_mm'])].append(r)

    for (E, D), pts in by_ed.items():
        best = min(pts, key=lambda x: x['Dist'])
        i, j = e_idx[E], d_idx[D]
        grid_dist[i, j] = best['Dist']
        grid_g[i, j]    = best['Peak_G']
        grid_f[i, j]    = best['F_brake']

    if np.all(np.isnan(grid_dist)):
        print("No valid configurations found — relax constraints or extend parameter ranges.")
        return

    min_dist  = float(np.nanmin(grid_dist))
    opt_idx   = np.unravel_index(np.nanargmin(grid_dist), grid_dist.shape)
    opt_E     = moduli_gpa[opt_idx[0]]
    opt_D     = diameters_mm[opt_idx[1]]
    opt_G     = float(grid_g[opt_idx])
    opt_F     = float(grid_f[opt_idx])

    print(f"\n{'='*52}")
    print(f"  OPTIMUM CONFIGURATION")
    print(f"  Modulus  E  : {opt_E} GPa")
    print(f"  Diameter D  : {opt_D} mm")
    print(f"  Brake force : {opt_F:.1f} N")
    print(f"  Stop dist   : {min_dist:.3f} m")
    print(f"  Peak G      : {opt_G:.1f} g")
    print(f"{'='*52}\n")

    D_mesh, E_mesh = np.meshgrid(diameters_mm, moduli_gpa)
    dist_ma = np.ma.masked_where(np.isnan(grid_dist), grid_dist)
    g_ma    = np.ma.masked_where(np.isnan(grid_g),    grid_g)

    # ------------------------------------------------------------------ Figure 1: surfaces
    fig1 = plt.figure(figsize=(18, 7))
    fig1.suptitle(
        f"Tether Trade-off Surface  |  G ≤ {MAX_G} g,  Dist ≤ {MAX_DIST} m\n"
        f"Optimum: E = {opt_E} GPa,  D = {opt_D} mm,  "
        f"F_brake = {opt_F:.1f} N   →   Dist = {min_dist:.3f} m,  Peak G = {opt_G:.1f} g",
        fontsize=11
    )

    # -- Left: stopping distance surface
    ax1 = fig1.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(E_mesh, D_mesh, dist_ma,
                              cmap='viridis_r', alpha=0.88,
                              linewidth=0.3, edgecolor='grey')
    ax1.scatter([opt_E], [opt_D], [min_dist],
                color='red', s=250, marker='*', zorder=10,
                label=f'Optimum\n{opt_E} GPa / {opt_D} mm\n{min_dist:.3f} m')
    ax1.set_xlabel('Modulus E (GPa)', labelpad=8)
    ax1.set_ylabel('Diameter D (mm)', labelpad=8)
    ax1.set_zlabel('Min Stop Distance (m)', labelpad=8)
    ax1.set_title('Stopping Distance\n(best F_brake per node)')
    fig1.colorbar(surf1, ax=ax1, shrink=0.45, pad=0.1, label='Dist (m)')
    ax1.legend(fontsize=8, loc='upper right')

    # -- Right: G-force surface
    ax2 = fig1.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(E_mesh, D_mesh, g_ma,
                              cmap='hot_r', alpha=0.88,
                              linewidth=0.3, edgecolor='grey')
    ax2.scatter([opt_E], [opt_D], [opt_G],
                color='blue', s=250, marker='*', zorder=10,
                label=f'At optimum\n{opt_G:.1f} g')
    ax2.set_xlabel('Modulus E (GPa)', labelpad=8)
    ax2.set_ylabel('Diameter D (mm)', labelpad=8)
    ax2.set_zlabel('Peak G-force (g)', labelpad=8)
    ax2.set_title('Peak G-force\n(at best F_brake per node)')
    fig1.colorbar(surf2, ax=ax2, shrink=0.45, pad=0.1, label='G (g)')
    ax2.legend(fontsize=8, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # ------------------------------------------------------------------ Figure 2: F sweep
    opt_sims = [r for r in all_results
                if r['E_gpa'] == opt_E and r['D_mm'] == opt_D
                and r['Peak_G'] is not None and r['Dist'] is not None]

    # Best result per brake-force level (minimise distance, over all ramp times)
    by_f = defaultdict(list)
    for r in opt_sims:
        by_f[r['F_brake']].append(r)

    rows = sorted(
        [min(pts, key=lambda x: x['Dist']) for pts in by_f.values()],
        key=lambda x: x['F_brake']
    )

    if rows:
        fs = [r['F_brake'] for r in rows]
        ds = [r['Dist']    for r in rows]
        gs = [r['Peak_G']  for r in rows]

        fig2, ax_d = plt.subplots(figsize=(10, 5))
        ax_g = ax_d.twinx()

        ax_d.plot(fs, ds, 'b-o', markersize=5, label='Stop distance (m)')
        ax_g.plot(fs, gs, 'r-s', markersize=5, label='Peak G-force (g)')

        ax_d.axhline(MAX_DIST, color='blue',  linestyle='--', alpha=0.5,
                     label=f'Dist limit = {MAX_DIST} m')
        ax_g.axhline(MAX_G,   color='red',   linestyle='--', alpha=0.5,
                     label=f'G limit = {MAX_G} g')
        ax_d.axvline(opt_F, color='green', linestyle=':', linewidth=2,
                     label=f'Optimal F = {opt_F:.1f} N')

        ax_d.set_xscale('log')
        ax_d.set_xlabel('Brake Force F (N)', fontsize=12)
        ax_d.set_ylabel('Stopping Distance (m)', color='blue', fontsize=12)
        ax_g.set_ylabel('Peak G-force (g)',       color='red',  fontsize=12)
        ax_d.set_title(
            f'Brake Force Sweep at Optimal Node  (E = {opt_E} GPa,  D = {opt_D} mm)')

        # Collect all handles/labels from both axes
        h1, l1 = ax_d.get_legend_handles_labels()
        h2, l2 = ax_g.get_legend_handles_labels()
        ax_d.legend(h1 + h2, l1 + l2, loc='best', fontsize=9)
        ax_d.grid(True, alpha=0.3)
        fig2.tight_layout()

    plt.show()


def run_constrained_optimization():
    moduli_gpa    = [1.0]
    diameters_mm  = [0.5]
    # Wide log-spaced brake force range: 5 N → 500 N
    brake_forces  = np.logspace(np.log10(5), np.log10(100), 25).tolist()
    ramp_times    = [0.01, 0.05, 0.1]

    all_results = []
    total_runs  = len(moduli_gpa) * len(diameters_mm) * len(brake_forces) * len(ramp_times)
    current_run = 0

    print(f"Starting optimization: {total_runs} runs")
    print(f"  E:     {moduli_gpa} GPa")
    print(f"  D:     {diameters_mm} mm")
    print(f"  F:     {brake_forces[0]:.1f} – {brake_forces[-1]:.1f} N  ({len(brake_forces)} log-spaced)")
    print(f"  Ramp:  {ramp_times} s")
    print(f"  Constraints: G ≤ {MAX_G} g,  Dist ≤ {MAX_DIST} m\n")
    start_time = time.time()

    for E_gpa, D_mm, F, Ramp in itertools.product(
            moduli_gpa, diameters_mm, brake_forces, ramp_times):
        current_run += 1
        E = E_gpa * 1e9
        D = D_mm / 1000.0

        if current_run % max(1, total_runs // 25) == 0:
            elapsed = time.time() - start_time
            eta     = elapsed / current_run * (total_runs - current_run)
            print(f"  {current_run:>5}/{total_runs}  ({100*current_run/total_runs:.0f}%)  "
                  f"ETA {eta:.0f}s")

        peak_g, stop_dist, max_tension = run_headless_sim(F, E, Ramp, D)

        all_results.append({
            'E_gpa': E_gpa, 'D_mm': D_mm, 'F_brake': F, 'Ramp': Ramp,
            'Peak_G': peak_g, 'Dist': stop_dist, 'Max_Tension': max_tension
        })

    elapsed = time.time() - start_time
    valid   = [r for r in all_results
               if r['Peak_G'] is not None and r['Dist'] is not None
               and r['Peak_G'] <= MAX_G and r['Dist'] <= MAX_DIST]

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Valid configurations: {len(valid)} / {total_runs}")

    if valid:
        plot_results(all_results, moduli_gpa, diameters_mm)
    else:
        print("No valid configurations found — try relaxing MAX_G or MAX_DIST.")


if __name__ == "__main__":
    run_constrained_optimization()
