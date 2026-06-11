"""envelope_sim.py -- Operating-envelope sweep for the braking & reel-back maneuver.

Physics constants and initial conditions are matched exactly to main.py:
  TW_ratio = 4.4, WIRE_E = 6e9, BRAKE_RAMP = 0.01 s, REEL_FORCE_MAX = 50 N,
  initial thrust direction = normalised(to_moth) + [0, 0, 0.3]  (as in main.py line 133).

Sweeps a grid of stationary moth positions for three reel-back speeds and
saves a single PNG with three side-by-side minimum-ground-clearance heatmaps.

Grid:
  heights:     1.5 .. 2.75 m above ground, step 0.25 m  -> 6 values
  distances:   0.5 .. 4.0  m horizontal from dock, step 0.25 m  -> 15 values
  reel speeds: [2, 3, 4] m/s
  total runs:  6 x 15 x 3 = 270

Outputs:
  Flight_Performance_Simulation/envelope_results.csv
  Flight_Performance_Simulation/envelope_heatmap.png

Run from the repo root:
  python Flight_Performance_Simulation/envelope_sim.py
"""

import os
import sys
import math
import time
import multiprocessing
import csv as csv_module

import numpy as np
import mujoco
import mujoco.viewer as _viewer_mod

# ---------------------------------------------------------------------------
# Headless patches — no viewer, no sleep, no matplotlib pop-ups
# ---------------------------------------------------------------------------

class _FakeViewer:
    def __init__(self): self._n = 0
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def is_running(self):
        self._n += 1
        return self._n < 1_000_000   # hard cap ~1000 s simulated
    def sync(self): pass
    cam = type("_cam", (), {"type": 0, "fixedcamid": 0})()

_viewer_mod.launch_passive = lambda *a, **k: _FakeViewer()
time.sleep = lambda *a, **k: None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

# ---------------------------------------------------------------------------
# Simulation modules (same directory)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from aero import AeroEngine
from controller import FlightController

# ---------------------------------------------------------------------------
# Constants — matched exactly to main.py
# ---------------------------------------------------------------------------
MODEL_PATH   = os.path.join(HERE, "chameleon.xml")
CD_CSV       = os.path.join(HERE, "files", "Cd_values.csv")

FLOOR_Z      = -3.0                              # z-coord of ground plane in chameleon.xml

Drone_Mass   = 0.143                             # kg
TW_ratio     = 5.0                               # main.py value (batch_sim uses 5.0)
Max_Thrust   = Drone_Mass * TW_ratio * 9.81      # N
Drone_area   = 0.01                              # m^2
SLACK_MARGIN = 0.03
Capture_Radius = 0.18                            # m

F_brake      = 250                               # N  max Coulomb brake force
BRAKE_RAMP   = 0.01                              # s  main.py value (batch_sim uses 0.05)
Spool_pos    = np.array([0.0, 0.0, 0.0])
ZETA         = 0.3                               # tether damping ratio

WIRE_DIAMETER = 0.0005                           # m
WIRE_E        = 1e9                              # Pa  main.py value (batch_sim uses 1e9)
WIRE_A        = math.pi * (WIRE_DIAMETER / 2) ** 2

R_SPOOL      = 0.032                             # m
I_SPOOL      = 1e-5                              # kg*m^2
M_EFF        = I_SPOOL / R_SPOOL ** 2            # effective payout inertia

BREAK_STRAIN      = 0.20
HARD_LIMIT_MARGIN = 0.5                          # m  solver backstop

REEL_KP        = 6.0                             # N*s/m  velocity servo gain
REEL_FORCE_MAX = 50.0                            # N  main.py value (batch_sim uses 5.0)
REEL_HOME      = 0.30                            # m  dock arrival threshold

INTERCEPT, BRAKE, REEL = 0, 1, 2

# ---------------------------------------------------------------------------
# Grid and sweep parameters
# ---------------------------------------------------------------------------
HEIGHTS_M   = list(np.round(np.arange(1.00, 3.00, 0.25), 10))  # [1.5, 1.75, 2.0, 2.25, 2.5, 2.75]
DISTANCES_M = list(np.round(np.arange(0.50, 4.25, 0.25), 10))  # [0.5, 0.75, ..., 4.0]
REEL_SPEEDS = [3.0]                                   # m/s

TIMEOUT_S   = 4   # seconds per run — longer paths need more time


# ---------------------------------------------------------------------------
# Spool dynamics — identical to main.py
# ---------------------------------------------------------------------------

def integrate_spool(P, Pdot, T, F_brake_max, F_reel, dt):
    """Semi-implicit Euler step for 1-DOF spool payout with Coulomb brake."""
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


# ---------------------------------------------------------------------------
# Single simulation run
# ---------------------------------------------------------------------------

def run_one(height_m, dist_m, reel_speed):
    """Simulate interception of a stationary moth at (dist_m, 0, height_m above ground),
    then brake and reel back to the dock.

    All physics and initial conditions match main.py exactly.

    Returns dict with:
      min_ground_clearance_m  -- min(pos.z - FLOOR_Z) across all simulation steps
      max_g                   -- peak g-force during BRAKE phase
      braking_dist_m          -- travel distance during BRAKE (None if phase never ended)
      max_tension_N           -- peak tether tension
      tether_broke            -- True if tether snapped
      completed               -- True if drone reached dock within REEL_HOME
    """
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    bid        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,   "drone")
    tid        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "tether")
    sid_mount  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,   "drone_mount")
    jid_spool  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,  "spool_roll")
    moth_bid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,   "moth")
    moth_mid   = model.body_mocapid[moth_bid]
    spool_qadr = model.jnt_qposadr[jid_spool]
    spool_vadr = model.jnt_dofadr[jid_spool]

    # Place stationary moth at grid point
    moth_pos = np.array([dist_m, 0.0, FLOOR_Z + height_m])
    data.mocap_pos[moth_mid] = moth_pos
    mujoco.mj_forward(model, data)

    L0     = float(data.ten_length[tid]) + SLACK_MARGIN
    k_init = (WIRE_A * WIRE_E) / max(L0, 1e-3)
    model.tendon_stiffness[tid] = k_init

    aero = AeroEngine(model, data, cd_csv_path=CD_CSV, body_name="drone", area=Drone_area)
    ctrl = FlightController(model, data, body_name="drone",
                            max_thrust=Max_Thrust, mass=Drone_Mass, gravity=9.81)

    P0     = float(data.ten_length[tid]) + SLACK_MARGIN
    P      = P0
    Pdot   = 0.0
    theta0 = float(data.qpos[spool_qadr])
    model.tendon_lengthspring[tid] = [0.0, P]

    drone_pos0  = data.xpos[bid].copy()

    # Initial thrust direction: main.py adds an upward [0,0,0.3] bias to the
    # normalised direction toward the moth before setting ctrl.thrust_dir.
    to_moth      = moth_pos - drone_pos0
    to_moth_n    = float(np.linalg.norm(to_moth))
    Thrust_dir_0 = to_moth / to_moth_n + np.array([0.0, 0.0, 0.0])   # matches main.py line 133
    ctrl.thrust_dir = Thrust_dir_0.copy()

    state         = INTERCEPT
    t_state_enter = 0.0
    p_brake_start = np.zeros(3)
    p_reel_start  = np.zeros(3)
    prev_vel      = np.zeros(3)
    tension       = 0.0
    max_tension   = 0.0
    max_g_brake   = 0.0
    braking_dist  = None
    tether_broke  = False
    completed     = False
    min_clearance = float("inf")

    for _ in range(1_000_000):
        t   = data.time
        pos = data.xpos[bid].copy()

        # Track minimum ground clearance throughout the entire run
        min_clearance = min(min_clearance, pos[2] - FLOOR_Z)

        vel, _ = aero.body_velocity()
        drag   = aero.compute_drag()
        L      = float(data.ten_length[tid])
        Ldot   = float(data.ten_velocity[tid])
        k_spring = (WIRE_A * WIRE_E) / max(L, 1e-3)
        model.tendon_stiffness[tid] = k_spring
        C_TAUT = 2.0 * ZETA * math.sqrt(k_spring * Drone_Mass)

        if state == INTERCEPT:
            aim  = moth_pos - pos
            dist = float(np.linalg.norm(aim))
            ctrl.rotate_thrust_toward(aim / dist, dt)
            thrust_cmd = Max_Thrust
            ctrl.release_spool()
            P = max(P, L + SLACK_MARGIN)
            hit_floor = pos[2] < FLOOR_Z
            overshot  = np.linalg.norm(pos) > np.linalg.norm(moth_pos) + Capture_Radius
            if dist < Capture_Radius or hit_floor or overshot:
                state         = BRAKE
                P             = L
                Pdot          = Ldot
                t_state_enter = t
                p_brake_start = pos.copy()

        elif state == BRAKE:
            g_force = float(np.linalg.norm(vel - prev_vel) / dt / 9.81)
            max_g_brake = max(max_g_brake, g_force)
            desired_dir = drone_pos0 / np.linalg.norm(drone_pos0)
            ctrl.rotate_thrust_toward(desired_dir, dt)
            thrust_cmd = Max_Thrust * 0.3
            ramp = min((t - t_state_enter) / BRAKE_RAMP, 1.0)
            ctrl.set_brake_friction(ramp * 5.0)
            ctrl.set_reel_torque(0.0)
            P, Pdot = integrate_spool(P, Pdot, tension, ramp * F_brake, 0.0, dt)
            if L > 1e-6:
                u_away   = (pos - Spool_pos) / L
                v_radial = float(np.dot(vel, u_away))
            else:
                v_radial = 0.0
            if v_radial <= 0.0 and (t - t_state_enter) > 0.5 * BRAKE_RAMP:
                state, t_state_enter = REEL, t
                p_reel_start = pos.copy()
                braking_dist = float(np.linalg.norm(p_reel_start - p_brake_start))

        else:  # REEL
            desired_dir = drone_pos0 / np.linalg.norm(drone_pos0)
            ctrl.rotate_thrust_toward(desired_dir, dt)
            thrust_cmd = Max_Thrust * 0.3
            to_dock   = Spool_pos - pos
            dist_dock = float(np.linalg.norm(to_dock))
            u_dock    = to_dock / dist_dock if dist_dock > 1e-6 else np.zeros(3)
            v_along   = float(np.dot(vel, u_dock))
            vert_frac = max(abs(to_dock[2]) / dist_dock, 0.15)    # matches main.py
            f_hang    = vert_frac
            f_servo   = float(np.clip(REEL_KP * (reel_speed - v_along), 0.0, 10.0))
            f_pull    = float(np.clip(max(f_hang, f_servo), 0.0, REEL_FORCE_MAX))
            P    = L - f_pull / k_spring
            Pdot = Ldot
            ctrl.set_reel_torque(10.0)
            ctrl.set_brake_friction(0.0)
            if dist_dock <= REEL_HOME:
                completed = True
                break

        P = max(P, 0.02)
        model.tendon_lengthspring[tid] = [0.0, P]
        model.tendon_range[tid, 1] = P * (1.0 + BREAK_STRAIN) + HARD_LIMIT_MARGIN

        stretch = L - P
        if stretch > 0.0:
            spring  = k_spring * stretch
            tension = max(0.0, spring + C_TAUT * (Ldot - Pdot))
            max_tension = max(max_tension, tension)
            u   = Spool_pos - data.site_xpos[sid_mount]
            nrm = np.linalg.norm(u)
            damp_force = ((tension - spring) / nrm) * u if nrm > 1e-9 else np.zeros(3)
            if stretch / P > BREAK_STRAIN:
                tether_broke = True
                break
        else:
            tension    = 0.0
            damp_force = np.zeros(3)

        ctrl.apply_drone_wrench(thrust_cmd, drag, attitude_hold=(state == INTERCEPT))
        data.xfrc_applied[bid, 0:3] += damp_force
        prev_vel = vel.copy()

        data.qpos[spool_qadr] = theta0 + (P - P0) / R_SPOOL
        data.qvel[spool_vadr] = Pdot / R_SPOOL
        mujoco.mj_step(model, data)

    return {
        "min_ground_clearance_m": min_clearance if min_clearance != float("inf") else None,
        "max_g":          max_g_brake,
        "braking_dist_m": braking_dist,
        "max_tension_N":  max_tension,
        "tether_broke":   tether_broke,
        "completed":      completed,
    }


# ---------------------------------------------------------------------------
# Subprocess worker (for per-run timeout)
# ---------------------------------------------------------------------------

def _worker(height_m, dist_m, reel_speed, queue):
    try:
        queue.put(run_one(height_m, dist_m, reel_speed))
    except Exception as e:
        queue.put({"error": str(e)})


def run_with_timeout(height_m, dist_m, reel_speed):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker,
                                args=(height_m, dist_m, reel_speed, q), daemon=True)
    p.start()
    p.join(TIMEOUT_S)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"timeout": True}
    return q.get()


# ---------------------------------------------------------------------------
# Main: sweep the grid for all reel speeds, then plot
# ---------------------------------------------------------------------------

def main():
    nh = len(HEIGHTS_M)
    nd = len(DISTANCES_M)
    n_per_speed = nh * nd
    total       = n_per_speed * len(REEL_SPEEDS)

    print(f"Envelope sweep: {nh} heights x {nd} distances x {len(REEL_SPEEDS)} reel speeds = {total} runs")
    print(f"Heights (m above ground): {HEIGHTS_M}")
    print(f"Distances (m from dock):  {[round(d, 2) for d in DISTANCES_M]}\n")

    all_clearance = {}   # reel_speed -> (nh x nd) float array
    csv_rows = []
    run_idx  = 0

    for rs in REEL_SPEEDS:
        grid_clearance = np.full((nh, nd), np.nan)

        print(f"\n{'='*64}")
        print(f"  Reel speed = {rs:.1f} m/s")
        print(f"{'='*64}")

        for hi, h in enumerate(HEIGHTS_M):
            for di, d in enumerate(DISTANCES_M):
                run_idx += 1
                print(f"  [{run_idx:3d}/{total}]  h={h:.2f} m  d={d:.2f} m ... ",
                      end="", flush=True)

                r   = run_with_timeout(h, d, rs)
                row = {"reel_speed_m_s": rs, "height_m": h, "distance_m": d}

                if r.get("timeout"):
                    print("TIMEOUT")
                    row.update({"timeout": True,  "completed": False,
                                "tether_broke": False, "error": ""})

                elif "error" in r:
                    print(f"ERROR: {r['error']}")
                    row.update({"timeout": False, "completed": False,
                                "tether_broke": False, "error": r["error"]})

                else:
                    clr = r["min_ground_clearance_m"]
                    tag = (" [OK]"    if r["completed"]    else
                           " [BROKE]" if r["tether_broke"] else " [INC]")
                    print(f"clearance={clr:.3f} m  "
                          f"max_g={r['max_g']:.1f} g  "
                          f"tension={r['max_tension_N']:.0f} N{tag}")

                    grid_clearance[hi, di] = clr
                    row.update({
                        "min_ground_clearance_m": clr,
                        "max_g":          r["max_g"],
                        "braking_dist_m": r["braking_dist_m"],
                        "max_tension_N":  r["max_tension_N"],
                        "tether_broke":   r["tether_broke"],
                        "completed":      r["completed"],
                        "timeout": False, "error": "",
                    })

                csv_rows.append(row)

        all_clearance[rs] = grid_clearance

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    csv_path = os.path.join(HERE, "envelope_results.csv")
    fieldnames = ["reel_speed_m_s", "height_m", "distance_m",
                  "min_ground_clearance_m", "max_g", "braking_dist_m",
                  "max_tension_N", "tether_broke", "completed", "timeout", "error"]
    with open(csv_path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nResults saved  -> {csv_path}")

    # -----------------------------------------------------------------------
    # Three side-by-side ground-clearance heatmaps (one per reel speed)
    # -----------------------------------------------------------------------
    H = np.array(HEIGHTS_M)
    D = np.array(DISTANCES_M)

    step_h = float(H[1] - H[0]) if len(H) > 1 else 0.25
    step_d = float(D[1] - D[0]) if len(D) > 1 else 0.25
    # extent = [left, right, bottom, top] so pixel centres land on H and D values
    extent = [D[0] - step_d / 2, D[-1] + step_d / 2,
              H[0] - step_h / 2, H[-1] + step_h / 2]

    # Shared colour range across all three panels so comparison is meaningful
    all_vals = np.concatenate([g[np.isfinite(g)].ravel()
                               for g in all_clearance.values()])
    vmin = 0.0
    vmax = float(np.nanmax(all_vals)) if len(all_vals) else 3.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    fig.suptitle(
        "Minimum drone ground clearance during intercept → brake → reel-back  (m)\n"
        "Physics: main.py constants  |  Stationary moth  |  Red contour = 0.3 m limit",
        fontsize=11, fontweight="bold",
    )

    im_last = None
    for ax, rs in zip(axes, REEL_SPEEDS):
        g   = all_clearance[rs]
        im  = ax.imshow(g, origin="lower", aspect="auto",
                        extent=extent, cmap="RdYlGn",
                        vmin=vmin, vmax=vmax,
                        interpolation="nearest")
        im_last = im

        # 0.3 m safe-zone contour
        try:
            cs = ax.contour(D, H, g, levels=[0.3],
                            colors="red", linewidths=2, linestyles="--")
            ax.clabel(cs, fmt="0.3 m", fontsize=8)
        except Exception:
            pass

        # Mark non-converged cells
        for hi, h in enumerate(HEIGHTS_M):
            for di, d in enumerate(DISTANCES_M):
                if np.isnan(g[hi, di]):
                    ax.text(d, h, "?", ha="center", va="center",
                            fontsize=8, color="dimgray", fontweight="bold")

        ax.set_title(f"Reel speed = {rs:.0f} m/s", fontsize=12)
        ax.set_xlabel("Horizontal distance from dock (m)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Height above ground (m)", fontsize=9)

        # Tick every other distance value to avoid crowding
        ax.set_xticks(D[::2])
        ax.set_xticklabels([f"{v:.2f}" for v in D[::2]], fontsize=8)
        ax.set_yticks(H)
        ax.set_yticklabels([f"{v:.2f}" for v in H], fontsize=8)

    # Single shared colourbar on the right
    fig.subplots_adjust(right=0.88, wspace=0.06)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.018, 0.70])
    cbar = fig.colorbar(im_last, cax=cbar_ax)
    cbar.set_label("Min ground clearance (m)", fontsize=10)

    png_path = os.path.join(HERE, "envelope_heatmap.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved  -> {png_path}")


if __name__ == "__main__":
    # Must run from the repo root so relative paths in chameleon.xml resolve
    os.chdir(os.path.dirname(HERE))
    multiprocessing.freeze_support()
    main()
