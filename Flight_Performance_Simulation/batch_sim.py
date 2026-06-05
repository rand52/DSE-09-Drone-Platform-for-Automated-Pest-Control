"""batch_sim.py -- Run the flight simulation headlessly over all moth CSV files.

Collects per-run results and prints averages at the end.
Run from the repo root:  python Flight_Performance_Simulation/batch_sim.py
"""

import os
import sys
import glob
import math
import time
import multiprocessing
import numpy as np
import mujoco

TIMEOUT_S = 3   # seconds: skip a run if it hasn't finished within this time

# ---------------------------------------------------------------------------
# Headless patches (no viewer, no sleep, no matplotlib pop-ups)
# ---------------------------------------------------------------------------
import mujoco.viewer as _viewer_mod

class _FakeViewer:
    def __init__(self): self._n = 0
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def is_running(self):
        self._n += 1
        return self._n < 1_000_000   # hard cap: 1 M steps ~ 1000 s simulated
    def sync(self): pass
    cam = type("_cam", (), {"type": 0, "fixedcamid": 0})()

_viewer_mod.launch_passive = lambda *a, **k: _FakeViewer()
time.sleep = lambda *a, **k: None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

# ---------------------------------------------------------------------------
# Import simulation modules (they live in the same directory)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from moth import MothTrajectory
from aero import AeroEngine
from controller import FlightController

# ---------------------------------------------------------------------------
# Simulation constants (mirrored from main.py — change here if you tune there)
# ---------------------------------------------------------------------------
MODEL_PATH   = os.path.join(HERE, "chameleon.xml")
FILES_ROOT   = os.path.join(HERE, "files")
CD_CSV       = os.path.join(HERE, "files", "Cd_values.csv")

Drone_Mass   = 0.143
TW_ratio     = 5
Max_Thrust   = Drone_Mass * TW_ratio * 9.81
Drone_area   = 0.01
SLACK_MARGIN = 0.03
Capture_Radius = 0.18

F_brake      = 250
BRAKE_RAMP   = 0.05
Spool_pos    = [0, 0, 0]
ZETA         = 0.3

WIRE_DIAMETER = 0.0005
WIRE_E        = 2e9
WIRE_A        = math.pi * (WIRE_DIAMETER / 2) ** 2

R_SPOOL      = 0.032
I_SPOOL      = 1e-5
M_EFF        = I_SPOOL / R_SPOOL ** 2

BREAK_STRAIN      = 0.20
HARD_LIMIT_MARGIN = 0.5

REEL_SPEED    = 1.5
REEL_KP       = 6.0
REEL_FORCE_MAX = 5.0
REEL_HOME     = 0.30

INTERCEPT, BRAKE, REEL = 0, 1, 2


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


def run_one(moth_csv):
    """Run a single simulation. Returns a result dict or None if it failed."""
    try:
        moth = MothTrajectory(moth_csv, require_valid=True)
    except Exception as e:
        return {"error": str(e)}

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    bid       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,   "drone")
    tid       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "tether")
    sid_mount = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,   "drone_mount")
    jid_spool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,  "spool_roll")
    moth_bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,   "moth")
    moth_mid  = model.body_mocapid[moth_bid]
    spool_qadr = model.jnt_qposadr[jid_spool]
    spool_vadr = model.jnt_dofadr[jid_spool]

    L0       = float(data.ten_length[tid]) + SLACK_MARGIN
    k_init   = (WIRE_A * WIRE_E) / max(L0, 1e-3)
    model.tendon_stiffness[tid] = k_init

    aero = AeroEngine(model, data, cd_csv_path=CD_CSV, body_name="drone", area=Drone_area)
    ctrl = FlightController(model, data, body_name="drone",
                            max_thrust=Max_Thrust, mass=Drone_Mass, gravity=9.81)

    data.mocap_pos[moth_mid] = moth.start_pos
    mujoco.mj_forward(model, data)

    P0     = float(data.ten_length[tid]) + SLACK_MARGIN
    P      = P0
    Pdot   = 0.0
    theta0 = float(data.qpos[spool_qadr])
    model.tendon_lengthspring[tid] = [0.0, P]

    drone_pos0   = data.xpos[bid].copy()
    moth.start_pos[2] += 1.5
    to_moth      = moth.start_pos - drone_pos0
    Thrust_dir_0 = to_moth / np.linalg.norm(to_moth)
    ctrl.thrust_dir = Thrust_dir_0.copy()

    state         = INTERCEPT
    t_state_enter = 0.0
    p_brake_start = np.zeros(3)
    p_reel_start  = np.zeros(3)
    prev_vel      = np.zeros(3)
    tension       = 0.0
    max_tension   = 0.0
    max_speed     = 0.0
    max_g_brake   = 0.0
    g_force_log   = []
    braking_dist  = None
    tether_broke  = False

    for _ in range(1_000_000):
        t   = data.time
        pos = data.xpos[bid].copy()
        vel, _ = aero.body_velocity()
        speed  = float(np.linalg.norm(vel))

        drag   = aero.compute_drag()
        L      = float(data.ten_length[tid])
        Ldot   = float(data.ten_velocity[tid])
        k_spring = (WIRE_A * WIRE_E) / max(L, 1e-3)
        model.tendon_stiffness[tid] = k_spring
        C_TAUT = 2.0 * ZETA * math.sqrt(k_spring * Drone_Mass)
        moth_p = moth.position(t)
        moth_p[2] += 1.5
        data.mocap_pos[moth_mid] = moth_p

        if state == INTERCEPT:
            aim  = moth_p - pos
            dist = float(np.linalg.norm(aim))
            max_speed = max(max_speed, speed)
            ctrl.rotate_thrust_toward(aim / dist, dt)
            thrust_cmd = Max_Thrust
            ctrl.release_spool()
            P = max(P, L + SLACK_MARGIN)
            hit_floor = pos[2] < -3
            overshot  = np.linalg.norm(pos) > np.linalg.norm(moth_p) + Capture_Radius
            if dist < Capture_Radius or hit_floor or overshot:
                state = BRAKE
                P = L
                Pdot = Ldot
                t_state_enter = t
                p_brake_start = pos.copy()

        elif state == BRAKE:
            g_force = float(np.linalg.norm(vel - prev_vel) / dt / 9.81)
            max_g_brake = max(max_g_brake, g_force)
            g_force_log.append(g_force)
            desired_dir = drone_pos0 / np.linalg.norm(drone_pos0)
            ctrl.rotate_thrust_toward(desired_dir, dt)
            thrust_cmd = Max_Thrust * 0.3
            ramp = min((t - t_state_enter) / BRAKE_RAMP, 1.0)
            ctrl.set_brake_friction(ramp * 5.0)
            ctrl.set_reel_torque(0.0)
            P, Pdot = integrate_spool(P, Pdot, tension, ramp * F_brake, 0.0, dt)
            if L > 1e-6:
                u_away   = (pos - np.zeros(3)) / L
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
            to_dock   = np.array(Spool_pos) - pos
            dist_dock = float(np.linalg.norm(to_dock))
            u_dock    = to_dock / dist_dock if dist_dock > 1e-6 else np.zeros(3)
            v_along   = float(np.dot(vel, u_dock))
            f_hang    = max(abs(to_dock[2]) / dist_dock, 0.15)
            f_servo   = float(np.clip(REEL_KP * (REEL_SPEED - v_along), 0.0, 10.0))
            f_pull    = float(np.clip(max(f_hang, f_servo), 0.0, REEL_FORCE_MAX))
            P    = L - f_pull / k_spring
            Pdot = Ldot
            ctrl.set_reel_torque(10.0)
            ctrl.set_brake_friction(0.0)
            if dist_dock <= REEL_HOME:
                break

        P = max(P, 0.02)
        model.tendon_lengthspring[tid] = [0.0, P]
        model.tendon_range[tid, 1] = P * (1.0 + BREAK_STRAIN) + HARD_LIMIT_MARGIN

        stretch = L - P
        if stretch > 0.0:
            spring  = k_spring * stretch
            tension = max(0.0, spring + C_TAUT * (Ldot - Pdot))
            max_tension = max(max_tension, tension)
            u   = np.array(Spool_pos) - data.site_xpos[sid_mount]
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

        # mirror drum for consistency (not needed headless, but cheap)
        data.qpos[spool_qadr] = theta0 + (P - P0) / R_SPOOL
        data.qvel[spool_vadr] = Pdot / R_SPOOL

        mujoco.mj_step(model, data)

    return {
        "max_speed_m_s":   max_speed,
        "braking_dist_m":  braking_dist,
        "max_g":           max_g_brake,
        "avg_g_brake":     float(np.mean(g_force_log)) if g_force_log else 0.0,
        "g_force_log":     g_force_log,
        "max_tension_N":   max_tension,
        "max_tension_kgf": max_tension / 9.81,
        "tether_broke":    tether_broke,
    }


# ---------------------------------------------------------------------------
# Worker: runs in a subprocess so it can be hard-killed on timeout
# ---------------------------------------------------------------------------
def _worker(csv, queue):
    try:
        queue.put(run_one(csv))
    except Exception as e:
        queue.put({"error": str(e)})


def run_with_timeout(csv):
    """Run run_one(csv) in a subprocess; return {"timeout": True} if it exceeds TIMEOUT_S."""
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(csv, q), daemon=True)
    p.start()
    p.join(TIMEOUT_S)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"timeout": True}
    return q.get()


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------
def main():
    pattern = os.path.join(FILES_ROOT, "**", "log_itrk*.csv")
    csv_files = sorted(glob.glob(pattern, recursive=True))

    # Also include any log_itrk*.csv directly in files/
    csv_files += sorted(glob.glob(os.path.join(FILES_ROOT, "log_itrk*.csv")))
    csv_files = sorted(set(csv_files))

    print(f"Found {len(csv_files)} moth trajectory files.\n")

    results   = []
    failed    = []
    skipped   = []

    for i, csv in enumerate(csv_files):
        rel = os.path.relpath(csv, start=os.path.dirname(HERE))
        print(f"[{i+1:3d}/{len(csv_files)}] {rel} ... ", end="", flush=True)
        r = run_with_timeout(csv)
        if r.get("timeout"):
            print(f"TIMEOUT (>{TIMEOUT_S}s), skipping")
            skipped.append(rel)
            continue
        if "error" in r:
            print(f"FAILED ({r['error']})")
            failed.append(rel)
            continue
        tag = " [BROKE]" if r["tether_broke"] else ""
        bd  = f"{r['braking_dist_m']:.3f} m" if r["braking_dist_m"] is not None else "N/A"
        print(f"speed={r['max_speed_m_s']:.1f} m/s  brake_dist={bd}  "
              f"max_g={r['max_g']:.1f} g  max_T={r['max_tension_N']:.0f} N{tag}")
        r["file"] = rel
        results.append(r)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"Total runs:    {len(csv_files)}")
    print(f"Completed:     {len(results)}")
    print(f"Timed out:     {len(skipped)}  (>{TIMEOUT_S}s each)")
    print(f"Failed:        {len(failed)}")
    broke = [r for r in results if r["tether_broke"]]
    print(f"Tether broke:  {len(broke)}")

    completed = [r for r in results if not r["tether_broke"]]
    if completed:
        speeds   = [r["max_speed_m_s"]  for r in completed]
        dists    = [r["braking_dist_m"] for r in completed if r["braking_dist_m"] is not None]
        max_gs   = [r["max_g"]          for r in completed]
        avg_gs   = [r["avg_g_brake"]    for r in completed]
        tensions = [r["max_tension_N"]  for r in completed]
        all_g    = [g for r in completed for g in r["g_force_log"]]

        print(f"\n--- Averages (over {len(completed)} non-breaking runs) ---")
        print(f"Max intercept speed:  avg={np.mean(speeds):.2f}  min={np.min(speeds):.2f}  max={np.max(speeds):.2f}  m/s")
        if dists:
            print(f"Braking distance:     avg={np.mean(dists):.3f}  min={np.min(dists):.3f}  max={np.max(dists):.3f}  m")
        print(f"Peak g-force:         avg={np.mean(max_gs):.1f}  min={np.min(max_gs):.1f}  max={np.max(max_gs):.1f}  g")
        print(f"Avg g (brake phase):  avg={np.mean(avg_gs):.2f}  m/s^2")
        if all_g:
            print(f"Overall g-force list: {len(all_g)} samples  mean={np.mean(all_g):.2f} g  "
                  f"p95={np.percentile(all_g, 95):.1f} g  p99={np.percentile(all_g, 99):.1f} g")
        print(f"Max tether tension:   avg={np.mean(tensions):.1f}  max={np.max(tensions):.1f}  N")
        print(f"                      avg={np.mean(tensions)/9.81:.2f}  max={np.max(tensions)/9.81:.2f}  kgf")

    if skipped:
        print(f"\nTimed-out files (>{TIMEOUT_S}s):")
        for f in skipped:
            print(f"  {f}")
    if failed:
        print(f"\nFailed files:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    # Must run from the repo root so relative paths in chameleon.xml resolve.
    os.chdir(os.path.dirname(HERE))
    main()
