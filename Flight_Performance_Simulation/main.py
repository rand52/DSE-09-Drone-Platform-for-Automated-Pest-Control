#imports
import math
import mujoco
import numpy as np
import mujoco.viewer
import time
import csv
import matplotlib.pyplot as plt
from moth import MothTrajectory

from aero import AeroEngine
from controller import FlightController



#Loading drone model and moth path
Model_path = r"Flight_Performance_Simulation\chameleon.xml"
Moth_Log = r"Flight_Performance_Simulation\\files\\log_itrk3.csv"


# Drone parameters
Drone_Mass = 0.160 #kg
TW_ratio = 4#Thrust to weight ratio
Max_Thrust = Drone_Mass*TW_ratio*9.81 #newtons
diameter = 96.72 * 10**-3
Drone_area = (math.pi*diameter**2)/4 #m^2 TODO: 
#Drone_Cd = 1.0 #Coefficient of drag FIND VALUES
Drone_pitch_rate = 30 # degrees/s
SLACK_MARGIN = 0.03
Capture_Radius = 0.18 #meter TODO: check if neccesary
F_brake = 142 # N  max tether force thSe spool brake can hold before it slips (Coulomb).
BRAKE_RAMP = 0.03 #seconds  time to ramp the brake from 0 to full
Spool_pos = [0,0,0]
# Tether axial damping. C_TAUT is recomputed each step as a fraction of critical damping
# (C = 2*ZETA*sqrt(k*m)); ZETA < 1 -> underdamped. Tune ZETA once we have a target response.
C_TAUT = 0.0   # recomputed every step in step_logic()
ZETA   = 0.02   # tether damping ratio (underdamped placeholder — TODO tune)

# Wire properties for length-dependent stiffness k = AE/L
WIRE_DIAMETER    = 0.0005                                # m  (0.5 mm — verify)
WIRE_E           = 2e9                                   # Pa (Nylon, 3 GPa — verify from datasheet)
WIRE_A           = math.pi * (WIRE_DIAMETER / 2) ** 2    # m^2 cross-section area
#WIRE_ETA           = 0.1

# ------------------------------------------------------------------------------------------
# Spool payout model (Python).  R_SPOOL is the single design knob; if the drum is redesigned
# (e.g. in CATIA) change R_SPOOL here and keep drum_geom radius in chameleon.xml consistent.
# The spool is modelled as an equivalent linear payout element of effective mass m_eff:
#       m_eff * Pddot = T - F_brake - F_reel          (P = deployed tether length)
# Reflected inertia felt by the drone = I_SPOOL / R_SPOOL**2 -> keep << Drone_Mass.
# ------------------------------------------------------------------------------------------
R_SPOOL = 0.035                         # m   spool drum radius (== drum_geom radius in chameleon.xml)
I_SPOOL = 0.0001376                          # kg*m^2  spool rotational inertia (reflected ~0.01 kg << drone)
M_EFF   = I_SPOOL / R_SPOOL**2          # kg  effective payout inertia reflected to the tether

# Tether failure: nylon snaps near ~20% strain. With WIRE_E=3 GPa that is a break stress of
# E*eps = 600 MPa -> break force ~ stress*WIRE_A ~ 118 N, in line with real 0.5 mm nylon mono.
# To target a specific break force instead: BREAK_STRAIN = break_force / (WIRE_E * WIRE_A).
BREAK_STRAIN      = 0.2                          # -  engineering strain at which the tether breaks
HARD_LIMIT_MARGIN = 0.5                                  # m  solver limit backstop (never reached: break fires first)

#Reel in settings
REEL_SPEED     = 3.5       # reel-in target speed [m/s]
REEL_KP        = 6.0        # reel velocity-servo gain [N/(m/s)]
REEL_FORCE_MAX = 50.0       # max reel pull force [N] 
REEL_HOME      = 0.30


SLOW_MO  = 8    # 1.0 = real time, 4.0 = 4x slower, 0.5 = 2x faster

INTERCEPT, BRAKE, REEL = 0, 1, 2
STATE_NAMES = {INTERCEPT: "INTERCEPT", BRAKE: "BRAKE", REEL: "REEL"}


def integrate_spool(P, Pdot, T, F_brake_max, F_reel, dt):
    """Advance the 1-DOF spool payout one step (semi-implicit Euler with Coulomb friction).

    Linear payout model: M_EFF * Pddot = F_drive - friction, where
      F_drive = T - F_reel   (tether tension pays out / raises P; reel force winds it in)
      friction is a Coulomb force of magnitude <= F_brake_max opposing motion. It can bring
      Pdot to zero (stiction / locked spool) but never reverse it within a step.
    Returns (P_new, Pdot_new).
    """
    F_drive   = T - F_reel
    Pdot_tent = Pdot + (F_drive / M_EFF) * dt          # velocity from drive force
    dv_fric   = (F_brake_max / M_EFF) * dt             # max velocity the brake can shed this step
    if Pdot_tent > 0.0:
        Pdot_new = max(0.0, Pdot_tent - dv_fric)
    elif Pdot_tent < 0.0:
        Pdot_new = min(0.0, Pdot_tent + dv_fric)
    else:
        Pdot_new = 0.0
    return P + Pdot_new * dt, Pdot_new


def main():
    model = mujoco.MjModel.from_xml_path(Model_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep # Timestep of the simulation

    moth = MothTrajectory(Moth_Log)

    bid         = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,   "drone")
    tid         = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "tether")
    sid_center  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,   "tether_anchor")
    sid_mount   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,   "drone_mount")
    jid_spool   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,  "spool_roll")
    moth_bid    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,   "moth")
    moth_mid    = model.body_mocapid[moth_bid]
    spool_qadr  = model.jnt_qposadr[jid_spool]
    spool_vadr  = model.jnt_dofadr[jid_spool]
    L0          = float(data.ten_length[tid]) + SLACK_MARGIN   # initial deployed length
    k_spring    = (WIRE_A * WIRE_E) / max(L0, 1e-3)            # AE/L at startup
    model.tendon_stiffness[tid] = k_spring
    aero = AeroEngine(model, data, cd_csv_path=r"Flight_Performance_Simulation\files\Cd_values.csv", body_name="drone", area=Drone_area)
    ctrl = FlightController(model, data, body_name="drone",
                            max_thrust=Max_Thrust, mass=Drone_Mass, gravity=9.81)

    data.mocap_pos[moth_mid] = moth.start_pos #Place moth at start
    mujoco.mj_forward(model, data) #calculates all kinematics TODO Check if neccearry
    P0 = float(data.ten_length[tid]) + SLACK_MARGIN
    P  = P0
    Pdot   = 0.0                              # spool payout rate [m/s] (Python spool state)
    theta0 = float(data.qpos[spool_qadr])     # drum angle at startup (for visual mirroring)
    model.tendon_lengthspring[tid] = [0.0, P] # models the initial length of the tether

    drone_pos0  = data.xpos[bid].copy() #Initial drone position
    moth.start_pos[2] += 1.5
    #moth.start_pos[1] = 3
    #moth.start_pos[0] = 3
    to_moth     = moth.start_pos - drone_pos0 #Vector from drone to moth
    print(moth.start_pos)
    print(to_moth)
    to_moth_n   = float(np.linalg.norm(to_moth)) # Length of set vector
    Thrust_dir_0 = to_moth/to_moth_n + [0,0,0.3] #Unit vector towards moth and initial vector
    ctrl.thrust_dir = Thrust_dir_0.copy()
    print(Thrust_dir_0)
    #Loop bookkeeping
    state = INTERCEPT
    t_state_enter = 0.0
    p_brake_start = np.zeros(3)
    p_reel_start  = np.zeros(3)
    max_speed_intercept = 0.0
    max_g_brake         = 0.0
    prev_vel            = np.zeros(3)
    g_force_log         = []  # (time, g_force) during BRAKE
    tension             = 0.0  # last-step tether tension, feeds the spool model
    max_tension         = 0.0  # peak tether tension over the run [N]
    trajectory_log      = []   # (t, x, y, z, qw, qx, qy, qz) per step

    def step_logic():
        nonlocal state, P, Pdot, tension, max_tension, t_state_enter, p_brake_start, p_reel_start, max_speed_intercept, max_g_brake, prev_vel, g_force_log, trajectory_log

        t = data.time
        pos = data.xpos[bid].copy()
        quat = data.xquat[bid].copy()  # [w, x, y, z]
        trajectory_log.append((t, pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]))
        vel, _ = aero.body_velocity() #Runs velocity calculations (world frame)
        speed = float(np.linalg.norm(vel)) #length of vel

        drag     = aero.compute_drag()
        L        = float(data.ten_length[tid])
        Ldot     = float(data.ten_velocity[tid])
        k_spring = (WIRE_A * WIRE_E) / max(L, 1e-3)   # update stiffness: k = AE/L
        model.tendon_stiffness[tid] = k_spring
        C_TAUT = 2.0 * ZETA * math.sqrt(k_spring * Drone_Mass)  # fraction of critical damping
        # Calculate C directly from the material viscosity, area, and length
        C_TAUT = (WIRE_A * WIRE_ETA) / L
        moth_p = moth.position(t)
        moth_p[2] += 1.5 
        #moth_p[1]= 3
        #moth_p[0] = 3

        data.mocap_pos[moth_mid] = moth_p

        if state == INTERCEPT:
            target = moth_p #Location of the moth
            aim = target - pos #Vector form drone to moth
            dist = float(np.linalg.norm(aim)) #Distance between drone and moth

            max_speed_intercept = max(max_speed_intercept, speed)

            target_vector = aim/dist
            ctrl.rotate_thrust_toward(target_vector, dt)
            thrust_cmd = Max_Thrust
            ctrl.release_spool()
            P = max(P, L + SLACK_MARGIN)
            hit_floor = pos[2] < -3
            overshot  = np.linalg.norm(pos) > np.linalg.norm(moth_p) + Capture_Radius
            if dist < Capture_Radius or hit_floor or overshot:
                state = BRAKE
                P = L
                Pdot = Ldot          # spool initially pays out with the drone (no tension jump)
                t_state_enter = t
                p_brake_start = pos.copy()
                print(f"Max speed during intercept: {max_speed_intercept:.2f} m/s")

        elif state == BRAKE:
            g_force = float(np.linalg.norm(vel - prev_vel) / dt / 9.81)
            max_g_brake = max(max_g_brake, g_force)
            g_force_log.append((t, g_force))
            ## Thrust becomes 0
            desired_dir = drone_pos0/np.linalg.norm(drone_pos0) #Vector from spool to drone
            ctrl.rotate_thrust_toward(np.array(desired_dir), dt)  # FIX 3: dt was inside np.array(), moved outside as separate argument
            thrust_cmd = Max_Thrust  * 0.3#TODO try if thrust reduces the bouncing

            ramp = min((t - t_state_enter) / BRAKE_RAMP, 1.0)
            ctrl.set_brake_friction(ramp * 5.0)   # cosmetic: actuator ctrl mirrors brake engagement
            ctrl.set_reel_torque(0.0)

            # Coulomb spool brake: holds up to (ramp*F_brake) of tether tension before slipping.
            # `tension` is from the previous step; integrated in Python (no MuJoCo-solver coupling).
            P, Pdot = integrate_spool(P, Pdot, tension, ramp * F_brake, 0.0, dt)

            if L > 1e-6:
                u_away   = (pos - [0,0,0]) / L
                v_radial = float(np.dot(vel, u_away))
            else:
                v_radial = 0.0
            if v_radial <= 0.0 and (t - t_state_enter) > 0.5 * BRAKE_RAMP:
                print(f"started braking at {t} seconds")
                state, t_state_enter = REEL, t
                p_reel_start = pos.copy()
                braking_dist = float(np.linalg.norm(p_reel_start - p_brake_start))
                print(f"Braking distance: {braking_dist:.3f} m")
                print(f"Max g-force during braking: {max_g_brake:.2f} g")
        else: #Reeling
            desired_dir = drone_pos0/np.linalg.norm(drone_pos0)
            ctrl.rotate_thrust_toward(desired_dir, dt)
            thrust_cmd = Max_Thrust * 0.3 #TODO change this to see the difference it makes

            to_dock = Spool_pos - pos #vector from drone to dock
            dist_dock =  float(np.linalg.norm(to_dock))
            u_dock    = to_dock / dist_dock if dist_dock > 1e-6 else np.zeros(3)
            v_along   = float(np.dot(vel, u_dock))
            vert_frac  = max(abs(to_dock[2]) / dist_dock, 0.15)
            f_hang     = vert_frac
            f_servo = float(np.clip(REEL_KP * (REEL_SPEED - v_along), 0.0, 10.0))
            f_pull = float(np.clip(max(f_hang, f_servo), 0.0, REEL_FORCE_MAX))
            # Reel motor is an actively-controlled winch (not a free spool): set the rest length so
            # the spring tension equals the commanded pull f_pull. Pdot tracks the drone so the
            # stretch-rate damping stays ~0 during the controlled reel-in.
            P    = L - f_pull / k_spring
            Pdot = Ldot
            ctrl.set_reel_torque(10.0)            # cosmetic: actuator ctrl mirrors reel-in
            ctrl.set_brake_friction(0.0)
            if dist_dock <= REEL_HOME:
                return

        P = max(P, 0.02)
        model.tendon_lengthspring[tid] = [0.0, P]
        # Solver limit sits beyond the break strain, so it never fires: the Python break below
        # always triggers first.
        model.tendon_range[tid, 1] = P * (1.0 + BREAK_STRAIN) + HARD_LIMIT_MARGIN

        stretch = L - P
        if stretch > 0.0:
            spring  = k_spring * stretch
            # Damping acts on the true elongation rate (Ldot - Pdot): when the spool pays out with
            # the drone there is no stretch rate, so the brake (not the damper) governs braking.
            tension = max(0.0, spring + C_TAUT * (Ldot - Pdot))
            max_tension = max(max_tension, tension)
            u       = Spool_pos - data.site_xpos[sid_mount]  # FIX 6: was spool_pos (undefined) — matches global Spool_pos
            nrm     = np.linalg.norm(u)
            damp_force = ((tension - spring) / nrm) * u if nrm > 1e-9 else np.zeros(3)
            # Tether failure: nylon snaps past BREAK_STRAIN -> stop the sim.
            if stretch / P > BREAK_STRAIN:
                print(f"TETHER BROKE: strain {stretch / P:.2f} > {BREAK_STRAIN:.2f} "
                      f"(tension {tension:.0f} N) at t={t:.3f}s")
                return False
        else:
            tension    = 0.0
            damp_force = np.zeros(3)


        ctrl.apply_drone_wrench(thrust_cmd, drag, attitude_hold=(state == INTERCEPT))
        data.xfrc_applied[bid, 0:3] += damp_force
        prev_vel = vel.copy()

    def reset():
        nonlocal state, P, Pdot, theta0, tension, t_state_enter, t, max_speed_intercept, max_g_brake, prev_vel, g_force_log
        mujoco.mj_resetData(model, data)
        data.mocap_pos[moth_mid] = moth.start_pos
        mujoco.mj_forward(model, data)
        P = float(data.ten_length[tid]) + SLACK_MARGIN
        Pdot   = 0.0
        theta0 = float(data.qpos[spool_qadr])
        tension = 0.0
        model.tendon_lengthspring[tid] = [0.0, P]
        model.tendon_range[tid, 1] = P * (1.0 + BREAK_STRAIN) + HARD_LIMIT_MARGIN
        ctrl.thrust_dir = Thrust_dir_0.copy()
        state = INTERCEPT
        t_state_enter = 0.0
        t = 0.0
        max_speed_intercept = 0.0
        max_g_brake         = 0.0
        prev_vel            = np.zeros(3)
        g_force_log         = []

    paused = False

    def key_callback(keycode):
        nonlocal paused
        if keycode == ord('R'):
            reset()
        elif keycode == 32:  # Spacebar
            paused = not paused

    t = 0.0
    print(f"The braking started at {p_brake_start}")
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running() and t < 100:
            step_start = time.time()
            if not paused:
                result = step_logic()  # FIX 8: capture return value so docking (return False) stops the loop
                if result is False:
                    break
                mujoco.mj_step(model, data)
                # Mirror the Python spool state onto the drum joint so it visibly spins in sync.
                data.qpos[spool_qadr] = theta0 + (P - P0) / R_SPOOL
                data.qvel[spool_vadr] = Pdot / R_SPOOL
                t += dt
            viewer.sync()
            elapsed = time.time() - step_start
            time.sleep(max(0, dt * SLOW_MO - elapsed))

    csv_path = r"Flight_Performance_Simulation\drone_trajectory.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "z", "qw", "qx", "qy", "qz"])
        writer.writerows(trajectory_log)
    print(f"Trajectory saved to {csv_path} ({len(trajectory_log)} frames)")

    if g_force_log:
        times, gforces = zip(*g_force_log)
        plt.figure()
        plt.plot(times, gforces)
        plt.xlabel("Time (s)")
        plt.ylabel("G-force (g)")
        plt.title("G-force during braking")
        plt.grid(True)
        plt.show()


main()