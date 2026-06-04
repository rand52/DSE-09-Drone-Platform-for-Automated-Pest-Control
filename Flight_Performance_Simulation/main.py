#imports
import mujoco
import numpy as np
import mujoco.viewer
import time
from moth import MothTrajectory

from aero import AeroEngine
from controller import FlightController



#Loading drone model and moth path
Model_path = r"Flight_Performance_Simulation\chameleon.xml"
Moth_Log = "log_itrk5.csv"


# Drone parameters
Drone_Mass = 0.143 #kg
TW_ratio = 3 #Thrust to weight ratio
Max_Thrust = Drone_Mass*TW_ratio*9.81 #newtons
Drone_area = 0.01 #m^2 FIND VALUES
#Drone_Cd = 1.0 #Coefficient of drag FIND VALUES
Drone_pitch_rate = 30 # degrees/s
SLACK_MARGIN = 0.03
Capture_Radius = 0.18 #meter TODO: check if neccesary
F_brake = 10 # Newtons
BRAKE_RAMP = 0.1 #seconds
Spool_pos = [0,0,0]  
C_TAUT = 0.0  # FIX 1: was undefined — tether damping coefficient, set to 0 as placeholder TODO: tune value

#Reel in settings
REEL_SPEED     = 3.0        # reel-in target speed [m/s]
REEL_KP        = 6.0        # reel velocity-servo gain [N/(m/s)]
REEL_FORCE_MAX = 25.0       # max reel pull force [N]
REEL_HOME      = 0.30


SLOW_MO = 8.0  # 1.0 = real time, 4.0 = 4x slower, 0.5 = 2x faster

INTERCEPT, BRAKE, REEL = 0, 1, 2
STATE_NAMES = {INTERCEPT: "INTERCEPT", BRAKE: "BRAKE", REEL: "REEL"}


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
    k_spring    = abs(float(model.tendon_solref_lim[tid][0]))
    model.tendon_stiffness[tid] = k_spring
    aero = AeroEngine(model, data, cd_csv_path=r"Cd_values.csv", body_name="drone", area=Drone_area)
    ctrl = FlightController(model, data, body_name="drone",
                            max_thrust=Max_Thrust, mass=Drone_Mass, gravity=9.81)

    data.mocap_pos[moth_mid] = moth.start_pos #Place moth at start
    mujoco.mj_forward(model, data) #calculates all kinematics TODO Check if neccearry
    P0 = float(data.ten_length[tid]) + SLACK_MARGIN
    P  = P0
    model.tendon_lengthspring[tid] = [0.0, P] # models the initial length of the tether

    drone_pos0  = data.xpos[bid].copy() #Initial drone position
    moth.start_pos[2] += 1.5
    to_moth     = moth.start_pos - drone_pos0 #Vector from drone to moth
    print(moth.start_pos)
    print(to_moth)
    to_moth_n   = float(np.linalg.norm(to_moth)) # Length of set vector
    Thrust_dir_0 = to_moth/to_moth_n #Unit vector towards moth and initial vector
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

    def step_logic():
        nonlocal state, P, t_state_enter, p_brake_start, p_reel_start, max_speed_intercept, max_g_brake, prev_vel

        t = data.time
        pos = data.xpos[bid].copy()
        vel, _ = aero.body_velocity() #Runs velocity calculations (world frame)
        speed = float(np.linalg.norm(vel)) #length of vel

        drag     = aero.compute_drag()
        L        = float(data.ten_length[tid])
        Ldot     = float(data.ten_velocity[tid])
        moth_p = moth.position(t)
        moth_p[2] += 1.5

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
                t_state_enter = t
                p_brake_start = pos.copy()
                print(f"Max speed during intercept: {max_speed_intercept:.2f} m/s")

        elif state == BRAKE:
            g_force = float(np.linalg.norm(vel - prev_vel) / dt / 9.81)
            max_g_brake = max(max_g_brake, g_force)
            ## Thrust becomes 0
            desired_dir = drone_pos0/np.linalg.norm(drone_pos0) #Vector from spool to drone
            ctrl.rotate_thrust_toward(np.array(desired_dir), dt)  # FIX 3: dt was inside np.array(), moved outside as separate argument
            thrust_cmd = Max_Thrust #TODO try if thrust reduces the bouncing

            ramp = min((t - t_state_enter) / BRAKE_RAMP, 1.0)
            ctrl.set_brake_friction(ramp * 5.0)
            ctrl.set_reel_torque(0.0)

            P_slip = L - (ramp*F_brake) / k_spring
            P = min(P, P_slip)  # allow P to slip downward under braking force

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
            thrust_cmd = Max_Thrust #TODO change this to see the difference it makes

            to_dock = Spool_pos - pos #vector from drone to dock
            dist_dock =  float(np.linalg.norm(to_dock))
            u_dock    = to_dock / dist_dock if dist_dock > 1e-6 else np.zeros(3)
            v_along   = float(np.dot(vel, u_dock))
            vert_frac  = max(abs(to_dock[2]) / dist_dock, 0.15)
            f_hang     = vert_frac
            f_servo = float(np.clip(REEL_KP * (REEL_SPEED - v_along), 0.0, 10.0))
            f_pull = float(np.clip(max(f_hang, f_servo), 0.0, REEL_FORCE_MAX))
            P = L - f_pull / k_spring
            ctrl.set_reel_torque(10.0)
            ctrl.set_brake_friction(0.0)
            if dist_dock <= REEL_HOME:
                return

        P = max(P, 0.02)
        model.tendon_lengthspring[tid] = [0.0, P]
        model.tendon_range[tid, 1] = P  # update hard constraint so MuJoCo enforces the new max length

        stretch = L - P
        if stretch > 0.0:
            spring  = k_spring * stretch
            tension = max(0.0, spring + C_TAUT * Ldot)
            u       = Spool_pos - data.site_xpos[sid_mount]  # FIX 6: was spool_pos (undefined) — matches global Spool_pos
            nrm     = np.linalg.norm(u)
            damp_force = ((tension - spring) / nrm) * u if nrm > 1e-9 else np.zeros(3)
        else:
            tension    = 0.0
            damp_force = np.zeros(3)


        ctrl.apply_drone_wrench(thrust_cmd, drag, attitude_hold=(state == INTERCEPT))
        data.xfrc_applied[bid, 0:3] += damp_force
        prev_vel = vel.copy()

    def reset():
        nonlocal state, P, t_state_enter, t, max_speed_intercept, max_g_brake, prev_vel
        mujoco.mj_resetData(model, data)
        data.mocap_pos[moth_mid] = moth.start_pos
        mujoco.mj_forward(model, data)
        P = float(data.ten_length[tid]) + SLACK_MARGIN
        model.tendon_lengthspring[tid] = [0.0, P]
        model.tendon_range[tid, 1] = P
        ctrl.thrust_dir = Thrust_dir_0.copy()
        state = INTERCEPT
        t_state_enter = 0.0
        t = 0.0
        max_speed_intercept = 0.0
        max_g_brake         = 0.0
        prev_vel            = np.zeros(3)

    def key_callback(keycode):
        if keycode == ord('R'):
            reset()

    t = 0.0
    print(f"The braking started at {p_brake_start}")
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running() and t < 100.0:  # FIX 7: removed state != REEL — loop was exiting immediately when REEL began, so reel-in never ran
            step_start = time.time()
            result = step_logic()  # FIX 8: capture return value so docking (return False) stops the loop
            if result is False:
                break
            mujoco.mj_step(model, data)
            t += dt
            viewer.sync()
            elapsed = time.time() - step_start
            time.sleep(max(0, dt * SLOW_MO - elapsed))




main()