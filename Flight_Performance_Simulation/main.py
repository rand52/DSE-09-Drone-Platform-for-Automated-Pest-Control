#imports
import mujoco
import numpy as np
from moth import load_moth_track, moth_position
import mujoco.viewer
import time
from controller import run_controller
from moth import MothTrajectory

from aero import AeroEngine


#Loading drone model and moth path
Model_path = "Flight_Performance_Simulation\chameleon.xml"
Moth_Log = "log_itrk3.csv"


# Drone parameters
Drone_Mass = 0.143 #grams
TW_ratio = 4 #Thrust to weight ratio
Max_Thrust = Drone_Mass*TW_ratio*9.81 #newtons
Drone_area = 0.01 #m^2 FIND VALUES
Drone_Cd = 1.0 #Coefficient of drag FIND VALUES
Drone_pitch_rate = 30 # degrees/s
SLACK_MARGIN = 0.03


capture_radius = 0.18 #meter TODO: check if neccesary

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
    aero = AeroEngine(model, data, body_name="drone", Cd=Drone_Cd, area=Drone_area)


    data.mocap_pos[moth_mid] = moth.start_pos #Place moth at start 
    mujoco.mj_forward(model, data) #calculates all kinematics TODO Check if neccearry 
    P0 = float(data.ten_length[tid]) + SLACK_MARGIN
    P  = P0
    model.tendon_lengthspring[tid] = [0.0, P] # models the initial length of the tether

    drone_pos0  = data.xpos[bid].copy() #Initial drone position
    to_moth     = moth.start_pos - drone_pos0 #Vector from drone to moth 
    to_moth_n   = float(np.linalg.norm(to_moth)) # Length of set vector
    Thrust_dir_0 = to_moth/to_moth_n #Unit vector towards moth and initial vector
    
    #Loop bookkeeping
    state = INTERCEPT

    def step_logic():
        nonlocal state, P

        t = data.time
        pos = data.xpos[bid].copy
        vel = aero.body_velocity() #Runs velocity calculations 
        speed = float(np.linalg.norm(vel)) #length of vel
        
        drag     = aero.compute_drag()
        L        = float(data.ten_length[tid])
        Ldot     = float(data.ten_velocity[tid])

        if state == INTERCEPT:
            target = moth.position(t) #Location of the moth 
            aim = target - pos #Vector form drone to moth 
            dist = float(np.linalg.norm(aim)) #Distance between drone and moth

            target_vector = aim/dist

        if state == BRAKE:
            

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and state != REEL and t < 100.0:
            step_start = time.time()
            step_logic()
            mujoco.mj_step(model, data)
            t += dt
            viewer.sync()
            elapsed = time.time() - step_start
            time.sleep(max(0, dt - elapsed))

        


main()