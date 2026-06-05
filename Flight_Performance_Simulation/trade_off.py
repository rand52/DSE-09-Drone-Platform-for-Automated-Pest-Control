import math
import mujoco
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from moth import MothTrajectory
from aero import AeroEngine
from controller import FlightController

# --- File Paths & Fixed Parameters ---
Model_path = r"Flight_Performance_Simulation\chameleon.xml"
Moth_Log = r"Flight_Performance_Simulation\files\log_itrk3.csv"

Drone_Mass = 0.143 
Max_Thrust = Drone_Mass * 5 * 9.81 
Drone_area = 0.01 
SLACK_MARGIN = 0.03
Capture_Radius = 0.18 
R_SPOOL = 0.032
I_SPOOL = 1e-5
M_EFF = I_SPOOL / R_SPOOL**2
ZETA = 0.3
INTERCEPT, BRAKE, REEL = 0, 1, 2

# --- CONSTRAINTS ---
MAX_G = 26.0       # g
MAX_DIST = 0.6     # meters

def integrate_spool(P, Pdot, T, F_brake_max, F_reel, dt):
    """EXACT 1:1 copy of the Group 9 spool integration math."""
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
    model = mujoco.MjModel.from_xml_path(Model_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    wire_a = math.pi * (wire_d_test / 2.0) ** 2

    moth = MothTrajectory(Moth_Log)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "tether")
    sid_mount = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "drone_mount")
    moth_mid = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moth")]
    
    L0 = float(data.ten_length[tid]) + SLACK_MARGIN
    k_spring = (wire_a * wire_e_test) / max(L0, 1e-3)
    model.tendon_stiffness[tid] = k_spring
    
    aero = AeroEngine(model, data, cd_csv_path=r"Flight_Performance_Simulation\files\Cd_values.csv", body_name="drone", area=Drone_area)
    ctrl = FlightController(model, data, body_name="drone", max_thrust=Max_Thrust, mass=Drone_Mass, gravity=9.81)

    data.mocap_pos[moth_mid] = moth.start_pos
    mujoco.mj_forward(model, data)
    
    P = float(data.ten_length[tid]) + SLACK_MARGIN
    Pdot = 0.0
    model.tendon_lengthspring[tid] = [0.0, P]
    
    drone_pos0 = data.xpos[bid].copy()
    moth_start = moth.start_pos.copy()
    moth_start[2] += 1.5
    
    to_moth = moth_start - drone_pos0
    ctrl.thrust_dir = to_moth / np.linalg.norm(to_moth)

    state = INTERCEPT
    t_state_enter = 0.0
    p_brake_start = np.zeros(3)
    max_g_brake = 0.0
    max_tension = 0.0
    prev_vel = np.zeros(3)
    tension = 0.0
    t = 0.0
    timeout = 10.0 

    while t < timeout:
        pos = data.xpos[bid].copy()
        vel, _ = aero.body_velocity()
        L = float(data.ten_length[tid])
        Ldot = float(data.ten_velocity[tid])
        
        k_spring = (wire_a * wire_e_test) / max(L, 1e-3) 
        model.tendon_stiffness[tid] = k_spring
        C_TAUT = 2.0 * ZETA * math.sqrt(k_spring * Drone_Mass)
        
        moth_p = moth.position(t)
        moth_p[2] += 1.5
        data.mocap_pos[moth_mid] = moth_p

        if state == INTERCEPT:
            aim = moth_p - pos
            dist = float(np.linalg.norm(aim))
            ctrl.rotate_thrust_toward(aim/dist, dt)
            thrust_cmd = Max_Thrust
            ctrl.release_spool()
            P = max(P, L + SLACK_MARGIN)
            
            if dist < Capture_Radius or pos[2] < -3 or np.linalg.norm(pos) > np.linalg.norm(moth_p) + Capture_Radius:
                state = BRAKE
                P = L
                Pdot = Ldot
                t_state_enter = t
                p_brake_start = pos.copy()

        elif state == BRAKE:
            g_force = float(np.linalg.norm(vel - prev_vel) / dt / 9.81)
            # Ignore the first 3 ticks of braking to avoid solver artifact spikes
            if (t - t_state_enter) > 3 * dt:
                max_g_brake = max(max_g_brake, g_force)
                
            desired_dir = drone_pos0/np.linalg.norm(drone_pos0)
            ctrl.rotate_thrust_toward(desired_dir, dt)
            thrust_cmd = Max_Thrust
            
            ramp = min((t - t_state_enter) / brake_ramp_test, 1.0)
            
            P, Pdot = integrate_spool(P, Pdot, tension, ramp * f_brake_test, 0.0, dt)
            
            u_away = (pos - [0,0,0]) / L if L > 1e-6 else np.zeros(3)
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
            u       = [0,0,0] - data.site_xpos[sid_mount] 
            nrm     = np.linalg.norm(u)
            damp_force = ((tension - spring) / nrm) * u if nrm > 1e-9 else np.zeros(3)
        else:
            tension    = 0.0
            damp_force = np.zeros(3)

        ctrl.apply_drone_wrench(thrust_cmd, aero.compute_drag(), attitude_hold=(state == INTERCEPT))
        data.xfrc_applied[bid, 0:3] += damp_force
        prev_vel = vel.copy()

        try:
            mujoco.mj_step(model, data)
        except Exception:
            return None, None, max_tension

        t += dt

    return None, None, max_tension 

def plot_3d_valid_space(results, best_config):
    """Generates a smooth Pareto frontier plot."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    dists = [r['Dist'] for r in results]
    gs = [r['Peak_G'] for r in results]
    fs = [r['F_brake'] for r in results]

    # Color by Brake Force to show the transition from 'locked' to 'slipping'
    scatter = ax.scatter(dists, gs, fs, c=fs, cmap='magma', s=60, alpha=0.7)
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Brake Force (N)')

    ax.set_xlabel('Stopping Distance (m)')
    ax.set_ylabel('Peak G-Force (g)')
    ax.set_zlabel('Brake Force (N)')
    ax.set_title('Pareto Frontier: Braking Performance')
    
    plt.show()

# Update this line inside run_constrained_optimization:
brake_forces = np.linspace(10, 80, 20)

def run_constrained_optimization():
    moduli_gpa = [1.0, 1.5, 2.0]          
    diameters_mm = [0.2, 0.3, 0.4, 0.5]        
    brake_forces = np.linspace(80, 250, 15)   
    ramp_times = [0.01, 0.03, 0.05, 0.1]         

    valid_results = []
    total_runs = len(moduli_gpa) * len(diameters_mm) * len(brake_forces) * len(ramp_times)
    current_run = 0

    print(f"Starting Visualized Optimization: {total_runs} permutations...")
    start_time = time.time()

    for E_gpa, D_mm, F, Ramp in itertools.product(moduli_gpa, diameters_mm, brake_forces, ramp_times):
        current_run += 1
        E = E_gpa * 1e9
        D = D_mm / 1000.0
        
        if current_run % max(1, int(total_runs / 10)) == 0:
            print(f"Progress: {current_run}/{total_runs} runs completed...")

        peak_g, stop_dist, max_tension = run_headless_sim(F, E, Ramp, D)
        
        if stop_dist is None or peak_g > MAX_G or stop_dist > MAX_DIST:
            continue
            
        valid_results.append({
            'E_gpa': E_gpa,
            'D_mm': D_mm,
            'F_brake': F,
            'Ramp': Ramp,
            'Peak_G': peak_g,
            'Dist': stop_dist,
            'Max_Tension': max_tension
        })

    print(f"\n--- OPTIMIZATION COMPLETE in {time.time() - start_time:.1f}s ---")
    print(f"Configurations satisfying constraints (G <= {MAX_G}, Dist <= {MAX_DIST}m): {len(valid_results)}")
    
    if valid_results:
        # Objective: minimize stopping distance
        valid_results.sort(key=lambda x: x['Dist'])
        best = valid_results[0]
        
        print("\n🏆 OPTIMAL CONFIGURATION (Minimum Distance) 🏆")
        print(f"Diameter:       {best['D_mm']} mm")
        print(f"Modulus (E):    {best['E_gpa']} GPa")
        print(f"Brake Force:    {best['F_brake']:.1f} N")
        print(f"Ramp Time:      {best['Ramp']:.2f} s")
        print("-" * 30)
        print(f"Stopping Dist:  {best['Dist']:.3f} m")
        print(f"Peak Decel:     {best['Peak_G']:.1f} g")
        print(f"Max Tension:    {best['Max_Tension']:.1f} N")
        
        # Trigger the 3D graph
        plot_3d_valid_space(valid_results, best)
    else:
        print("\n❌ NO CONFIGURATIONS FOUND. Constraints still broken.")

if __name__ == "__main__":
    run_constrained_optimization()