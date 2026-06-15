
import math
import mujoco
import numpy as np
import mujoco.viewer
import time
import matplotlib.pyplot as plt
from aero import AeroEngine

# --- Simulatie Parameters ---
XML_PATH = r"Flight_Performance_Simulation\Validation\uiaa_validation.xml"
CD_CSV_PATH = r"Flight_Performance_Simulation\files\Cd_values.csv"

# --- UIAA 101 Test Data ---
MASS = 80.0                # kg
ROPE_LENGTH = 2.8          # m (L0)
DROP_HEIGHT = 4.8          # m (Vrije val, Fall factor ~1.71)
WIRE_DIAMETER = 0.0089     # m (8.9 mm Edelrid Swift)
WIRE_E = 0.42e9             # Pa (Dynamische modulus Nylon 6, ~2.5 GPa)
WIRE_A = math.pi * (WIRE_DIAMETER / 2) ** 2
#WIRE_ETA=0.1e9

# Demping ratio
ZETA = 0.02 

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    # AeroEngine initialiseren met jullie vastgestelde area
    aero = AeroEngine(model, data, cd_csv_path=CD_CSV_PATH, body_name="cylinder_mass", area=0.004)

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cylinder_mass")
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "tether")
    sid_anchor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "anchor")
    
    k_spring = (WIRE_A * WIRE_E) / ROPE_LENGTH
    
    # Native MuJoCo tendon dynamica uitschakelen
    model.tendon_stiffness[tid] = 0.0 
    model.tendon_damping[tid] = 0.0

    time_log = []
    tension_log = []
    
    max_tension = 0.0
    max_dynamic_elongation = 0.0

    t = 0.0
    sim_duration = 5.0 # Seconden
    
    print(f"Berekende K (Stijfheid): {k_spring/1000:.2f} kN/m")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and t < sim_duration:
            step_start = time.time()
            
            drag = aero.compute_drag()
            
            L = float(data.ten_length[tid])
            Ldot = float(data.ten_velocity[tid])
            
            stretch = L - ROPE_LENGTH
            tension = 0.0
            damp_force = np.zeros(3)
            
            if stretch > 0.0:
                spring = k_spring * stretch
                c_taut = 2.0 * ZETA * math.sqrt(k_spring * MASS)
                #c_taut = (WIRE_A * WIRE_ETA) / ROPE_LENGTH
                
                tension = max(0.0, spring + c_taut * Ldot)
                
                # Bepaal 3D-vector van de massa naar het anker
                pos_mass = data.xpos[bid]
                pos_anchor = data.site_xpos[sid_anchor]
                to_anchor = pos_anchor - pos_mass
                u_away = to_anchor / np.linalg.norm(to_anchor)
                
                # Kracht wijst áltijd in de richting van het touw (werkt in zowel Z-up als Z-down)
                damp_force = tension * u_away

                max_tension = max(max_tension, tension)
                current_elongation = (stretch / ROPE_LENGTH) * 100
                max_dynamic_elongation = max(max_dynamic_elongation, current_elongation)
            
            data.xfrc_applied[bid, 0:3] = damp_force + drag
            
            time_log.append(t)
            tension_log.append(tension / 1000.0)

            mujoco.mj_step(model, data)
            viewer.sync()
            
            t += dt
            elapsed = time.time() - step_start
            time.sleep(max(0, dt - elapsed))

    static_stretch = max(0.0, float(data.ten_length[tid]) - ROPE_LENGTH)
    static_elongation = (static_stretch / ROPE_LENGTH) * 100

    print("\n--- VALIDATIE RESULTATEN ---")
    print(f"Max Impact Force:       {max_tension / 1000:.2f} kN")
    print(f"Max Dynamic Elongation: {max_dynamic_elongation:.2f} %")
    print(f"Static Elongation:      {static_elongation:.2f} %")
    print("----------------------------")

    plt.figure(figsize=(10, 5))
    plt.plot(time_log, tension_log, label="Gerekende Tether Spanning (kN)", color="red")
    plt.axhline(y=8.8, color='b', linestyle='--', label="Edelrid Target (8.8 kN)")
    plt.axhline(y=8.6, color='g', linestyle='--', label="Petzl Target (8.6 kN)")
    plt.axhline(y=8.5, color='orange', linestyle='--', label="Mammut Target (8.5 kN)")
    plt.xlabel("Tijd (s)")
    plt.ylabel("Spanning (kN)")
    plt.title("UIAA 101 Validatie: Python Tether Logica & AeroEngine Integratie")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()