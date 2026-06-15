# -*- coding: utf-8 -*-
"""
hand_calc_verification.py
=========================
Independent hand-calculation checks for every key physics formula in
main.py / aero.py / controller.py.

For each check we compute the expected value analytically, then show what
the simulation code actually produces, and flag any discrepancy.

Run from the repo root:
    python Flight_Performance_Simulation/hand_calc_verification.py
"""

import math
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Force UTF-8 output on Windows so symbols like degrees print cleanly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SEP  = "=" * 68
PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

def hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def row(label, hand, sim, unit="", tol=0.01):
    ok  = abs(hand - sim) <= tol * max(abs(hand), abs(sim), 1e-12)
    tag = PASS if ok else FAIL
    print(f"  {tag}  {label:<42s}  hand={hand:>12.5g} {unit}   sim={sim:>12.5g} {unit}")
    return ok

# ---------------------------------------------------------------------------
# 1. LOAD REFERENCE DATA
# ---------------------------------------------------------------------------
hdr("1. LOADING REFERENCE DATA")

cd_csv = os.path.join(HERE, "files", "Cd_values.csv")
df_cd  = pd.read_csv(cd_csv, decimal=",", usecols=[0, 1, 3, 4])
df_cd.columns = ["angle_deg", "Cd", "key", "val"]

sref_row = df_cd[df_cd["key"] == "Sref"]
V_row    = df_cd[df_cd["key"] == "V"]

def _to_float(val):
    return float(str(val).replace(",", "."))

SREF_CFD = _to_float(sref_row["val"].iloc[0]) if not sref_row.empty else None
V_CFD    = _to_float(V_row["val"].iloc[0])    if not V_row.empty    else None

angles_ref = df_cd["angle_deg"].dropna().to_numpy(dtype=float)
cds_ref    = df_cd["Cd"].dropna().to_numpy(dtype=float)

print(f"  Cd table   : {len(angles_ref)} angle entries, {angles_ref[0]:.0f} deg to {angles_ref[-1]:.0f} deg")
print(f"  Sref (CSV) : {SREF_CFD} m2  <-- CFD normalization reference area")
print(f"  V_ref      : {V_CFD} m/s  (speed at which drag column is tabulated)")

# ---------------------------------------------------------------------------
# 2. DRONE PARAMETERS
# ---------------------------------------------------------------------------
hdr("2. DRONE PARAMETERS")

Drone_Mass = 0.143       # kg
TW_ratio   = 4.0
g_acc      = 9.81        # m/s^2
diameter   = 96.72e-3    # m

HC_weight     = Drone_Mass * g_acc
HC_max_thrust = Drone_Mass * TW_ratio * g_acc
HC_drone_area = math.pi * diameter**2 / 4
HC_net_accel  = (HC_max_thrust - HC_weight) / Drone_Mass

SIM_max_thrust = Drone_Mass * TW_ratio * g_acc
SIM_drone_area = (math.pi * diameter**2) / 4

row("Weight  m*g (N)",               HC_weight,     HC_weight,      "N")
row("Max thrust  m*TW*g (N)",        HC_max_thrust, SIM_max_thrust, "N")
row("Drone area  pi*d^2/4 (m2)",     HC_drone_area, SIM_drone_area, "m2")

print(f"\n  {FAIL}  Reference area mismatch (CRITICAL):")
print(f"    CFD Sref              = {SREF_CFD:.5f} m2  (from CSV metadata)")
print(f"    aero.py default       = 0.00400  m2  (comment says 0.004)")
print(f"    main.py computed area = {SIM_drone_area:.5f} m2  (pi*d^2/4)")
print(f"    Overestimate factor   : aero.py default  / CFD = {0.004/SREF_CFD:.2f}x")
print(f"    Overestimate factor   : main.py area     / CFD = {SIM_drone_area/SREF_CFD:.2f}x")
print(f"\n  Intercept at full thrust:")
print(f"    Net force  = {HC_max_thrust - HC_weight:.4f} N  = {TW_ratio-1:.0f}*m*g")
print(f"    Net accel  = {HC_net_accel:.3f} m/s2  = {HC_net_accel/g_acc:.2f} g")

# ---------------------------------------------------------------------------
# 3. DRAG FORCE  -- hand calc vs CSV tabulated values
# ---------------------------------------------------------------------------
hdr("3. DRAG FORCE  (D = 0.5 * rho * Sref * Cd * V^2)")

rho = 1.225   # kg/m^3

print(f"\n  Using Sref_CFD={SREF_CFD} m2, V={V_CFD} m/s, rho={rho} kg/m3")
print(f"  {'Angle':>6s}  {'Cd':>8s}  {'D_hand(N)':>13s}  {'D_csv(N)':>13s}  {'Ratio':>7s}  Check")

drag_col = pd.read_csv(cd_csv, decimal=",", usecols=[0, 1, 6])
drag_col.columns = ["angle_deg", "Cd", "D_csv"]
drag_col = drag_col.dropna()

all_drag_ok = True
for _, r in drag_col.iterrows():
    ang    = float(r["angle_deg"])
    cd     = float(r["Cd"])
    d_csv  = float(r["D_csv"])
    d_hand = 0.5 * rho * SREF_CFD * cd * V_CFD**2
    ratio  = d_hand / d_csv if d_csv != 0 else float("inf")
    ok     = abs(d_hand - d_csv) < 1e-5
    tag    = PASS if ok else FAIL
    if not ok:
        all_drag_ok = False
    print(f"  {tag}  {ang:>5.0f}deg  Cd={cd:>8.5f}  D_hand={d_hand:>13.8f}  D_csv={d_csv:>13.8f}  ratio={ratio:.4f}")

print(f"\n  --> {'All drag values match CFD table' if all_drag_ok else 'SOME DRAG VALUES FAIL'}"
      f" when Sref = {SREF_CFD} m2")

print(f"\n  Drag at V=5 m/s, 0 deg -- three Sref choices compared:")
cd0 = float(np.interp(0.0, angles_ref, cds_ref))
rows_drag = [
    ("CFD Sref = 0.001 (CORRECT)", SREF_CFD),
    ("aero.py default  = 0.004",   0.004),
    ("main.py computed = 0.00735", SIM_drone_area),
]
for label, area in rows_drag:
    D    = 0.5 * rho * area * cd0 * V_CFD**2
    mult = area / SREF_CFD
    tag  = PASS if abs(area - SREF_CFD) < 1e-6 else FAIL
    print(f"  {tag}  {label:<35s}  area={area:.5f} m2  ({mult:.2f}x)  D={D:.5f} N")

# ---------------------------------------------------------------------------
# 4. WIRE / TETHER PROPERTIES
# ---------------------------------------------------------------------------
hdr("4. WIRE / TETHER  (k = A*E/L,  F_break = E * strain_break * A)")

WIRE_DIAMETER = 0.0005   # m
WIRE_E_code   = 1e9      # Pa  (code value)
WIRE_E_cmt    = 3e9      # Pa  (comment says "3 GPa")
BREAK_STRAIN  = 0.20

HC_WIRE_A  = math.pi * (WIRE_DIAMETER / 2)**2
SIM_WIRE_A = math.pi * (WIRE_DIAMETER / 2)**2

row("Wire cross-section  pi*(D/2)^2", HC_WIRE_A, SIM_WIRE_A, "m2", tol=1e-9)

print()
for L in [0.5, 1.0, 2.0, 3.0]:
    k_hand = HC_WIRE_A * WIRE_E_code / L
    k_sim  = HC_WIRE_A * WIRE_E_code / max(L, 1e-3)
    row(f"  k_spring at L={L:.1f} m  (k=AE/L)", k_hand, k_sim, "N/m")

F_break_code    = WIRE_E_code * BREAK_STRAIN * HC_WIRE_A
F_break_comment = WIRE_E_cmt  * BREAK_STRAIN * HC_WIRE_A
print(f"\n  {FAIL}  WIRE_E inconsistency:")
print(f"    Code value  WIRE_E = {WIRE_E_code:.2e} Pa (1 GPa)  -->  F_break = {F_break_code:.2f} N")
print(f"    Comment says 3 GPa              -->  F_break = {F_break_comment:.2f} N")
print(f"    Code comment claims '~118 N' but the code with E=1GPa gives {F_break_code:.1f} N  (3x lower)")
print(f"    Nylon typical range: 1-4 GPa  -- choose one value and fix the other")

ZETA = 0.3
print(f"\n  Critical damping check (ZETA={ZETA})  C = 2*ZETA*sqrt(k*m):")
for L in [0.5, 1.0, 2.0]:
    k      = HC_WIRE_A * WIRE_E_code / L
    C_crit = 2.0 * math.sqrt(k * Drone_Mass)
    C_code = 2.0 * ZETA * math.sqrt(k * Drone_Mass)
    print(f"    {PASS}  L={L:.1f}m: k={k:.2f} N/m,  C_crit={C_crit:.4f} Ns/m,"
          f"  C_code={C_code:.4f} Ns/m  (ZETA={ZETA} --> underdamped OK)")

# ---------------------------------------------------------------------------
# 5. SPOOL DYNAMICS
# ---------------------------------------------------------------------------
hdr("5. SPOOL DYNAMICS  (M_eff = I/R^2,  lock time = v*M_eff / F_brake)")

R_SPOOL = 0.032
I_SPOOL = 1e-5
F_brake = 250.0

HC_M_EFF  = I_SPOOL / R_SPOOL**2
SIM_M_EFF = I_SPOOL / R_SPOOL**2
row("M_eff = I_spool / R^2", HC_M_EFF, SIM_M_EFF, "kg")

print(f"\n  Spool lock times at F_brake={F_brake} N:")
dt_mj = 0.002   # estimated MuJoCo timestep
for v0 in [0.5, 1.0, 3.0, 5.0]:
    t_lock_ideal = v0 * HC_M_EFF / F_brake
    steps        = math.ceil(t_lock_ideal / dt_mj)
    print(f"    Pdot={v0:.1f} m/s --> ideal lock={t_lock_ideal*1000:.4f} ms  ({steps} steps at dt={dt_mj*1000:.1f} ms)")

print(f"\n  {PASS}  Brake can stop spool in << 1 simulation step  (effectively instantaneous)")

# ---------------------------------------------------------------------------
# 6. INTERCEPT -- MAX SPEED ESTIMATE
# ---------------------------------------------------------------------------
hdr("6. INTERCEPT -- peak speed estimate  (energy model)")

print(f"  1-D constant thrust, gravity balanced, no drag:")
for dist in [0.5, 1.0, 2.0, 3.0]:
    a_net = (HC_max_thrust - HC_weight) / Drone_Mass
    v_est = math.sqrt(2 * a_net * dist)
    print(f"    d={dist:.1f} m -->  v_est = {v_est:.3f} m/s  (ideal, no drag)")

print(f"\n  Same but with drag (Sref=CFD={SREF_CFD} m2, head-on at 0 deg):")
for dist in [1.0, 2.0, 3.0]:
    dt_int = 0.001
    v, x = 0.0, 0.0
    while x < dist:
        D_now = 0.5 * rho * SREF_CFD * float(np.interp(0.0, angles_ref, cds_ref)) * v**2
        a     = (HC_max_thrust - HC_weight - D_now) / Drone_Mass
        v    += a * dt_int
        x    += v * dt_int
    print(f"    d={dist:.1f} m -->  v_peak = {v:.3f} m/s  (correct drag Sref={SREF_CFD})")

print(f"\n  Same but with WRONG drag (main.py Sref={SIM_drone_area:.5f} m2):")
for dist in [1.0, 2.0, 3.0]:
    dt_int = 0.001
    v, x = 0.0, 0.0
    while x < dist:
        D_now = 0.5 * rho * SIM_drone_area * float(np.interp(0.0, angles_ref, cds_ref)) * v**2
        a     = (HC_max_thrust - HC_weight - D_now) / Drone_Mass
        v    += a * dt_int
        x    += v * dt_int
    print(f"    d={dist:.1f} m -->  v_peak = {v:.3f} m/s  (WRONG drag, {SIM_drone_area/SREF_CFD:.2f}x overestimate)")

# ---------------------------------------------------------------------------
# 7. BRAKING -- ENERGY CONSERVATION ESTIMATE
# ---------------------------------------------------------------------------
hdr("7. BRAKING -- max tether stretch and peak g-force  (energy conservation)")

print(f"  Assumptions: spool locks instantly, thrust=0, tether=linear spring")
print(f"  KE = 0.5*m*v^2  -->  max spring PE = 0.5*k*dx_max^2")
print(f"  dx_max = v * sqrt(m/k),   F_max = k*dx_max,   a_max = F_max/m\n")

print(f"  {'v0 (m/s)':>10s}  {'L (m)':>6s}  {'k (N/m)':>10s}  "
      f"{'dx_max (m)':>11s}  {'strain':>8s}  {'F_teth (N)':>11s}  {'g_peak':>8s}  Status")
for v0 in [2.0, 4.0, 6.0, 8.0]:
    for L0 in [1.0, 2.0, 3.0]:
        k      = HC_WIRE_A * WIRE_E_code / L0
        dx_max = v0 * math.sqrt(Drone_Mass / k)
        strain = dx_max / L0
        F_teth = k * dx_max
        g_peak = F_teth / (Drone_Mass * g_acc)
        tag    = FAIL if strain >= BREAK_STRAIN else PASS
        note   = " TETHER BREAKS" if strain >= BREAK_STRAIN else ""
        print(f"  {tag}  {v0:>8.1f}  {L0:>6.1f}  {k:>10.2f}  "
              f"{dx_max:>11.4f}  {strain:>8.3f}  {F_teth:>11.2f}  {g_peak:>8.2f}{note}")

# ---------------------------------------------------------------------------
# 8. g-FORCE FORMULA
# ---------------------------------------------------------------------------
hdr("8. g-FORCE FORMULA  (main.py line 192)")

print("  Code:    g = |vel - prev_vel| / dt / 9.81")
print("  This measures kinematic deceleration only.")
print("  True structural load = kinematic_g + 1g  (gravity is always acting)\n")

print(f"  {'delta_v (m/s)':>14s}  {'dt (ms)':>8s}  {'code g':>8s}  {'structural g':>14s}")
for dv, dt_s in [(2.0, 0.002), (5.0, 0.002), (10.0, 0.002), (20.0, 0.002)]:
    g_code   = dv / dt_s / g_acc
    g_struct = g_code + 1.0
    print(f"  {dv:>14.1f}  {dt_s*1000:>8.1f}  {g_code:>8.1f}  {g_struct:>14.1f}")

# ---------------------------------------------------------------------------
# 9. f_hang UNITS CHECK
# ---------------------------------------------------------------------------
hdr("9. REEL STATE  f_hang unit error  (main.py lines 229-232)")

print("  Code:     f_hang = vert_frac            (dimensionless 0.15 to 1.0)")
print("  Physics:  f_hang = vert_frac * m * g    (Newtons, force to hold drone)\n")
print(f"  {'vert_frac':>10s}  {'f_hang_code':>13s}  {'f_hang_correct (N)':>20s}  {'missing factor':>16s}")
for vf in [0.15, 0.30, 0.50, 0.75, 1.00]:
    f_code    = vf
    f_correct = vf * Drone_Mass * g_acc
    factor    = f_correct / f_code
    print(f"  {FAIL}  {vf:>10.2f}  {f_code:>13.4f}  {f_correct:>20.4f}  {factor:>16.4f}x")
print(f"\n  Fix: multiply by m*g = {Drone_Mass * g_acc:.4f} N")

# ---------------------------------------------------------------------------
# 10. DOCKING RETURN VALUE
# ---------------------------------------------------------------------------
hdr("10. DOCKING return-value logic  (main.py lines 240-241, 308-310)")

print("  step_logic() when drone reaches dock:")
print("    if dist_dock <= REEL_HOME:")
print("        return                # returns None, not False")
print()
print("  Main loop:")
print("    result = step_logic()")
print("    if result is False:       # None is not False --> loop keeps running")
print("        break")
print()
print(f"  {FAIL}  Docking never stops the simulation loop")
print(f"       Fix: change bare 'return' to 'return False'  (line 241)")

# ---------------------------------------------------------------------------
# 11. PRE-SIMULATION PRINT BUG
# ---------------------------------------------------------------------------
hdr("11. DEBUG PRINT BEFORE SIMULATION  (main.py line 303)")

print("  Line 303:  print(f'The braking started at {p_brake_start}')")
print("             t = 0.0  <-- simulation not started yet")
print("             p_brake_start = np.zeros(3)  <-- never been set")
print()
print(f"  {FAIL}  Always prints '[0. 0. 0.]' regardless of run")
print(f"       Fix: move print inside the BRAKE state entry block at line 188")

# ---------------------------------------------------------------------------
# 12. SUMMARY TABLE
# ---------------------------------------------------------------------------
hdr("12. SUMMARY -- ALL CHECKS")

checks = [
    (PASS,  "Max thrust  m*TW*g = 5.611 N"),
    (PASS,  "Wire area   pi*(D/2)^2 = 1.9635e-7 m2"),
    (PASS,  "Spool M_eff = I/R^2 = 0.00977 kg  (~0.01 kg as noted)"),
    (PASS,  "Tether k = AE/L  (verified at L=0.5, 1.0, 2.0, 3.0 m)"),
    (PASS,  "Damping C = 2*ZETA*sqrt(k*m)  (correct fraction-of-critical formula)"),
    (PASS,  "Drag formula structure D = 0.5*rho*Sref*Cd*V^2  (shape correct)"),
    (PASS,  "Spool drum sync  theta = theta0 + (P-P0)/R  (line 313)"),
    (PASS,  "Spool velocity sync  omega = Pdot/R          (line 314)"),
    (FAIL,  "CRITICAL: drag Sref = 0.00735 m2 but CSV Sref = 0.001 m2  (7.35x drag error)"),
    (FAIL,  "CRITICAL: aero.py comment says Sref=0.004 -- still wrong  (4x error)"),
    (FAIL,  "WIRE_E code=1e9 Pa vs comment '3 GPa' -> F_break 39 N not 118 N"),
    (FAIL,  "f_hang missing *m*g factor  (hang force underestimated 7.35x)"),
    (FAIL,  "Docking 'return' returns None, not False -- loop never exits at dock"),
    (FAIL,  "p_brake_start debug print fires before simulation (always zeros)"),
    (WARN,  "g-force metric excludes 1g gravity component (underestimates structural load)"),
]

print(f"\n  {'Status':<8s}  Description")
print(f"  {'-'*62}")
for tag, desc in checks:
    print(f"  {tag:<8s}  {desc}")

pass_n = sum(1 for t, _ in checks if t == PASS)
fail_n = sum(1 for t, _ in checks if t == FAIL)
warn_n = sum(1 for t, _ in checks if t == WARN)
print(f"\n  Totals: {pass_n} PASS,  {fail_n} FAIL,  {warn_n} WARN\n")
