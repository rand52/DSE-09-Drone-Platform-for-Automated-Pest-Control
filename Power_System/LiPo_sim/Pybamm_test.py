import numpy as np
import matplotlib.pyplot as plt
import pybamm
if __name__ == "__main__":
    # ── Flight & Power Parameters (matching your function defaults) ───────────────
    P_max_input  = 320      # W (propeller shaft)
    P_avg_input  = 128      # W (propeller shaft)
    t_flight     = 10       # s

    P_max = P_max_input / 0.8   # electrical draw
    P_avg = P_avg_input / 0.8

    t_p_max          = 4        # duration of peak power [s]
    t_t_p_max_frac   = 0        # start of peak as fraction of flight

    # ── Precompute the spool-lagged power profile (your exact loop logic) ─────────
    dt   = 0.001
    t    = np.arange(0, t_flight + dt, dt)

    t_start_max_power = t_t_p_max_frac * t_flight
    t_max_power_start = t_start_max_power
    t_max_power_end   = t_start_max_power + t_p_max

    tau_up   = 0.1
    tau_down = 0.6
    t_ramp   = 0.05

    def smoothstep(x):
        return 1 / (1 + np.exp(-x))

    P_actual = P_avg
    P_profile = np.zeros(len(t))

    for idx, time in enumerate(t):
        transition = (smoothstep((time - t_max_power_start) / t_ramp)
                    - smoothstep((time - t_max_power_end)   / t_ramp))

        P_target = P_avg + (P_max - P_avg) * transition

        tau_use  = tau_up if P_target > P_actual else tau_down
        P_actual += (P_target - P_actual) * (dt / tau_use)

        P_profile[idx] = P_actual   # total pack power [W]

    # Per-cell power (4S, cells in series → same current, divide power by num_cells)
    num_cells    = 4
    P_per_cell   = P_profile / num_cells

    # ── PyBaMM Power Interpolant ──────────────────────────────────────────────────
    P_interpolant = pybamm.Interpolant(
        t, P_per_cell, pybamm.t, interpolator="linear"
    )

    # ── Model ─────────────────────────────────────────────────────────────────────
    model = pybamm.lithium_ion.DFN(options={"operating mode": "power"})

    # ── Parameters ────────────────────────────────────────────────────────────────
    param = pybamm.ParameterValues("Chen2020")

    param["Nominal cell capacity [A.h]"]  = 0.5
    param["Upper voltage cut-off [V]"]    = 4.2
    param["Lower voltage cut-off [V]"]    = 2.75
    param["Power function [W]"]           = P_interpolant

    # ── Simulation ────────────────────────────────────────────────────────────────
    initial_soc = 0.65

    sim = pybamm.Simulation(
        model,
        parameter_values=param,
        solver=pybamm.CasadiSolver(mode="safe"),
    )

    t_eval = np.linspace(0, t_flight, 2000)

    sol = sim.solve(t_eval=t_eval, initial_soc=initial_soc)

    # ── Extract Results ───────────────────────────────────────────────────────────
    t_out    = sol["Time [s]"].entries
    V_cell   = sol["Terminal voltage [V]"].entries
    I_cell   = sol["Current [A]"].entries
    capacity_Ah      = 0.5
    discharge_cap    = sol["Discharge capacity [A.h]"].entries
    soc              = initial_soc - (discharge_cap / capacity_Ah)

    V_pack   = V_cell * num_cells
    I_pack   = I_cell                       # series cells → same current
    P_pack   = V_pack * I_pack

    # ── Recharge Time (matching your calculation) ─────────────────────────────────
    initial_charge_soc = 0.65
    charging_rate      = 2
    DoD                = initial_charge_soc - soc[-1]
    recharge_time      = DoD * 3600 / charging_rate
    minutes_recharge   = int(recharge_time // 60)
    seconds_recharge   = int(recharge_time % 60)

    print(f"Final SOC:          {soc[-1]*100:.1f}%")
    print(f"DoD:                {DoD*100:.1f}%")
    print(f"Recharge Time (2C): {minutes_recharge} min {seconds_recharge} s")

    # ── Plot (matching your 2x2 layout) ──────────────────────────────────────────
    fig, ax = plt.subplots(2, 2, figsize=(9, 5))

    # C-rate
    C_rate = I_cell / 0.5
    ax[0,0].plot(t_out, C_rate,       color='steelblue', linewidth=2)
    ax[0,0].set(xlabel='Time (s)', ylabel='C-rate (-)')

    # SOC
    ax[0,1].plot(t_out, soc * 100,    color='steelblue', linewidth=2)
    ax[0,1].set(xlabel='Time (s)', ylabel='State of Charge (%)')

    # Voltage
    ax[1,0].plot(t_out, V_cell,       color='tomato',    linewidth=2, label='Voltage per cell')
    ax[1,0].plot(t_out, V_pack,       color='steelblue', linewidth=2, label='Total voltage')
    ax[1,0].axhline(2.75,             color='red', linestyle='--',    label='Cutoff voltage', linewidth=1)
    ax[1,0].set(xlabel='Time (s)', ylabel='Voltage (V)')
    ax[1,0].legend()

    # Current
    ax[1,1].plot(t_out, I_pack,       color='steelblue', linewidth=2)
    ax[1,1].set(xlabel='Time (s)', ylabel='Current (A)')

    plt.tight_layout()
    plt.show()