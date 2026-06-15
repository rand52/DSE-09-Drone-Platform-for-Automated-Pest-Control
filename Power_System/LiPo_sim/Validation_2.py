import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from Final_Model_Fit_Discharge import intercept, coef
from Final_Model_Fit_Internal_Resistance import intercept_res, coef_res

plt.style.use(['science', 'no-latex', 'grid'])


# ── Fitted model functions ────────────────────────────────────────────────────

def Vocv_poly(soc: np.ndarray | float) -> np.ndarray | float:
    """OCV–SoC polynomial (degree 4, fitted)."""
    x2 = soc ** 2
    x3 = soc ** 3
    x4 = soc ** 4
    return intercept + coef[1]*soc + coef[2]*x2 + coef[3]*x3 + coef[4]*x4


def R_eff_scale(soc: np.ndarray | float) -> np.ndarray | float:
    """SoC-dependent resistance scaling factor (linear, fitted).
    Multiply by R0_base to get the effective cell resistance [Ω]."""
    return intercept_res + coef_res[0] * soc


# ── Constant-current discharge simulation ────────────────────────────────────

def simulate_cc_discharge(
    discharge_current: float = 3.0,   # Applied current [A]
    capacity_Ah: float = 1.0,         # Nominal capacity [Ah]  → 1000 mAh
    initial_soc: float = 1.0,         # Starting SoC [0–1]; tune to match dataset
    R0_base: float = 0.015,           # Base internal resistance [Ω]; tune to match dataset
    v_cutoff: float = 3.0,            # End-of-discharge cutoff voltage [V]
    dt: float = 0.01,                  # Integration timestep [s]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a single constant-current discharge using the fitted OCV and
    resistance models.  No power control, no C-rate checks, no Monte Carlo.

    Returns
    -------
    t_arr   : time [s]
    I_arr   : current [A]  (constant = discharge_current)
    V_arr   : terminal voltage [V]
    soc_arr : state of charge [–]
    """
    capacity_As  = capacity_Ah * 3600.0
    used_charge  = (1.0 - initial_soc) * capacity_As   # Pre-depleted charge [As]

    t_list, I_list, V_list, soc_list = [], [], [], []
    t = 0.0

    while True:
        soc    = float(np.clip((capacity_As - used_charge) / capacity_As, 0.0, 1.0))
        V_ocv  = Vocv_poly(soc)                              # Open-circuit voltage [V]
        R_eff  = R0_base * R_eff_scale(soc)                  # Effective resistance [Ω]
        V_term = V_ocv - discharge_current * R_eff           # Terminal voltage [V]

        # Stop at cutoff or full depletion
        if V_term < v_cutoff or soc <= 0.0:
            break

        t_list.append(t)
        I_list.append(discharge_current)
        V_list.append(V_term)
        soc_list.append(soc)

        used_charge += discharge_current * dt
        t            = round(t + dt, 10)   # avoid float drift

    return (
        np.array(t_list),
        np.array(I_list),
        np.array(V_list),
        np.array(soc_list),
    )


# ── Validation metrics ────────────────────────────────────────────────────────

def compute_metrics(
    t_meas: np.ndarray,
    V_meas: np.ndarray,
    t_sim: np.ndarray,
    V_sim: np.ndarray,
    trim_frac: float = 0.05,   # Fraction of points stripped from each end for
                                # the trimmed max-relative-error calculation
) -> dict:
    """
    Interpolate the simulated voltage onto the measured time grid and compute
    validation statistics.

    trim_frac controls how much of the signal is treated as "beginning" and
    "end" for the max relative error calculation.  At 0.05 (default) the first
    and last 5 % of measured time-points are excluded.
    """
    V_interp = np.interp(t_meas, t_sim, V_sim, left=np.nan, right=np.nan)
    mask     = ~np.isnan(V_interp)
    err      = V_interp[mask] - V_meas[mask]

    # ── Trimmed max relative error ─────────────────────────────────────────
    n          = np.sum(mask)
    trim_n     = max(1, int(np.round(n * trim_frac)))   # points to drop each side
    inner_mask = np.zeros(n, dtype=bool)
    inner_mask[trim_n : n - trim_n] = True

    V_meas_inner  = V_meas[mask][inner_mask]
    err_inner     = err[inner_mask]
    rel_err_inner = np.abs(err_inner) / np.abs(V_meas_inner) * 100.0   # [%]

    return {
        'rmse_mV':             np.sqrt(np.mean(err ** 2)) * 1e3,
        'mae_mV':              np.mean(np.abs(err)) * 1e3,
        'max_err_mV':          np.max(np.abs(err)) * 1e3,
        'max_rel_err_pct':     np.max(rel_err_inner),        # NEW
        'trim_frac':           trim_frac,                    # NEW (for display)
        't_discharge_sim_s':   t_sim[-1],
        't_discharge_meas_s':  t_meas[-1],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Configuration ─────────────────────────────────────────────────────────
    DISCHARGE_CURRENT = 3.0     # [A]  – must match the dataset
    CAPACITY_Ah       = 1.0     # [Ah] – 1000 mAh
    INITIAL_SOC       = 1.0     # [–]  – tune if measured curve starts below max OCV
    R0_BASE           = 0.012   # [Ω]  – tune to shift voltage curve up/down uniformly
    V_CUTOFF          = 2.75     # [V]  – end-of-discharge cell voltage
    TRIM_FRAC         = 0.05    # Fraction of endpoints ignored for max rel-error

    # ── Load measured data ────────────────────────────────────────────────────
    PATH = r"C:\Users\spash\OneDrive\Desktop\Uni\Bachelor\Year 3\DSE\DSE Code\DSE-09-Drone-Platform-for-Automated-Pest-Control\Power_System\CSV files\2_0_stressdischarge_3.csv"

    df = pd.read_csv(
    PATH,
    sep='\t',
    header=None,
    names=['Time_s', 'Current_A', 'Voltage_V'],
    skiprows=1
)
    
    df['Time_s'] = pd.to_numeric(df['Time_s'], errors='coerce')
    df['Current_A'] = pd.to_numeric(df['Current_A'], errors='coerce')
    df['Voltage_V'] = pd.to_numeric(df['Voltage_V'], errors='coerce')

    df = df.dropna()
    df['Current_A'] = df['Current_A'].abs()

    t_meas = df['Time_s'].values 
    I_meas = df['Current_A'].values 
    V_meas = df['Voltage_V'].values

    # ── Run simulation ────────────────────────────────────────────────────────
    t_sim, I_sim, V_sim, soc_sim = simulate_cc_discharge(
        discharge_current = DISCHARGE_CURRENT,
        capacity_Ah       = CAPACITY_Ah,
        initial_soc       = INITIAL_SOC,
        R0_base           = R0_BASE,
        v_cutoff          = V_CUTOFF,
        dt                = 0.1,
    )
    
    # ── Print validation metrics ──────────────────────────────────────────────
    metrics = compute_metrics(t_meas, V_meas, t_sim, V_sim, trim_frac=TRIM_FRAC)
    trim_pct = metrics['trim_frac'] * 100
    print("─" * 42)
    print(f"  Voltage RMSE          : {metrics['rmse_mV']:.2f} mV")
    print(f"  Voltage MAE           : {metrics['mae_mV']:.2f} mV")
    print(f"  Max absolute error    : {metrics['max_err_mV']:.2f} mV")
    print(f"  Max relative error    : {metrics['max_rel_err_pct']:.3f} %"
          f"  (inner {100 - 2*trim_pct:.0f}% of signal)")
    print(f"  Discharge time  (sim) : {metrics['t_discharge_sim_s']:.0f} s"
          f"  ({metrics['t_discharge_sim_s']/60:.1f} min)")
    print(f"  Discharge time (meas) : {metrics['t_discharge_meas_s']:.0f} s"
          f"  ({metrics['t_discharge_meas_s']/60:.1f} min)")
    print("─" * 42)

    # ── Figure 1: Terminal Voltage ────────────────────────────────────────────
    fig, ax = plt.subplots()
    ax.plot(t_meas, V_meas,
            color='tomato',    linewidth=1.5,  label='Measured')
    ax.plot(t_sim,  V_sim,
            color='steelblue', linewidth=1.5,  linestyle='--', label='Simulated')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')
    ax.set_ylim(3,4.6)
    ax.legend()
    plt.tight_layout()

    # ── Figure 2: Discharge Current ───────────────────────────────────────────
    fig, ax = plt.subplots()
    ax.plot(t_meas, I_meas,
            color='tomato',    linewidth=1.5,  label='Measured')
    ax.plot(t_sim,  I_sim,
            color='steelblue', linewidth=1.5,  linestyle='--', label='Simulated')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Current (A)')
    ax.set_ylim(0,4)
    ax.set_title('Discharge Current – Measured vs Simulated')
    ax.legend()
    plt.tight_layout()

    # ── Figure 3: Simulated State of Charge ───────────────────────────────────
    fig, ax = plt.subplots()
    ax.plot(t_sim, soc_sim * 100, color='steelblue', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('State of Charge (%)')
    ax.set_title('Simulated State of Charge')
    plt.tight_layout()

    plt.show()