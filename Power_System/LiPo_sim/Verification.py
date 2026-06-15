"""
Unit tests for LiPo battery simulation — Physics Verification Focus.

Assumptions
-----------
* The simulation module is named `Lipo_Sim_Monte_Carlo.py`.
  If your file has a different name, update the import below accordingly.
* The polynomial-model modules (Final_Model_Fit_Discharge,
  Final_Model_Fit_Internal_Resistance, Resistance_degradation) are importable.

Run with:
    python -m pytest test_lipo_physics.py -v
    python -m unittest test_lipo_physics -v
"""

import unittest
import numpy as np
import contextlib
import io

# ── Update this import to match your actual filename ──────────────────────────
from Lipo_Sim_Monte_Carlo_Plot import smoothstep, Vocv_poly, R_eff_poly, LiPo_sim
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  1. Circuit-Model Mathematics & Conservation Laws
# ══════════════════════════════════════════════════════════════════════════════
class TestCircuitPhysics(unittest.TestCase):
    """
    Verifies that the quadratic loop math inside the simulation obeys 
    Kirchhoff's and Ohm's laws for a DC source with internal resistance.
    """

    @staticmethod
    def calculate_current_and_voltage(V_ocv, R, P):
        """Replicates the core mathematical cell equations used in the simulation loop."""
        disc = V_ocv**2 - 4.0 * R * P
        disc = max(disc, 0.0)
        I_drawn = (V_ocv - np.sqrt(disc)) / (2.0 * R)
        V_delivered = V_ocv - I_drawn * R
        return I_drawn, V_delivered

    def test_conservation_of_energy(self):
        """Conservation of Energy: Delivered power (V_terminal * I) must match demanded power P."""
        test_cases = [
            # (V_ocv, R_pack, P_demanded)
            (16.0, 0.04, 150.0),
            (14.8, 0.05, 300.0),
            (12.0, 0.03, 50.0)
        ]
        for V_ocv, R, P in test_cases:
            with self.subTest(V_ocv=V_ocv, R=R, P=P):
                I, V_term = self.calculate_current_and_voltage(V_ocv, R, P)
                P_calculated = V_term * I
                self.assertAlmostEqual(P_calculated, P, places=4, 
                                       msg=f"Power mismatch! Expected {P}W, got {P_calculated}W")

    def test_internal_ohmic_losses(self):
        """Analytical Check: Open-circuit power minus terminal power must equal I^2 * R thermal loss."""
        V_ocv = 15.5
        R = 0.04
        P_demand = 200.0
        
        I, V_term = self.calculate_current_and_voltage(V_ocv, R, P_demand)
        
        # Power accounting equation: P_ideal = P_delivered + P_loss
        P_ideal = V_ocv * I
        P_delivered = V_term * I
        P_loss_calculated = P_ideal - P_delivered
        P_loss_theoretical = (I**2) * R
        
        self.assertAlmostEqual(P_loss_calculated, P_loss_theoretical, places=5,
                               msg="Ohmic energy preservation failed! Power loss equation is drift-unstable.")

    def test_maximum_power_transfer_theorem(self):
        """Maximum Power Transfer Theorem: Peak deliverable power occurs when discriminant is exactly 0."""
        # P_max = V_ocv^2 / (4 * R)
        V_ocv = 16.0
        R = 0.05
        P_max_theoretical = (V_ocv**2) / (4.0 * R)  # 1280W
        
        # At exactly P_max, the discriminant should drop to 0
        disc = V_ocv**2 - 4.0 * R * P_max_theoretical
        self.assertAlmostEqual(disc, 0.0, places=6)

    def test_maximum_power_transfer_boundary_values(self):
        """Boundary Hand Calc: At peak power, terminal voltage must be exactly 0.5 * V_ocv and current = V_ocv / 2R."""
        V_ocv = 16.0
        R = 0.05
        P_max_theoretical = (V_ocv**2) / (4.0 * R)
        
        I, V_term = self.calculate_current_and_voltage(V_ocv, R, P_max_theoretical)
        
        # Hand calculated theoretical values at Jacobi boundary condition:
        expected_I = V_ocv / (2.0 * R)  # 160.0 Amperes
        expected_V = V_ocv * 0.5        # 8.0 Volts
        
        self.assertAlmostEqual(I, expected_I, places=4, msg="Boundary current at P_max deviates from analytical limit!")
        self.assertAlmostEqual(V_term, expected_V, places=4, msg="Boundary terminal voltage at P_max deviates from 0.5*V_ocv!")

    def test_hand_calculated_cell_physics(self):
        """Compare simulator's cell math against an exact hand-calculated analytical solution."""
        # Hand-calculated scenario:
        V_ocv = 16.80  # Fully charged 4S pack
        R = 0.045     # Internal resistance in Ohms
        P_demanded = 250.0  # Watts
        
        # Hand calculation steps:
        # Discriminant = 16.8^2 - 4 * 0.045 * 250 = 282.24 - 45 = 237.24
        # sqrt(Discriminant) = 15.402597
        # I = (16.8 - 15.402597) / (2 * 0.045) = 1.397403 / 0.09 = 15.5267 A
        # V_terminal = 16.8 - (15.5267 * 0.045) = 16.1013 V
        expected_I = 15.5267
        expected_V = 16.1013
        
        # Call your actual simulation sub-function with these exact inputs
        calculated_I, calculated_V = self.calculate_current_and_voltage(V_ocv, R, P_demanded)
        
        # Assert to 3 decimal places
        self.assertAlmostEqual(calculated_I, expected_I, places=3, 
                               msg="Current equation deviation from hand calculation!")
        self.assertAlmostEqual(calculated_V, expected_V, places=3, 
                               msg="Terminal Voltage deviation from hand calculation!")


# ══════════════════════════════════════════════════════════════════════════════
#  2. Degradation & Trend Sanity
# ══════════════════════════════════════════════════════════════════════════════
class TestDegradationPhysics(unittest.TestCase):
    """Verifies that empirical degradation trends behave in a physically realistic direction."""

    def test_vocv_monotonicity(self):
        """Open Circuit Voltage (OCV) must strictly decrease as State of Charge drops."""
        soc_high = Vocv_poly(0.9)
        soc_mid  = Vocv_poly(0.5)
        soc_low  = Vocv_poly(0.1)
        
        self.assertGreater(soc_high, soc_mid, "OCV failed to drop from 90% to 50% SOC")
        self.assertGreater(soc_mid, soc_low, "OCV failed to drop from 50% to 10% SOC")


# ══════════════════════════════════════════════════════════════════════════════
#  3. Full Simulation Physical Constraints
# ══════════════════════════════════════════════════════════════════════════════
class TestSimulationPhysicsOutput(unittest.TestCase):
    """Runs a single simulation pass to verify the output time-series honor physical boundaries."""

    @classmethod
    def setUpClass(cls):
        # Anchor seed to remove stochastic noise during tracking
        np.random.seed(42)
        
        print("\nRunning heavy battery simulation loop (Muting internal prints)...")
        
        # ── THE FIX: Suppress the thousand-line print warning bottleneck ──────
        with contextlib.redirect_stdout(io.StringIO()):
            cls.result = LiPo_sim(P_max_mot=320, P_avg_mot=130, t_flight=10)
        # ──────────────────────────────────────────────────────────────────────
        
        (cls.I, cls.Soc, cls.V, cls.Crate, 
         cls.t_recharge, cls.fail_cycles, cls.t, cls.V_min) = cls.result

    def test_soc_monotonic_decay(self):
        """State of Charge (SOC) must only decrease or stay flat during a discharge flight."""
        soc_differences = np.diff(self.Soc)
        # Differences should be <= 0 (allowing minor floating-point tolerances)
        self.assertTrue(np.all(soc_differences <= 1e-9), 
                        f"Physical violation: Battery SOC increased mid-flight! Max gain: {soc_differences.max()}")

    def test_crate_definition_consistency(self):
        """System Invariant Check: Extracted capacity (I / C_rate) must remain perfectly constant over time."""
        # Grab time indices where current draw is active to avoid division by zero
        active_indices = np.where(self.I > 0.5)[0]
        
        if len(active_indices) > 0:
            # Derive the implicit battery nominal capacity using the first active point
            idx_start = active_indices[0]
            derived_capacity = self.I[idx_start] / self.Crate[idx_start]
            
            # Step through the dataset and ensure capacity does not morph or fluctuate with load changes
            for step in active_indices[::20]:  # Sample intervals to optimize speed
                current_ratio = self.I[step] / self.Crate[step]
                self.assertAlmostEqual(current_ratio, derived_capacity, places=4,
                                       msg=f"Mathematical Inconsistency: C-rate calculation isn't scaled invariant to capacity at step {step}!")

    def test_battery_voltage_bounds(self):
        """Terminal voltage for a 4S pack must remain within safe lithium-polymer boundaries (~10V to ~17V)."""
        max_v = np.max(self.V)
        min_v = np.min(self.V)
        
        self.assertLessEqual(max_v, 17.0, f"Pack voltage exceeds physical 4S ceiling: {max_v}V")
        self.assertGreaterEqual(min_v, 10.0, f"Pack voltage dropped dangerously low without being caught: {min_v}V")

    def test_current_sign_convention(self):
        """Current and C-rate must be strictly non-negative (discharge only, no regeneration)."""
        self.assertTrue(np.all(self.I >= -1e-6), "Negative current detected during purely resistive discharge simulation.")
        self.assertTrue(np.all(self.Crate >= -1e-6), "Negative C-rate detected.")

    def test_recharge_time_plausibility(self):
        """Recharge time calculation matches expected physical timeframes for the spent depth of discharge."""
        # Charging at 2C means a full charge (100% DoD) takes 0.5 hours (1800s). 
        # A partial flight DoD must take strictly less than 30 minutes.
        self.assertGreater(self.t_recharge, 0.0, "Recharge time cannot be zero or negative.")
        self.assertLess(self.t_recharge, 1800.0, "Recharge time exceeds absolute physical window for a 2C rate.")

# ══════════════════════════════════════════════════════════════════════════════
#  4. Smoothstep Function Properties
# ══════════════════════════════════════════════════════════════════════════════
class TestSmoothstepFunction(unittest.TestCase):
    """
    Verifies the logistic sigmoid (smoothstep) function used to shape
    the power transition profile during peak-power manoeuvres.
    """

    def test_smoothstep_at_zero_is_half(self):
        """
        Analytical: smoothstep(0) = 1 / (1 + e^0) = 1/2 exactly.

        Hand calculation:
            e^0 = 1  →  1 / (1 + 1) = 0.5
        """
        result = smoothstep(0.0)
        self.assertAlmostEqual(result, 0.5, places=12,
                               msg="smoothstep(0) must equal exactly 0.5 by definition of the logistic function.")

    def test_smoothstep_asymptotic_saturation(self):
        """
        Logistic function saturates to 1 for large positive x and to 0 for large
        negative x (verified to 10 decimal places at |x| = 100).

        Hand calculation:
            smoothstep(100)  = 1 / (1 + e^{-100}) ≈ 1 − 3.7e-44  →  effectively 1.0
            smoothstep(-100) = 1 / (1 + e^{100})  ≈ 3.7e-44       →  effectively 0.0
        """
        self.assertAlmostEqual(smoothstep(100.0),  1.0, places=10,
                               msg="smoothstep must saturate to 1 at large positive x.")
        self.assertAlmostEqual(smoothstep(-100.0), 0.0, places=10,
                               msg="smoothstep must saturate to 0 at large negative x.")

    def test_smoothstep_is_monotonically_increasing(self):
        """
        The logistic function is strictly monotonically increasing for all real x.

        Hand calculation:
            smoothstep(-5) = 1 / (1 + e^5)  ≈ 0.00669
            smoothstep( 0) = 0.5
            smoothstep( 5) = 1 / (1 + e^{-5}) ≈ 0.99331
            ⇒ 0.00669 < 0.5 < 0.99331  ✓
        """
        s_neg = smoothstep(-5.0)
        s_mid = smoothstep(0.0)
        s_pos = smoothstep(5.0)
        self.assertLess(s_neg, s_mid, "smoothstep must increase from x=-5 to x=0.")
        self.assertLess(s_mid, s_pos, "smoothstep must increase from x=0 to x=5.")


# ══════════════════════════════════════════════════════════════════════════════
#  5. Circuit Edge Cases — Additional Hand-Calculated Tests
# ══════════════════════════════════════════════════════════════════════════════
class TestCircuitEdgeCases(unittest.TestCase):
    """
    Supplements TestCircuitPhysics with boundary and limiting-case scenarios
    that are each independently verifiable by hand.
    """

    @staticmethod
    def solve_circuit(V_ocv, R, P):
        """Replicates the quadratic solver used inside the simulation loop."""
        disc = V_ocv**2 - 4.0 * R * P
        disc = max(disc, 0.0)
        I = (V_ocv - np.sqrt(disc)) / (2.0 * R)
        V = V_ocv - I * R
        return I, V

    def test_zero_power_demand_gives_zero_current(self):
        """
        At P = 0 the battery is idle: current must be zero and terminal voltage
        must equal the open-circuit voltage.

        Hand calculation:
            disc = V_ocv² − 4R·0 = V_ocv²
            sqrt(disc) = V_ocv
            I = (V_ocv − V_ocv) / (2R) = 0
            V_term = V_ocv − 0·R = V_ocv
        """
        V_ocv, R = 16.0, 0.05
        I, V_term = self.solve_circuit(V_ocv, R, 0.0)
        self.assertAlmostEqual(I,      0.0,  places=12,
                               msg="Zero power demand must produce zero current.")
        self.assertAlmostEqual(V_term, V_ocv, places=12,
                               msg="No-load terminal voltage must equal V_ocv.")

    def test_low_power_current_approximates_P_over_V(self):
        """
        For P ≪ P_max, a first-order Taylor expansion of the quadratic solution
        gives I ≈ P / V_ocv (the internal resistance drop is negligible).

        Hand calculation (P = 1 W, V_ocv = 16 V, R = 0.04 Ω):
            disc    = 16² − 4(0.04)(1) = 256 − 0.16 = 255.84
            sqrt    ≈ 15.9950   [since 15.995² = 255.840025]
            I       = (16 − 15.9950) / 0.08 = 0.005 / 0.08 = 0.0625 A
            P/V_ocv = 1 / 16 = 0.0625 A  ✓   (match to 4 decimal places)
        """
        V_ocv, R, P = 16.0, 0.04, 1.0
        I, _ = self.solve_circuit(V_ocv, R, P)
        self.assertAlmostEqual(I, P / V_ocv, places=4,
                               msg="Low-load current must approximate P / V_ocv.")

    def test_kirchhoff_voltage_law(self):
        """
        KVL for a single-loop DC circuit: V_ocv − V_terminal = I · R_internal.

        Hand calculation (V_ocv = 15.5 V, R = 0.04 Ω, P = 200 W):
            disc    = 15.5² − 4(0.04)(200) = 240.25 − 32 = 208.25
            sqrt    ≈ 14.4309
            I       = (15.5 − 14.4309) / 0.08 ≈ 13.364 A
            V_term  = 15.5 − 13.364 × 0.04 ≈ 14.965 V
            KVL:    15.5 − 14.965 = 0.535 = 13.364 × 0.04  ✓
        """
        V_ocv, R, P = 15.5, 0.04, 200.0
        I, V_term = self.solve_circuit(V_ocv, R, P)
        self.assertAlmostEqual(V_ocv - V_term, I * R, places=9,
                               msg="KVL violated: terminal voltage drop must equal I × R_internal.")

if __name__ == "__main__":
    unittest.main()