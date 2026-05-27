#!/usr/bin/env python3
"""
PATS Insect Tracker – 3D Flight Visualisation with Pure Pursuit
Reads log_itrk2.csv and animates the smoothed insect trajectory in 3D,
overlaid with a simulated pursuer following a pure pursuit strategy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import argparse


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIL_FRAMES   = 60          # how many past frames the fading trail shows
FPS            = 30          # animation playback speed
INTERVAL_MS    = 1000 / FPS  # ms between frames
MARKER_FP      = "fp_not_a_fp"   # label for true detections

# ---------------------------------------------------------------------------
# Pursuer configuration  — tune these to match your scenario
# ---------------------------------------------------------------------------
PURSUER_SPEED        = 6   # m/s — set relative to typical insect speed
PURSUER_MAX_TURN     = 18.0   # rad/s — reduce to make pursuer less agile
                              #         set to None for perfect instant-turn
PURSUER_START_OFFSET = (0.0, 0.0, 0.0)  # (dx, dy, dz) offset from insect t=0


# ---------------------------------------------------------------------------
# Pursuer model
# ---------------------------------------------------------------------------
class Pursuer:
    """
    Point-mass pure pursuit model in 3D.

    The pursuer steers directly toward the target at each timestep.
    Agility is controlled via max_turn_rate (rad/s): the maximum angular
    rate the pursuer can change its heading per second. Set to None for
    an ideal pursuer that instantly faces the target.
    """

    def __init__(self, x: float, y: float, z: float,
                 speed: float, max_turn_rate: float | None = None):
        self.x = x
        self.y = y
        self.z = z
        self.speed        = speed
        self.max_turn_rate = max_turn_rate

        # Initialise headings pointing at origin — will be corrected on
        # first step anyway.
        self.heading_xy = 0.0   # azimuth  (radians, XY plane)
        self.heading_z  = 0.0   # elevation (radians)

    @staticmethod
    def _clamp_angle(desired: float, current: float, max_delta: float) -> float:
        """Rotate `current` toward `desired` by at most `max_delta` rad."""
        err = np.arctan2(np.sin(desired - current),
                         np.cos(desired - current))
        return current + np.clip(err, -max_delta, max_delta)

    def step(self, tx: float, ty: float, tz: float, dt: float) -> None:
        """Advance pursuer one timestep toward target (tx, ty, tz)."""
        dx, dy, dz = tx - self.x, ty - self.y, tz - self.z

        desired_xy = np.arctan2(dy, dx)
        desired_z  = np.arctan2(dz, np.hypot(dx, dy))

        if self.max_turn_rate is not None:
            max_delta = self.max_turn_rate * dt
            self.heading_xy = self._clamp_angle(desired_xy, self.heading_xy, max_delta)
            self.heading_z  = self._clamp_angle(desired_z,  self.heading_z,  max_delta)
        else:
            self.heading_xy = desired_xy
            self.heading_z  = desired_z

        cos_z    = np.cos(self.heading_z)
        self.x  += self.speed * np.cos(self.heading_xy) * cos_z * dt
        self.y  += self.speed * np.sin(self.heading_xy) * cos_z * dt
        self.z  += self.speed * np.sin(self.heading_z)          * dt


# ---------------------------------------------------------------------------
# Load & prepare data
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")

    # Drop trailing empty column if present
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Keep only valid-position rows
    df = df[df["pos_valid_insect"] == 1].reset_index(drop=True)

    # Convenience: true detection vs. false positive
    df["is_true"] = df["fp"] == MARKER_FP

    return df

def compute_dynamics(df: pd.DataFrame):
    # Extract velocity components
    vx = df["svelX_insect"].to_numpy()
    vy = df["svelY_insect"].to_numpy()
    vz = df["svelZ_insect"].to_numpy()
    t  = df["elapsed"].to_numpy()

    # Compute dt
    dt = np.diff(t)
    dt[dt == 0] = np.nan  # avoid division by zero

    # Compute acceleration components (finite difference)
    ax = np.diff(vx) / dt
    ay = np.diff(vy) / dt
    az = np.diff(vz) / dt

    # Acceleration magnitude
    a_mag = np.sqrt(ax**2 + ay**2 + az**2)

    # Velocity magnitude (aligned: one element shorter due to diff)
    v_mag = np.sqrt(vx[:-1]**2 + vy[:-1]**2 + vz[:-1]**2)

    # Find max acceleration
    idx = np.nanargmax(a_mag)

    max_acc = a_mag[idx]
    vel_at_max_acc = v_mag[idx]
    time_at_max_acc = t[idx]

    return max_acc, vel_at_max_acc, time_at_max_acc
# ---------------------------------------------------------------------------
# Pre-compute pursuer trajectory
# ---------------------------------------------------------------------------
def simulate_pursuer(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the pursuer forward over the full insect trajectory and return
    (px, py, pz) arrays aligned frame-for-frame with the insect data.
    """
    dt_arr = np.diff(t, prepend=t[0])   # per-frame timestep from real timestamps
    # Guard against zero dt on first frame
    dt_arr[0] = dt_arr[1] if len(dt_arr) > 1 else 1.0 / FPS

    ox, oy, oz = PURSUER_START_OFFSET
    pursuer = Pursuer(
        x=x[0] + ox,
        y=y[0] + oy,
        z=z[0] + oz,
        speed=PURSUER_SPEED,
        max_turn_rate=PURSUER_MAX_TURN,
    )

    n = len(x)
    px, py, pz = np.empty(n), np.empty(n), np.empty(n)

    for i in range(n):
        pursuer.step(x[i], y[i], z[i], dt_arr[i])
        px[i], py[i], pz[i] = pursuer.x, pursuer.y, pursuer.z

    return px, py, pz


# ---------------------------------------------------------------------------
# Main animation
# ---------------------------------------------------------------------------
def build_animation(df: pd.DataFrame) -> tuple[plt.Figure, animation.FuncAnimation]:
    x = df["sposX_insect"].to_numpy()
    y = df["sposY_insect"].to_numpy()
    z = df["sposZ_insect"].to_numpy()
    t = df["elapsed"].to_numpy()
    is_true = df["is_true"].to_numpy()

    n_frames = len(df)

    # Speed magnitude for info display
    speed = np.sqrt(
        df["svelX_insect"] ** 2 +
        df["svelY_insect"] ** 2 +
        df["svelZ_insect"] ** 2
    ).to_numpy()
    speed_norm = (speed - speed.min()) / (np.ptp(speed) + 1e-9)  # noqa: kept for potential future use

    # Pre-compute pursuer path
    print("Simulating pursuer …")
    px, py, pz = simulate_pursuer(x, y, z, t)
    miss = np.sqrt((px - x)**2 + (py - y)**2 + (pz - z)**2)
    print(f"  min miss distance = {miss.min():.4f} m  "
          f"(frame {miss.argmin() + 1})  |  "
          f"final miss = {miss[-1]:.4f} m")

    # ---- Figure layout ----
    fig = plt.figure(figsize=(12, 8), facecolor="#0d0d0d")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#0d0d0d")

    # Axis styling
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#333333")
    ax.grid(True, color="#222222", linewidth=0.5)
    ax.tick_params(colors="#888888", labelsize=7)
    for label in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
        label.set_color("#aaaaaa")
        label.set_fontsize(9)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Fixed axis limits — expand to include pursuer trajectory
    all_x = np.concatenate([x, px])
    all_y = np.concatenate([y, py])
    all_z = np.concatenate([z, pz])
    pad = 0.05
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
    ax.set_zlim(all_z.min() - pad, all_z.max() + pad)

    # ---- Static ghost trajectories (dim) ----
    ax.plot(x,  y,  z,  color="#333333", linewidth=0.6, zorder=1)   # insect ghost
    ax.plot(px, py, pz, color="#003355", linewidth=0.6, zorder=1)   # pursuer ghost

    # ---- Insect animated artists ----
    trail_lines = [
        ax.plot([], [], [], "-", linewidth=1.5)[0]
        for _ in range(TRAIL_FRAMES)
    ]

    (dot,) = ax.plot([], [], [], "o", color="white",
                     markersize=7, zorder=5,
                     markeredgecolor="#ffcc00", markeredgewidth=1.2)

    (fp_dot,) = ax.plot([], [], [], "x", color="#ff4444",
                        markersize=9, markeredgewidth=2, zorder=6)

    # ---- Pursuer animated artists ----
    pursuer_trail_lines = [
        ax.plot([], [], [], "-", linewidth=1.5)[0]
        for _ in range(TRAIL_FRAMES)
    ]

    (p_dot,) = ax.plot([], [], [], "o", color="#00ccff",
                       markersize=7, zorder=5,
                       markeredgecolor="#0066ff", markeredgewidth=1.2)

    # Dashed line connecting pursuer → insect (engagement geometry)
    (engagement_line,) = ax.plot([], [], [], "--",
                                  color="#ffffff", linewidth=0.6,
                                  alpha=0.4, zorder=4)

    # ---- Text overlays ----
    info_text = fig.text(
        0.01, 0.97, "", va="top", ha="left",
        color="#cccccc", fontsize=9,
        fontfamily="monospace",
        transform=fig.transFigure,
    )

    title = fig.text(
        0.5, 0.98, "PATS Insect Tracker — 3-D Flight + Pure Pursuit",
        va="top", ha="center",
        color="#eeeeee", fontsize=13, fontweight="bold",
        transform=fig.transFigure,
    )

    turn_label = (f"{PURSUER_MAX_TURN} rad/s"
                  if PURSUER_MAX_TURN is not None else "∞ (perfect)")
    legend_text = fig.text(
        0.01, 0.06,
        f"●  insect (white)    ✕  false positive    "
        f"●  pursuer (cyan)\n"
        f"── ghost trajectories    trail = last {TRAIL_FRAMES} frames\n"
        f"pursuer speed = {PURSUER_SPEED} m/s    "
        f"max turn rate = {turn_label}",
        va="bottom", ha="left",
        color="#888888", fontsize=7.5,
        transform=fig.transFigure,
    )

    # ---- Init callback ----
    def init():
        for line in trail_lines:
            line.set_data_3d([], [], [])
            line.set_color((1.0, 0.25, 0.0, 0.0))
        for line in pursuer_trail_lines:
            line.set_data_3d([], [], [])
            line.set_color((0.0, 0.6, 1.0, 0.0))
        dot.set_data_3d([], [], [])
        fp_dot.set_data_3d([], [], [])
        p_dot.set_data_3d([], [], [])
        engagement_line.set_data_3d([], [], [])
        info_text.set_text("")
        return (*trail_lines, dot, fp_dot,
                *pursuer_trail_lines, p_dot, engagement_line, info_text)

    # ---- Update callback ----
    def update(frame):
        i = frame

        # --- Insect trail ---
        start = max(0, i - TRAIL_FRAMES)
        seg_xs = x[start:i + 1]
        seg_ys = y[start:i + 1]
        seg_zs = z[start:i + 1]
        n_segs = len(seg_xs) - 1

        for k, line in enumerate(trail_lines):
            if k < n_segs:
                alpha = (k + 1) / TRAIL_FRAMES * 0.85
                line.set_data_3d(
                    [seg_xs[k], seg_xs[k + 1]],
                    [seg_ys[k], seg_ys[k + 1]],
                    [seg_zs[k], seg_zs[k + 1]],
                )
                line.set_color((1.0, 0.25, 0.0, alpha))
            else:
                line.set_data_3d([], [], [])
                line.set_color((1.0, 0.25, 0.0, 0.0))

        # --- Insect dot ---
        dot.set_data_3d([x[i]], [y[i]], [z[i]])

        # --- False-positive marker ---
        if not is_true[i]:
            fp_dot.set_data_3d([x[i]], [y[i]], [z[i]])
        else:
            fp_dot.set_data_3d([], [], [])

        # --- Pursuer trail ---
        p_seg_xs = px[start:i + 1]
        p_seg_ys = py[start:i + 1]
        p_seg_zs = pz[start:i + 1]
        p_n_segs = len(p_seg_xs) - 1

        for k, line in enumerate(pursuer_trail_lines):
            if k < p_n_segs:
                alpha = (k + 1) / TRAIL_FRAMES * 0.85
                line.set_data_3d(
                    [p_seg_xs[k], p_seg_xs[k + 1]],
                    [p_seg_ys[k], p_seg_ys[k + 1]],
                    [p_seg_zs[k], p_seg_zs[k + 1]],
                )
                line.set_color((0.0, 0.6, 1.0, alpha))
            else:
                line.set_data_3d([], [], [])
                line.set_color((0.0, 0.6, 1.0, 0.0))

        # --- Pursuer dot ---
        p_dot.set_data_3d([px[i]], [py[i]], [pz[i]])

        # --- Engagement line (pursuer → insect) ---
        engagement_line.set_data_3d(
            [px[i], x[i]], [py[i], y[i]], [pz[i], z[i]]
        )

        # --- Info text ---
        elapsed_rel = t[i] - t[0]
        info_text.set_text(
            f"t = {t[i]:.3f} s  (+{elapsed_rel:.2f} s)\n"
            f"insect   X={x[i]:+.3f}  Y={y[i]:+.3f}  Z={z[i]:+.3f} m\n"
            f"pursuer  X={px[i]:+.3f}  Y={py[i]:+.3f}  Z={pz[i]:+.3f} m\n"
            f"speed = {speed[i]:.3f} m/s    miss = {miss[i]:.3f} m\n"
            f"frame {i + 1} / {n_frames}"
        )

        return (*trail_lines, dot, fp_dot,
                *pursuer_trail_lines, p_dot, engagement_line, info_text)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=INTERVAL_MS,
        blit=False,
    )

    return fig, anim


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default="log_itrk3.csv")
    args = parser.parse_args()

    print(f"Loading {args.csv} …")
    #df = load_data(args.csv)
    df = load_data(r"C:\Users\spash\OneDrive\Desktop\Uni\Bachelor\Year 3\DSE\DSE Code\DSE-09-Drone-Platform-for-Automated-Pest-Control\Power_System\CSV files\log_itrk7.csv")

    max_acc, vel_at_max_acc, t_acc = compute_dynamics(df)

    print("\n--- Dynamics Estimate ---")
    print(f"Max acceleration: {max_acc:.3f} m/s²")
    print(f"Velocity at max acceleration: {vel_at_max_acc:.3f} m/s")
    print(f"Time of occurrence: {t_acc:.3f} s")


if __name__ == "__main__":
    main()
