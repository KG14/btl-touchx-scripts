"""
Interactive 3D bounding-box constraint visualizer for the TouchX → PyBullet workspace.

Shows the full reachable volume (derived from TouchX hardware limits) as a wireframe
box, and lets the user type XYZ min/max values to define a tighter constraint box.
Press OK to finalize; the chosen limits are printed to stdout and returned.
"""

import itertools
import numpy as np
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── TouchX hardware limits (mm), rounded values from minmax.txt ──────────
TOUCHX_X_MIN, TOUCHX_X_MAX = -210, 210
TOUCHX_Y_MIN, TOUCHX_Y_MAX = -100,  95
TOUCHX_Z_MIN, TOUCHX_Z_MAX = -145,  95

# ── Mapping constants (must match touchx_pybullet_tracking.py) ───────────
A = np.array([
    [0,  0,  1],   # pb_x = touch_z
    [1,  0,  0],   # pb_y = touch_x
    [0, -1,  0],   # pb_z = -touch_y
], dtype=float)

SCALE = 0.002                              # TouchX mm → PB meters
TOUCHX_CENTER = np.array([0, 95, -110])    # TouchX ref that maps to PB origin


def touchx_to_pb(tx_pos):
    """Convert a TouchX position (mm) to PyBullet world coords (m)."""
    return SCALE * (A @ (np.asarray(tx_pos, dtype=float) - TOUCHX_CENTER))


# ── Compute full reachable PB bounding box ───────────────────────────────
_corners_tx = np.array(list(itertools.product(
    [TOUCHX_X_MIN, TOUCHX_X_MAX],
    [TOUCHX_Y_MIN, TOUCHX_Y_MAX],
    [TOUCHX_Z_MIN, TOUCHX_Z_MAX],
)))
_corners_pb = np.array([touchx_to_pb(c) for c in _corners_tx])
PB_FULL_MIN = _corners_pb.min(axis=0)
PB_FULL_MAX = _corners_pb.max(axis=0)


# ── 3-D box drawing helpers ──────────────────────────────────────────────
def _box_verts(mins, maxs):
    """8 corner vertices of an axis-aligned box."""
    return np.array(list(itertools.product(
        [mins[0], maxs[0]],
        [mins[1], maxs[1]],
        [mins[2], maxs[2]],
    )))


def _box_faces(verts):
    """6 quad faces from 8 vertices (itertools.product ordering)."""
    idx = [
        [0, 2, 6, 4],  # z-min
        [1, 3, 7, 5],  # z-max
        [0, 1, 5, 4],  # y-min
        [2, 3, 7, 6],  # y-max
        [0, 1, 3, 2],  # x-min
        [4, 5, 7, 6],  # x-max
    ]
    return [verts[i] for i in idx]


# ── GUI ──────────────────────────────────────────────────────────────────
class ConstraintVisualizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PyBullet Workspace Constraint Visualizer")
        self.root.minsize(960, 640)
        self.result = None

        # ── matplotlib figure (left) ─────────────────────────────────────
        left = ttk.Frame(self.root)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, left)
        toolbar.update()

        # ── controls panel (right) ───────────────────────────────────────
        ctrl = ttk.Frame(self.root, padding=12)
        ctrl.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(ctrl, text="Constraint Bounds (m)",
                  font=("", 14, "bold")).pack(pady=(0, 10))

        axes = ["X", "Y", "Z"]
        self.entries: dict[str, ttk.Entry] = {}

        for i, axis in enumerate(axes):
            frame = ttk.LabelFrame(ctrl, text=f"PB {axis} axis", padding=6)
            frame.pack(fill=tk.X, pady=5)

            ttk.Label(frame, text="Min:").grid(row=0, column=0, sticky="e")
            mn = ttk.Entry(frame, width=12)
            mn.insert(0, f"{PB_FULL_MIN[i]:.4f}")
            mn.grid(row=0, column=1, padx=4, pady=2)

            ttk.Label(frame, text="Max:").grid(row=1, column=0, sticky="e")
            mx = ttk.Entry(frame, width=12)
            mx.insert(0, f"{PB_FULL_MAX[i]:.4f}")
            mx.grid(row=1, column=1, padx=4, pady=2)

            ttk.Label(
                frame,
                text=f"Full range: [{PB_FULL_MIN[i]:.4f}, {PB_FULL_MAX[i]:.4f}]",
                foreground="gray",
            ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))

            self.entries[f"{axis}_min"] = mn
            self.entries[f"{axis}_max"] = mx

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=10)

        ttk.Button(ctrl, text="Update Preview",
                   command=self._update_plot).pack(fill=tk.X, pady=4)

        self.ok_btn = ttk.Button(ctrl, text="OK  —  Finalize",
                                 command=self._finalize)
        self.ok_btn.pack(fill=tk.X, pady=4)

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=10)

        # info labels
        ttk.Label(ctrl, text="Blue wireframe = full reachable range",
                  foreground="steelblue").pack(anchor="w")
        ttk.Label(ctrl, text="Orange box = active constraint",
                  foreground="darkorange").pack(anchor="w")
        ttk.Label(ctrl, text="Red dot = PB origin  (0,0,0)",
                  foreground="red").pack(anchor="w")

        self._update_plot()

    # ── helpers ──────────────────────────────────────────────────────────
    def _read_constraints(self):
        mins, maxs = [], []
        for i, axis in enumerate(["X", "Y", "Z"]):
            try:
                lo = float(self.entries[f"{axis}_min"].get())
            except ValueError:
                lo = PB_FULL_MIN[i]
            try:
                hi = float(self.entries[f"{axis}_max"].get())
            except ValueError:
                hi = PB_FULL_MAX[i]
            lo = max(lo, PB_FULL_MIN[i])
            hi = min(hi, PB_FULL_MAX[i])
            if lo > hi:
                lo, hi = hi, lo
            mins.append(lo)
            maxs.append(hi)
        return np.array(mins), np.array(maxs)

    def _update_plot(self):
        self.ax.cla()

        # Full-range wireframe box
        full_v = _box_verts(PB_FULL_MIN, PB_FULL_MAX)
        self.ax.add_collection3d(Poly3DCollection(
            _box_faces(full_v),
            alpha=0.04, facecolor="skyblue",
            edgecolor="steelblue", linewidths=0.8,
        ))

        # Constraint box
        c_min, c_max = self._read_constraints()
        c_v = _box_verts(c_min, c_max)
        self.ax.add_collection3d(Poly3DCollection(
            _box_faces(c_v),
            alpha=0.18, facecolor="orange",
            edgecolor="darkorange", linewidths=1.2,
        ))

        # Corner annotations on constraint box
        for v in c_v:
            self.ax.text(v[0], v[1], v[2],
                         f"({v[0]:.3f},{v[1]:.3f},{v[2]:.3f})",
                         fontsize=6, color="saddlebrown", alpha=0.7)

        # Origin marker
        self.ax.scatter(*[[0]], *[[0]], *[[0]],
                        color="red", s=50, zorder=5, label="PB Origin")

        pad = 0.05
        self.ax.set_xlim(PB_FULL_MIN[0] - pad, PB_FULL_MAX[0] + pad)
        self.ax.set_ylim(PB_FULL_MIN[1] - pad, PB_FULL_MAX[1] + pad)
        self.ax.set_zlim(PB_FULL_MIN[2] - pad, PB_FULL_MAX[2] + pad)
        self.ax.set_xlabel("PB X (m)")
        self.ax.set_ylabel("PB Y (m)")
        self.ax.set_zlabel("PB Z (m)")
        self.ax.set_title("Workspace Bounding Box")
        self.ax.legend(fontsize=8, loc="upper left")

        self.canvas.draw_idle()

    def _finalize(self):
        c_min, c_max = self._read_constraints()
        self.result = {
            "x_min": round(c_min[0], 5), "x_max": round(c_max[0], 5),
            "y_min": round(c_min[1], 5), "y_max": round(c_max[1], 5),
            "z_min": round(c_min[2], 5), "z_max": round(c_max[2], 5),
        }
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.result


def main():
    print("TouchX → PyBullet coordinate mapping:")
    print(f"  pb_x = scale * (touch_z - ({TOUCHX_CENTER[2]}))")
    print(f"  pb_y = scale * (touch_x - ({TOUCHX_CENTER[0]}))")
    print(f"  pb_z = scale * -(touch_y - ({TOUCHX_CENTER[1]}))")
    print(f"  scale = {SCALE}")
    print()
    print("Full reachable PB bounding box (meters):")
    print(f"  X: [{PB_FULL_MIN[0]:.4f}, {PB_FULL_MAX[0]:.4f}]")
    print(f"  Y: [{PB_FULL_MIN[1]:.4f}, {PB_FULL_MAX[1]:.4f}]")
    print(f"  Z: [{PB_FULL_MIN[2]:.4f}, {PB_FULL_MAX[2]:.4f}]")
    print()

    viz = ConstraintVisualizer()
    result = viz.run()

    if result:
        print("\nFinalized constraints:")
        for k, v in result.items():
            print(f"  {k.upper()} = {v}")
    else:
        print("\nWindow closed without finalizing.")

    return result


if __name__ == "__main__":
    main()
