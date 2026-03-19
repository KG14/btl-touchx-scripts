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

# ── Default constants for standalone execution ───────────────────────────
DEFAULT_TOUCHX_X_MIN, DEFAULT_TOUCHX_X_MAX = -210, 210
DEFAULT_TOUCHX_Y_MIN, DEFAULT_TOUCHX_Y_MAX = -100, 95
DEFAULT_TOUCHX_Z_MIN, DEFAULT_TOUCHX_Z_MAX = -145, 95
DEFAULT_A = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
], dtype=float)
DEFAULT_SCALE = 0.002
DEFAULT_TOUCHX_CENTER = np.array([0, -60, -95], dtype=float)


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
    def __init__(
        self,
        *,
        touchx_limits=None,
        A=None,
        scale=None,
        touchx_center=None,
    ):
        if touchx_limits is None:
            touchx_limits = {
                "x_min": DEFAULT_TOUCHX_X_MIN,
                "x_max": DEFAULT_TOUCHX_X_MAX,
                "y_min": DEFAULT_TOUCHX_Y_MIN,
                "y_max": DEFAULT_TOUCHX_Y_MAX,
                "z_min": DEFAULT_TOUCHX_Z_MIN,
                "z_max": DEFAULT_TOUCHX_Z_MAX,
            }
        self.touchx_limits = touchx_limits
        self.A = np.asarray(DEFAULT_A if A is None else A, dtype=float)
        self.scale = float(DEFAULT_SCALE if scale is None else scale)
        self.touchx_center = np.asarray(
            DEFAULT_TOUCHX_CENTER if touchx_center is None else touchx_center,
            dtype=float,
        )
        self.pb_full_min, self.pb_full_max = self._compute_pb_full_bounds()

        self.root = tk.Tk()
        self.root.title("PyBullet Workspace Constraint Visualizer")
        self.root.minsize(960, 640)
        self.result = None

        # ── matplotlib figure (left) ─────────────────────────────────────
        left = ttk.Frame(self.root)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7, 6), dpi=100, facecolor="white")
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
            mn.insert(0, f"{self.pb_full_min[i]:.4f}")
            mn.grid(row=0, column=1, padx=4, pady=2)

            ttk.Label(frame, text="Max:").grid(row=1, column=0, sticky="e")
            mx = ttk.Entry(frame, width=12)
            mx.insert(0, f"{self.pb_full_max[i]:.4f}")
            mx.grid(row=1, column=1, padx=4, pady=2)

            ttk.Label(
                frame,
                text=f"Full range: [{self.pb_full_min[i]:.4f}, {self.pb_full_max[i]:.4f}]",
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
    def _touchx_to_pb(self, tx_pos):
        return self.scale * (self.A @ (np.asarray(tx_pos, dtype=float) - self.touchx_center))

    def _compute_pb_full_bounds(self):
        corners_tx = np.array(list(itertools.product(
            [self.touchx_limits["x_min"], self.touchx_limits["x_max"]],
            [self.touchx_limits["y_min"], self.touchx_limits["y_max"]],
            [self.touchx_limits["z_min"], self.touchx_limits["z_max"]],
        )))
        corners_pb = np.array([self._touchx_to_pb(corner) for corner in corners_tx])
        return corners_pb.min(axis=0), corners_pb.max(axis=0)

    def _read_constraints(self):
        mins, maxs = [], []
        for i, axis in enumerate(["X", "Y", "Z"]):
            try:
                lo = float(self.entries[f"{axis}_min"].get())
            except ValueError:
                lo = self.pb_full_min[i]
            try:
                hi = float(self.entries[f"{axis}_max"].get())
            except ValueError:
                hi = self.pb_full_max[i]
            lo = max(lo, self.pb_full_min[i])
            hi = min(hi, self.pb_full_max[i])
            if lo > hi:
                lo, hi = hi, lo
            mins.append(lo)
            maxs.append(hi)
        return np.array(mins), np.array(maxs)

    def _update_plot(self):
        self.ax.cla()

        # White pane backgrounds (reset after cla)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.pane.set_facecolor("white")
            axis.pane.set_edgecolor("lightgray")

        # Full-range wireframe box
        full_v = _box_verts(self.pb_full_min, self.pb_full_max)
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
        self.ax.set_xlim(self.pb_full_min[0] - pad, self.pb_full_max[0] + pad)
        self.ax.set_ylim(self.pb_full_min[1] - pad, self.pb_full_max[1] + pad)
        self.ax.set_zlim(self.pb_full_min[2] - pad, self.pb_full_max[2] + pad)
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
    viz = ConstraintVisualizer()
    pb_full_min = viz.pb_full_min
    pb_full_max = viz.pb_full_max
    center = viz.touchx_center
    print("TouchX → PyBullet coordinate mapping (defaults):")
    print(f"  pb_x = scale * (touch_z - ({center[2]}))")
    print(f"  pb_y = scale * (touch_x - ({center[0]}))")
    print(f"  pb_z = scale * (touch_y - ({center[1]}))")
    print(f"  scale = {viz.scale}")
    print()
    print("Full reachable PB bounding box (meters):")
    print(f"  X: [{pb_full_min[0]:.4f}, {pb_full_max[0]:.4f}]")
    print(f"  Y: [{pb_full_min[1]:.4f}, {pb_full_max[1]:.4f}]")
    print(f"  Z: [{pb_full_min[2]:.4f}, {pb_full_max[2]:.4f}]")
    print()

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
