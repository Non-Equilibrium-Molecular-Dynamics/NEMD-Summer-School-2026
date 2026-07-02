#!/usr/bin/env python3
"""Day 1 - see the channel for yourself.

Plots one frame of the atomistic channel (ordered FCC walls + disordered LJ
liquid) from the trajectory that `-var dump 1` writes, so you can see the domain
the four measurements run on. From inside a measurement folder, run that sheet with
the extra flag, then view it with this tool one level up, e.g.

    cd density
    ../submit.sh density.in -var dump 1    # (locally:  lmp_serial -in density.in -var dump 1)
    python3 ../view_domain.py

It reads the FIRST frame of day1_traj.xyz and saves day1_domain.png. The view is
a thin y-slice (so the wall lattice reads cleanly) of the x-z plane, walls in
blue and liquid in red. If matplotlib is missing, the script prints a message and
exits without plotting.
"""
import os
import sys

import numpy as np

BLUE, RED = "#1F3A5F", "#B23A2E"
TRAJ, OUT = "day1_traj.xyz", "day1_domain.png"


def read_first_frame(path):
    """First frame of an XYZ trajectory -> (element, x, y, z) arrays.
    Raises ValueError if the file is truncated or not a valid XYZ frame."""
    with open(path) as f:
        try:
            n = int(f.readline())                      # atom count
        except ValueError:
            raise ValueError("first line is not an atom count")
        f.readline()                                   # comment line
        rows = [f.readline().split() for _ in range(n)]
    if any(len(r) < 4 for r in rows):                  # file ended early / ragged row
        raise ValueError(f"expected {n} rows of 'element x y z', file ends early or is malformed")
    el = np.array([r[0] for r in rows])
    try:
        xyz = np.array([[float(v) for v in r[1:4]] for r in rows])
    except ValueError:
        raise ValueError("non-numeric coordinates")
    return el, xyz[:, 0], xyz[:, 1], xyz[:, 2]


def main():
    if not os.path.exists(TRAJ):
        sys.exit(f"{TRAJ} not found - from inside a case folder, run that sheet with "
                 f"`-var dump 1` first, e.g.  ../submit.sh density.in -var dump 1")
    try:
        el, x, y, z = read_first_frame(TRAJ)
    except (OSError, ValueError) as e:
        sys.exit(f"{TRAJ} is incomplete or not a valid XYZ trajectory ({e}); "
                 f"rerun the sheet with -var dump 1.")
    slab = np.abs(y - y.mean()) < 1.1                  # thin y-slice -> clean lattice
    wall, liq = (el != "Ar") & slab, (el == "Ar") & slab

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    (matplotlib not found - skipping the plot)")
        return

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(x[wall], z[wall], s=12, c=BLUE, label="FCC wall")
    ax.scatter(x[liq],  z[liq],  s=9,  c=RED, alpha=0.8, label="LJ liquid")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x\ (\sigma)$"); ax.set_ylabel(r"$z\ (\sigma)$")
    ax.legend(fontsize=8, frameon=True, facecolor="white", framealpha=0.9,
              edgecolor="none", loc="upper right")
    fig.tight_layout(); fig.savefig(OUT, dpi=150)
    print(f"    domain -> {OUT}")


if __name__ == "__main__":
    main()
