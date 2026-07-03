#!/usr/bin/env python3
"""Day 1 - see the channel for yourself.

Plots one frame of the atomistic channel (ordered FCC walls + disordered LJ
liquid) from the trajectory that `-var dump 1` writes, so you can see the domain
the four measurements run on. Run any sheet with the extra flag, e.g.

    ./submit.sh density.in -var dump 1     # (locally:  lmp_serial -in density.in -var dump 1)
    python3 view_domain.py

It reads the FIRST frame of day1_traj.xyz and saves day1_domain.png. The view is
a thin y-slice (so the wall lattice reads cleanly) of the x-z plane, walls in
blue and liquid in red. Same graceful style as the analyzers: if matplotlib is
missing it says so and exits without crashing.
"""
import os
import sys

import numpy as np

try:
    import matplotlib
    #matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("    (matplotlib not found - skipping the plot)")
    raise

def xyz_num_frames(path):
    """Return the number of frames in a LAMMPS XYZ trajectory."""
    with open(path) as f:
        n_atoms = int(f.readline())

        # Count total number of lines
        f.seek(0)
        n_lines = sum(1 for _ in f)

    lines_per_frame = n_atoms + 2
    return n_atoms, n_lines // lines_per_frame

def read_frame(path, frame):
    """Read a specific frame from a LAMMPS XYZ trajectory.

    Parameters
    ----------
    path : str
        XYZ trajectory filename.
    frame : int
        Zero-based frame index.

    Returns
    -------
    el : ndarray
        Element symbols.
    x, y, z : ndarray
        Coordinate arrays.
    """

    with open(path) as f:
        # Read first atom count to determine frame size
        n = int(f.readline())
        lines_per_frame = n + 2

        # Return to start of file
        f.seek(0)

        # Skip previous frames
        for _ in range(frame * lines_per_frame):
            f.readline()

        # Read requested frame
        n = int(f.readline())
        f.readline()  # comment line

        rows = [f.readline().split() for _ in range(n)]

    el = np.array([r[0] for r in rows])
    xyz = np.array([[float(v) for v in r[1:4]] for r in rows])

    return el, xyz[:, 0], xyz[:, 1], xyz[:, 2]


if __name__ == "__main__":

    TRAJ = "day2_traj.xyz"
    if not os.path.exists(TRAJ):
        sys.exit(f"{TRAJ} not found - run a sheet with `-var dump 1` first, e.g. "
                 f"./submit.sh density.in -var dump 1")

    Nplots = 5
    fig, axs = plt.subplots(Nplots,1)

    n_atoms, Nframes = xyz_num_frames(TRAJ)

    trajs = np.int32(np.round(np.linspace(0, Nframes-1, Nplots)))
    for i in range(Nplots):
        el, x, y, z = read_frame(TRAJ, trajs[i])
        axs[i].plot(z,  x, '.', ms=0.4)
        axs[i].set_aspect("equal")
    plt.show()

