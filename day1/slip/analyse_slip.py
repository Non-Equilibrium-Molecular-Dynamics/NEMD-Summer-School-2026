#!/usr/bin/env python3
"""Day 1 - slip length.

Reads the velocity profile written by slip.in and reports the slip length b:
how far past the wall the bulk (Couette) velocity profile must extrapolate to
reach the wall speed. b=0 -> the liquid sticks (no-slip); b>0 -> it slips.
The two walls are symmetric (equal T and eps_wf), so a single b describes both:
the profile runs from -vwall to +vwall over (h + 2b), so b = vwall/|s| - h/2.
Run after `lmp_serial -in slip.in`.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lammps_io import read_profile, read_params, read_series_or_none

def plot(z, vx, s, c, zwlo, zwhi, vwall, b, out="day1_slip.png"):
    """Save the velocity profile v_x(z) (z vertical, side-on like the channel)
    with the central Couette fit extrapolated to the wall faces, so the slip
    length b reads off as the offset between the fit and the wall speed."""
    try:
        import matplotlib
        if not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")     # headless: just save the PNG
        import matplotlib.pyplot as plt
    except ImportError:
        print("    (matplotlib not found - skipping the plot)")
        return
    BLUE, RED = "#1F3A5F", "#B23A2E"
    fig, ax = plt.subplots(figsize=(4.0, 4.2))
    zl = np.linspace(min(zwlo - 1.0, zwlo - b - 0.5), max(zwhi + 1.0, zwhi + b + 0.5), 80)
    ax.plot(2 * vwall / (zwhi - zwlo) * (zl - 0.5 * (zwlo + zwhi)), zl,
            color="#999", ls="--", lw=1.0, label=r"no-slip reference ($b=0$)")
    ax.scatter(vx, z, s=14, color=BLUE, alpha=0.75, label=r"measured $v_x(z)$")
    ax.plot(s * zl + c, zl, color=RED, lw=1.4, label="central fit, extrapolated")
    ax.axhline(zwlo, color="#888", ls=":", lw=0.9); ax.axhline(zwhi, color="#888", ls=":", lw=0.9)
    ax.axvline(-vwall, color="#bbb", ls="--", lw=0.7); ax.axvline(vwall, color="#bbb", ls="--", lw=0.7)
    # mark b: the distance beyond the top wall face where the fit reaches +vwall
    if abs(s) > 1e-12 and abs(b) < (zwhi - zwlo):
        z_hit = (vwall - c) / s
        ax.annotate("", xy=(vwall, zwhi), xytext=(vwall, z_hit),
                    arrowprops=dict(arrowstyle="<->", color=RED, lw=1.0))
        ax.text(vwall - 0.12 * vwall, 0.5 * (zwhi + z_hit), r"$b \approx %.1f\,\sigma$" % b,
                color=RED, ha="right", va="center", fontsize=8)
    ax.set_xlabel(r"$v_x(z)$"); ax.set_ylabel(r"$z\ (\sigma)$")
    ax.set_title(r"velocity profile  ($b = %.3f\,\sigma$)" % b)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"    plot -> {out}")
    if os.environ.get("DISPLAY"):     # ssh -X: also pop the figure up on screen
        try:
            plt.show()
        except KeyboardInterrupt:      # Ctrl-C with the window up: fine, the file is already saved
            pass

def main():
    for fn in ("day1_vx.profile", "day1_params.txt"):
        if not os.path.exists(fn):
            sys.exit(f"{fn} not found. If you submitted with ../submit.sh, wait for the job "
                     f"(squeue --me) to finish; otherwise run `lmp_serial -in slip.in` first.")
    z, n, vx = read_profile("day1_vx.profile")
    m = n > 0.5                              # keep only bins that hold atoms
    z, vx = z[m], vx[m]
    P = read_params("day1_params.txt")
    Lz, wall_th, vwall = P["Lz"], P["wall_th"], P["vwall"]
    zwlo, zwhi = wall_th, Lz - wall_th
    h = zwhi - zwlo                          # channel width (wall faces apart)

    # peculiar fluid temperature (production average), if slip.in wrote it
    T = read_series_or_none("day1_T.dat", 2)
    Tfluid = None if T is None else T[1][-1]   # last data row = production-averaged T

    # fit the central half of the channel (away from the structured near-wall layers)
    zc, W = 0.5 * (z.min() + z.max()), z.max() - z.min()
    cen = (z > zc - 0.25 * W) & (z < zc + 0.25 * W)
    if np.count_nonzero(cen) < 2:
        raise SystemExit("slip fit: fewer than 2 occupied bins in the central half of "
                         "day1_vx.profile - check the run completed and the binning/geometry.")
    s, c = np.polyfit(z[cen], vx[cen], 1)
    resid = vx[cen] - (s * z[cen] + c)
    ss_tot = np.sum((vx[cen] - vx[cen].mean()) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else float("nan")

    ns_slope = 2 * vwall / h                 # slope if the liquid stuck to the walls
    v_face_bot = s * zwlo + c                 # fluid velocity extrapolated to each wall face
    v_face_top = s * zwhi + c

    print("\nSheet 2: slip length")
    print(f"    central shear rate dvx/dz = {s:+.4f}  (R^2={r2:.3f})")
    print(f"    no-slip reference 2*vwall/h = {ns_slope:+.4f}  (slope if the liquid")
    print("      stuck to the walls; slip makes the measured slope smaller)")
    if Tfluid is not None:
        print(f"    fluid temperature = {Tfluid:.2f}  (set point 1.0)")
        if Tfluid > 1.3:
            print("      -> well above the set point: the shear has viscously heated the fluid")

    # Physical check: the central-half slope should not GREATLY exceed the no-slip
    # reference 2*vwall/h. A few-percent exceedance can be real physics -- a pinned
    # wetting layer (at larger eps_wf) or viscous-heating shear enhancement narrows the
    # shearing region, so the central slope runs a little above the width-averaged
    # reference and b goes slightly negative. Only a LARGE exceedance (> tol below)
    # signals a setup error (e.g. drive-velocity units, or a runaway), so it warns.
    tol = 1.15
    if abs(s) > tol * ns_slope or max(abs(v_face_bot), abs(v_face_top)) > tol * vwall:
        print(f"    fluid velocity at the walls = {v_face_bot:+.3f} / {v_face_top:+.3f}  "
              f"(wall speed +/-{vwall:g})")
        sys.exit(f"    FAIL: the central slope exceeds the no-slip reference by more than "
                 f"{(tol-1)*100:.0f}% -> unphysical.\n"
                 "    A few percent can be a pinned wetting layer or viscous heating, but this\n"
                 "    much is a setup error (check the drive-velocity units / steady state),\n"
                 "    not a slip length.")

    b = vwall / abs(s) - h / 2               # symmetric slip length (walls are equivalent)
    print(f"    slip length b ~ {b:.3f} sigma  (symmetric walls -> both equal),")
    print(f"      measured to the geometric wall face (z={zwlo:.2f}/{zwhi:.2f}).")
    if b > 0.05:
        print("    -> b > 0: the liquid SLIPS along the wall.")
    elif b < -0.05:
        if Tfluid is not None and Tfluid > 1.3:
            print(f"    -> b < 0 AND the fluid has heated to T~{Tfluid:.1f}: this is viscous heating.")
            print("       The hot mid-channel is less viscous and shears faster than the wall-speed/")
            print("       h reference, so the central fit exceeds no-slip and b turns negative. Not")
            print("       a real slip -- stay in the gentle (linear, isothermal) regime.")
        else:
            print("    -> b < 0: the central slope is steeper than no-slip, so the effective no-slip")
            print(f"       plane sits ~{-b:.2f} sigma inside the geometric wall face (e.g. a strongly-")
            print("       wetting wall pinning the first layer). |b| is below the ~sigma interface,")
            print("       so read the sign with care.")
    else:
        print("    -> b ~ 0: effectively no slip (the liquid sticks).")
    plot(z, vx, s, c, zwlo, zwhi, vwall, b)

if __name__ == "__main__":
    main()
