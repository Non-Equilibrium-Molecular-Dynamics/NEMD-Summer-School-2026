#!/usr/bin/env python3
"""Day 1 - shear viscosity.

Reads the velocity profile and the fluid-only shear-stress time series written by
viscosity.in and reports eta = |pxz| / (dvx/dz): the shear stress over the
shear rate. p_xz is the fluid-only virial (the stiff walls dropped, less noisy
than the global pressure tensor); eta from a short LJ run is still a QUALITATIVE
trend, not a precise number (it shifts with run length and seed). Run after
`lmp_serial -in viscosity.in`.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lammps_io import read_profile, read_timeseries, read_params

def plot(z, vx, s, c, zwlo, zwhi, Tpec, pxz_series, eta, out="day1_viscosity.png"):
    """Three panels: the clean linear velocity profile (left) gives the shear rate,
    the peculiar temperature T(z) (centre) shows whether viscous heating has humped
    the channel, and the fluid-only shear-stress trace (right) gives p_xz. The
    contrast is the point: the shear rate is well converged, eta (from the fluid-only
    virial) takes more averaging -- and once T(z) humps, viscous heating has set in
    even though the velocity profile stays close to straight."""
    try:
        import matplotlib
        if not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")     # headless: just save the PNG
        import matplotlib.pyplot as plt
    except ImportError:
        print("    (matplotlib not found - skipping the plot)")
        return
    BLUE, RED, GREY = "#1F3A5F", "#B23A2E", "#888888"
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(10.5, 3.6))
    axL.scatter(vx, z, s=14, color=BLUE, alpha=0.75)
    zl = np.linspace(zwlo, zwhi, 50)
    axL.plot(s * zl + c, zl, color=RED, lw=1.4)
    axL.set_xlabel(r"$v_x(z)$"); axL.set_ylabel(r"$z\ (\sigma)$")
    axL.set_title(r"velocity profile $v_x(z)$")
    # centre panel: peculiar temperature T(z) = (2/3)(<ke> - 0.5 vx^2). Flat at a
    # gentle shear; shear hard and viscous heating humps it (hot centre), so the
    # channel is no longer isothermal and a single eta no longer describes it.
    axM.scatter(z, Tpec, s=14, color=RED, alpha=0.85)
    axM.axvline(zwlo, color="#aaa", ls=":", lw=0.8); axM.axvline(zwhi, color="#aaa", ls=":", lw=0.8)
    axM.set_xlabel(r"$z\ (\sigma)$"); axM.set_ylabel(r"fluid temperature $T(z)$")
    axM.set_title(r"fluid temperature $T(z)$")
    # right panel: instantaneous p_xz (noisy) vs its RUNNING mean. The running mean
    # should flatten once eta has converged; if it is still drifting, it has not.
    k = np.arange(1, len(pxz_series) + 1)
    run = np.cumsum(pxz_series) / k
    axR.plot(k, pxz_series, color=GREY, lw=0.7, alpha=0.7, label="instantaneous")
    axR.plot(k, run, color=RED, lw=1.6, label=r"running mean $%+.3f$" % run[-1])
    axR.set_xlabel("sample"); axR.set_ylabel(r"$p_{xz}$")
    axR.set_title(r"shear stress $p_{xz}$  ($\eta \approx %.2f$)" % eta)
    axR.legend(fontsize=8, frameon=True, facecolor="white", framealpha=0.9,
               edgecolor="none", loc="upper right")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"    plot -> {out}")
    if os.environ.get("DISPLAY"):     # ssh -X: also pop the figure up on screen
        try:
            plt.show()
        except KeyboardInterrupt:      # Ctrl-C with the window up: fine, the file is already saved
            pass

def main():
    for fn in ("day1_vx.profile", "day1_KEz.profile", "day1_stress.dat", "day1_params.txt"):
        if not os.path.exists(fn):
            sys.exit(f"{fn} not found. If you submitted with ../submit.sh, wait for the job "
                     f"(squeue --me) to finish; otherwise run `lmp_serial -in viscosity.in` first.")
    z, n, vx = read_profile("day1_vx.profile")
    z_ke, n_ke, ke = read_profile("day1_KEz.profile")
    if not (np.allclose(z, z_ke) and np.allclose(n, n_ke)):
        sys.exit("day1_KEz.profile bins do not match day1_vx.profile")
    Tpec = (2.0 / 3.0) * (ke - 0.5 * vx ** 2)
    m = n > 0.5
    z, vx, Tpec = z[m], vx[m], Tpec[m]
    pxz_series = read_timeseries("day1_stress.dat", col=1)
    P = read_params("day1_params.txt")
    Lz, wall_th, vwall = P["Lz"], P["wall_th"], P["vwall"]
    zwlo, zwhi = wall_th, Lz - wall_th
    h = zwhi - zwlo

    zc, W = 0.5 * (z.min() + z.max()), z.max() - z.min()
    cen = (z > zc - 0.25 * W) & (z < zc + 0.25 * W)
    s, c = np.polyfit(z[cen], vx[cen], 1)
    resid = vx[cen] - (s * z[cen] + c)
    ss_tot = np.sum((vx[cen] - vx[cen].mean()) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else float("nan")

    ns_slope = 2 * vwall / h
    v_face_bot, v_face_top = s * zwlo + c, s * zwhi + c
    pxz = pxz_series.mean()

    print("\nSheet 3: viscosity")
    print(f"    central shear rate dvx/dz = {s:+.4f}  (R^2={r2:.3f})")

    # same check as slip: in steady Couette the fluid cannot shear faster
    # than the walls drive it (a violation poisons the shear rate, hence eta).
    tol = 1.15
    if abs(s) > tol * ns_slope or max(abs(v_face_bot), abs(v_face_top)) > tol * vwall:
        print(f"    fluid velocity at the walls = {v_face_bot:+.3f} / {v_face_top:+.3f} "
              f"(wall speed +/-{vwall:g})")
        sys.exit("    FAIL: the fluid is shearing faster than the walls drive it -> unphysical\n"
                 "    (check the drive-velocity units / steady state); eta would be meaningless.")

    eta = abs(pxz) / abs(s) if abs(s) > 1e-9 else float("nan")
    print(f"    mean shear stress pxz = {pxz:+.4f}")
    print(f"    viscosity eta = |pxz| / (dvx/dz) ~ {eta:.3f}")
    print("    (single-run values vary with the seed and the run length.)")

    Tcen = Tpec[len(Tpec) // 2]
    print(f"    fluid temperature T(z): walls ~{Tpec[0]:.2f} / {Tpec[-1]:.2f}, centre ~{Tcen:.2f}")
    print("    (flat under gentle shear; a hot-centre hump means viscous heating ->")
    print("     the channel is no longer isothermal, so a single eta no longer describes it.)")
    plot(z, vx, s, c, zwlo, zwhi, Tpec, pxz_series, eta)

if __name__ == "__main__":
    main()
