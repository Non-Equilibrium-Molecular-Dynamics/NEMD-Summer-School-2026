#!/usr/bin/env python3
"""Day 1 - interfacial thermal conductance (Kapitza).

Reads the temperature profile T(z) and the cumulative wall-bath heat written by
conductance.in and reports the interfacial conductance G = J / dT at each wall:
the heat current per unit area divided by the temperature JUMP from the wall
surface to the fluid. Run after, e.g.

    lmp_serial -in conductance.in
    python analyse_conductance.py

The wall-surface temperature is OCCUPANCY-WEIGHTED (weighted by the bin atom
count). The surface window spans the outermost ~SURF sigma of the wall: it holds
the dense outermost FCC plane PLUS a sparse bin that falls BETWEEN atomic planes
(Ncount ~ 1). An unweighted mean lets that near-empty inter-plane bin drag the
wall temperature far off the true surface-plane value; weighting by Ncount gives
it negligible weight against the dense plane. (The wall-fluid depletion gap sits
just OUTSIDE the wall and is excluded by the window's z < wall-face bound -- so
it is the sparse WALL bin, not the gap bin, that the weighting corrects.)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lammps_io import read_profile, read_params, read_timeseries

SURF = 1.2   # sigma: thermostatted wall layer sampled right at each interface


def linfit(x, y):
    """Least-squares slope, intercept, R^2 (guarded)."""
    if len(x) < 2 or np.allclose(x, x[0]):
        return np.nan, np.nan, np.nan
    s, b = np.polyfit(x, y, 1)
    yhat = s * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return s, b, r2


def plot(z, T, z_wall_bot, z_wall_top, sT, bT, Tw_bot, Tf_bot, Tw_top, Tf_top,
         out="day1_conductance.pdf"):
    """T(z) across the channel (z vertical, side-on) with the interfacial
    temperature JUMP marked at each wall: the weighted wall-surface T (circle)
    against the fluid conduction line extrapolated to the same face (square).
    The horizontal gap between them is dT, the Kapitza jump."""
    try:
        import matplotlib
        if not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")     # headless: just save the PNG
        import matplotlib.pyplot as plt
    except ImportError:
        print("    (matplotlib not found - skipping the plot)")
        return
    BLUE, RED = "#1F3A5F", "#B23A2E"
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(T, z, s=12, color="#888", alpha=0.6, label=r"$T(z)$ bins")
    zf = np.linspace(z_wall_bot, z_wall_top, 50)
    ax.plot(sT * zf + bT, zf, color=BLUE, lw=1.4, label="fluid conduction fit")
    ax.axhline(z_wall_bot, color="#888", ls=":", lw=0.9)
    ax.axhline(z_wall_top, color="#888", ls=":", lw=0.9)
    for Tw, Tf, zw in ((Tw_bot, Tf_bot, z_wall_bot), (Tw_top, Tf_top, z_wall_top)):
        ax.plot([Tw], [zw], "o", color=RED, ms=7, zorder=5)
        ax.plot([Tf], [zw], "s", color=BLUE, ms=6, zorder=5)
        ax.annotate("", xy=(Tw, zw), xytext=(Tf, zw),
                    arrowprops=dict(arrowstyle="<->", color=RED, lw=1.0))
        ax.text(0.5 * (Tw + Tf), zw + 0.4, r"$\Delta T=%.2f$" % abs(Tw - Tf),
                color=RED, ha="center", va="bottom", fontsize=8)
    ax.plot([], [], "o", color=RED, label="wall surface")
    ax.plot([], [], "s", color=BLUE, label="fluid at face")
    ax.set_xlabel(r"$T(z)$"); ax.set_ylabel(r"$z\ (\sigma)$")
    ax.set_title("temperature jump at each wall")
    ax.legend(fontsize=7.5, frameon=True, facecolor="white", framealpha=0.9,
              edgecolor="none", loc="upper right")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"    plot -> {out}")
    if os.environ.get("DISPLAY"):     # ssh -X: also pop the figure up on screen
        try:
            plt.show()
        except KeyboardInterrupt:      # Ctrl-C with the window up: fine, the file is already saved
            pass


def plot_heatflux(z, Jz, z_wall_bot, z_wall_top, out="day1_heatflux.pdf"):
    """J_z(z) in the fluid channel only -- the thermostatted wall bins are excluded
    because their per-atom energy spikes would crush the channel signal off-scale.
    With enough averaging J_z flattens (heat flux is conserved); at Day-1 length it
    is still noisy. The thermal analogue of the velocity profile in slip/viscosity."""
    try:
        import matplotlib
        if not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")     # headless: just save the PNG
        import matplotlib.pyplot as plt
    except ImportError:
        print("    (matplotlib not found - skipping the heat-flux plot)")
        return
    ch = (z > z_wall_bot) & (z < z_wall_top)   # channel bins only (exclude the walls)
    plt.plot(Jz[ch], z[ch])                     # flux on x, height z up the y-axis
    plt.xlabel(r"local heat flux $J_z(z)$")
    plt.ylabel(r"$z\ (\sigma)$")
    plt.title("heat flux across the channel")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"    heat-flux plot -> {out}")
    if os.environ.get("DISPLAY"):     # ssh -X: also pop the figure up on screen
        try:
            plt.show()
        except KeyboardInterrupt:      # Ctrl-C with the window up: fine, the file is already saved
            pass


def main():
    for fn in ("day1_Tz.profile", "day1_heat.dat", "day1_params.txt"):
        if not os.path.exists(fn):
            sys.exit(f"{fn} not found. If you submitted with ../submit.sh, wait for the job "
                     f"(squeue --me) to finish; otherwise run `lmp_serial -in conductance.in` first.")
    P = read_params("day1_params.txt")
    Lz, wall_th, area = P["Lz"], P["wall_th"], P["area"]
    dt, Tbot, Ttop = P["dt"], P["Tbot"], P["Ttop"]
    if abs(Ttop - Tbot) < 1e-9:
        sys.exit("Tbot == Ttop: no gradient was imposed, so there is no "
                 "conductance to measure. Re-run with e.g. -var Ttop 0.9.")
    z_wall_bot, z_wall_top = wall_th, Lz - wall_th

    # ---- temperature profile ------------------------------------------------
    z, N, T = read_profile("day1_Tz.profile")
    m = (N > 0.5) & (T > 1e-6)        # drop empty bins + the athermal anchor layer
    z, N, T = z[m], N[m], T[m]

    # wall SURFACE temperature: Ncount-weighted mean over the innermost
    # thermostatted layer at each face (see module docstring on the weighting).
    bot_win = (z < z_wall_bot) & (z >= z_wall_bot - SURF)
    top_win = (z > z_wall_top) & (z <= z_wall_top + SURF)
    if not bot_win.any() or not top_win.any():
        sys.exit("no wall-surface bins found - check wall_th / bin width.")
    Tw_bot = np.average(T[bot_win], weights=N[bot_win])
    Tw_top = np.average(T[top_win], weights=N[top_win])

    # fluid SURFACE temperature: extrapolate the fluid-interior conduction line
    # to each wall face, so the bulk gradient is NOT folded into the jump. Trim a
    # FIXED physical depth (near-wall layering is ~2 sigma deep regardless of the
    # channel width) rather than a fraction of the gap, so the fit stays clean if
    # the box is made thinner.
    LAYER = 2.0                                     # sigma: near-wall layered depth to skip
    fluid = (z > z_wall_bot + LAYER) & (z < z_wall_top - LAYER)
    if fluid.sum() < 3:
        sys.exit("fluid-interior window < 3 bins (channel too thin for a conduction "
                 "fit) - widen the box (-var Lz_t).")
    sT, bT, r2T = linfit(z[fluid], T[fluid])
    Tf_bot = sT * z_wall_bot + bT
    Tf_top = sT * z_wall_top + bT
    dT_bot = abs(Tf_bot - Tw_bot)
    dT_top = abs(Tf_top - Tw_top)

    # ---- per-wall heat current ----------------------------------------------
    # each bath's |power| (slope of its cumulative tallied heat) crosses THAT
    # wall; J = power / area. Do not average the two -- they are separate walls.
    ts, q_bot, q_top = read_timeseries("day1_heat.dat")
    t = ts * dt
    p_bot, _, r2lo = linfit(t, q_bot)
    p_top, _, r2hi = linfit(t, q_top)
    J_bot, J_top = abs(p_bot) / area, abs(p_top) / area
    G_bot = J_bot / dT_bot if dT_bot > 1e-9 else float("nan")
    G_top = J_top / dT_top if dT_top > 1e-9 else float("nan")
    imbal = abs(abs(p_bot) - abs(p_top)) / max(abs(p_bot), abs(p_top), 1e-12)

    hot = "bottom" if Tbot > Ttop else "top"
    print("\nSheet 4: interfacial conductance")
    print(f"    walls held at Tbot={Tbot:.3f} / Ttop={Ttop:.3f}  ({hot} wall is hot)")
    lin = "linear conduction" if r2T >= 0.95 else "POOR fit -- not yet linear/steady"
    print(f"    fluid-interior dT/dz = {sT:+.4f}  (R^2={r2T:.3f}; {lin})")
    print(f"    bottom: T_wall={Tw_bot:.3f}  T_fluid={Tf_bot:.3f}  dT={dT_bot:.3f}"
          f"   J={J_bot:.4f} -> G_bot={G_bot:.3f}")
    print(f"    top:    T_wall={Tw_top:.3f}  T_fluid={Tf_top:.3f}  dT={dT_top:.3f}"
          f"   J={J_top:.4f} -> G_top={G_top:.3f}")
    p_hot, p_cold = (p_bot, p_top) if Tbot > Ttop else (p_top, p_bot)
    print(f"    heat added at the hot wall / removed at the cold wall = "
          f"{abs(p_hot):.4f} / {abs(p_cold):.4f} per unit time")
    print(f"    imbalance {100*imbal:.1f}%   (energy in equals energy out in a steady run)")

    # ---- checks -------------------------------------------------------------
    # the hot wall must sit ABOVE the fluid it heats; the cold wall below. With
    # the wall-fluid gap bin excluded from the surface window (see Tw above), a
    # flipped jump is an EQUILIBRATION signature, not a weighting failure.
    bot_ok = (Tw_bot > Tf_bot) if Tbot > Ttop else (Tw_bot < Tf_bot)
    top_ok = (Tw_top > Tf_top) if Ttop > Tbot else (Tw_top < Tf_top)
    if not (bot_ok and top_ok):
        print("    WARNING: a wall is on the wrong side of the fluid temperature")
        print("             (hot wall colder than the fluid, or vice versa) -> the")
        print("             interfacial jump is not resolved; lengthen nequil/nprod.")
    # steady state: the bath-power IMBALANCE is the primary signal -- it is the
    # net energy-accumulation rate, so imbal ~ 0 means energy in = out = steady.
    # It is checked on its own, not gated by the R^2 of the cumulative heat: that
    # R^2 stays ~0.99 even mid-transient (it is dominated by its own trend), so it
    # is nearly blind to steadiness. A 5% imbalance threshold is used here; larger
    # values mean the run has not reached steady state and should be lengthened.
    if imbal > 0.05:
        print(f"    WARNING: energy in does not equal energy out (imbalance {100*imbal:.1f}%)")
        print("             -> the run is not steady; lengthen nequil (~40000).")
    if min(r2lo, r2hi) < 0.95:
        print("    WARNING: the cumulative-heat curves are far from straight lines "
              f"(linear-fit R^2 {r2lo:.3f}/{r2hi:.3f}) -> the run is not steady; lengthen nequil.")
    if r2T < 0.95:
        print(f"    WARNING: fluid-interior T(z) fit is poor (R^2={r2T:.3f}) -> the profile")
        print("             is not yet linear/steady, so dT/dz and the extrapolated face")
        print("             temperatures (hence G) are unreliable -- lengthen nequil.")
    if np.isfinite(G_bot) and np.isfinite(G_top) and min(G_bot, G_top) > 0:
        gr = max(G_bot, G_top) / min(G_bot, G_top)
        if gr > 1.4:
            print(f"    NOTE: the two walls give G differing {gr:.1f}x. Some asymmetry is")
            print("          expected (they sit at different T), but a large gap means the")
            print("          jump is under-resolved -- run replicas (-var seed) and average.")
    # (wall T is the occupancy-weighted surface-plane value while fluid T is
    # extrapolated to the face ~0.5 sigma beyond it, so each per-wall G carries a
    # small geometric offset -- methods detail, kept out of the student-facing text.)
    print("    (reduced LJ units; G = J/dT per wall, hot and cold separate.)")

    # ---- optional: local heat-flux profile (only present if -var heatflux 1) ----
    if os.path.exists("day1_heatflux.profile"):
        zJ, _, Jsum = read_profile("day1_heatflux.profile")
        dz = Lz / len(zJ)
        Jz = Jsum / (area * dz)            # bin-summed J_z -> local flux density
        inside = (zJ > z_wall_bot + 2.0) & (zJ < z_wall_top - 2.0)
        if inside.sum() >= 3:
            Jm, Js = Jz[inside].mean(), Jz[inside].std()
            verdict = ("~uniform across the channel (heat flux conserved)"
                       if abs(Js) < 0.3 * abs(Jm) else
                       "still noisy -- lengthen the run to see it flatten")
            print(f"    local heat flux J_z ~ {Jm:+.4f} (interior spread {Js:.4f}) -> {verdict}")
        plot_heatflux(zJ, Jz, z_wall_bot, z_wall_top)

    plot(z, T, z_wall_bot, z_wall_top, sT, bT, Tw_bot, Tf_bot, Tw_top, Tf_top)


if __name__ == "__main__":
    main()
