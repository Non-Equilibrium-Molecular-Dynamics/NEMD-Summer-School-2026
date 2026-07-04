#!/usr/bin/env python3
"""Day 1 - live overview of all four runs (a live view, not a measurement result).

One dashboard over all four Day-1 measurements at once -- handy to project at the
front of the room, or to watch your own runs evolve. It reads each measurement's
output from its own folder and lays them out in a single 2x2 figure, one panel per
measurement:

  density rho(z)                <- density/day1_density.profile
  velocity v_x(z)               <- slip/day1_vx.profile
  shear-stress running mean     <- viscosity/day1_stress.dat   (-> eta)
  temperature T(z) / bath heat  <- conductance/day1_Tz.profile, conductance/day1_heat.dat

Run it from this top-level folder (the one holding the four case folders):
  python3 dashboard.py            read once, save day1_overview.png, exit
  python3 dashboard.py --watch    refresh live (re-read every ~1 s) so you can watch
                                  the noisy stress trace and the bath-heat slopes
                                  converge. Ctrl-C saves the final PNG and exits.

It prints no measurement result; it is only a live overview. If matplotlib is unavailable it falls back to a short text summary so it
still runs on a bare node. Panels with no data yet show a "waiting" placeholder, so it
is safe to launch before, during, or after any subset of the four runs.
"""
import argparse
import os
import sys

import numpy as np

BLUE, RED, GREY = "#1F3A5F", "#B23A2E", "#888888"


# ---------- readers: imported from the shared lammps_io module -----------------
# dashboard is a LIVE poller, so it uses the fail-SOFT variants (return None on an
# absent or half-written file) rather than the analysers' fail-LOUD read_profile /
# read_timeseries, which exit on bad data. lammps_io sits in this same folder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lammps_io import read_chunk_or_none as read_chunk, read_series_or_none as read_series


def running_mean(y):
    """Cumulative mean y[:k].mean() for each k -- the trace that should flatten."""
    return np.cumsum(y) / np.arange(1, len(y) + 1)


# ---------- text fallback (no matplotlib) ----------

def text_summary():
    """Print a one-line headline per measurement that has produced output."""
    print("\nDay-1 overview (text mode -- matplotlib not available)")
    print("-" * 56)
    any_data = False

    d = read_chunk("density/day1_density.profile")
    if d is not None:
        z, n, rho = d
        m = n > 0.5
        if m.any():
            zc = z[m]; r = rho[m]
            W = zc.max() - zc.min()
            bulk = r[(zc > zc.min() + 0.3 * W) &
                     (zc < zc.max() - 0.3 * W)]
            print(f"Sheet 1  density   bulk rho ~ {np.mean(bulk):.3f}   peak ~ {r.max():.3f}")
            any_data = True

    v = read_chunk("slip/day1_vx.profile")
    if v is not None:
        z, n, vx = v
        m = n > 0.5
        if m.sum() >= 2:
            s = np.polyfit(z[m], vx[m], 1)[0]
            print(f"Sheet 2/3 velocity central shear rate dvx/dz ~ {s:+.4f}")
            any_data = True

    s = read_series("viscosity/day1_stress.dat", 2)
    if s is not None:
        pxz = s[1]
        print(f"Sheet 3  stress    mean p_xz ~ {pxz.mean():+.4f}   ({len(pxz)} samples)")
        any_data = True

    T = read_chunk("conductance/day1_Tz.profile")
    if T is not None:
        z, n, t = T
        m = (n > 0.5) & (t > 1e-6)
        if m.any():
            print(f"Sheet 4  temperature T(z) spans {t[m].min():.3f} .. {t[m].max():.3f}")
            any_data = True
    h = read_series("conductance/day1_heat.dat", 3)
    if h is not None:
        print(f"Sheet 4  bath heat last q_bot/q_top = {h[1][-1]:+.3f} / {h[2][-1]:+.3f} "
              f"({len(h[0])} samples)")
        any_data = True

    if not any_data:
        print("no day1_* output files found in this directory yet.")
    print("-" * 56)


# ---------- panels ----------

def _placeholder(ax, title):
    ax.text(0.5, 0.5, "waiting for data...", ha="center", va="center",
            color=GREY, fontsize=11, transform=ax.transAxes)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])


def draw(axes):
    """Clear and redraw all four panels from the files currently on disk."""
    axD, axV, axS, axC = axes
    for ax in axes:
        ax.clear()

    # Sheet 1: density rho(z)
    d = read_chunk("density/day1_density.profile")
    if d is not None and (d[1] > 0.5).any():
        z, n, rho = d; m = n > 0.5
        axD.plot(z[m], rho[m], color=BLUE, lw=1.3)
        axD.set_xlabel(r"$z\ (\sigma)$"); axD.set_ylabel(r"$\rho(z)$")
        axD.set_title("Sheet 1: density"); axD.grid(alpha=0.25)
    else:
        _placeholder(axD, "Sheet 1: density")

    # Sheet 2/3: velocity v_x(z)
    v = read_chunk("slip/day1_vx.profile")
    if v is not None and (v[1] > 0.5).sum() >= 2:
        z, n, vx = v; m = n > 0.5
        axV.scatter(z[m], vx[m], s=12, color=BLUE, alpha=0.75)
        fit = np.polyfit(z[m], vx[m], 1)
        s = fit[0]
        zl = np.linspace(z[m].min(), z[m].max(), 50)
        axV.plot(zl, np.polyval(fit, zl), color=RED, lw=1.2)
        axV.set_xlabel(r"$z\ (\sigma)$"); axV.set_ylabel(r"$v_x(z)$")
        axV.set_title(r"Sheet 2/3: velocity  ($\mathrm{d}v_x/\mathrm{d}z=%+.4f$)" % s)
        axV.grid(alpha=0.25)
    else:
        _placeholder(axV, "Sheet 2/3: velocity")

    # Sheet 3: shear-stress running mean (the live-convergence panel)
    s = read_series("viscosity/day1_stress.dat", 2)
    if s is not None:
        pxz = s[1]; k = np.arange(1, len(pxz) + 1)
        axS.plot(k, pxz, color=GREY, lw=0.6, alpha=0.7, label="instantaneous")
        run = running_mean(pxz)
        axS.plot(k, run, color=RED, lw=1.6,
                 label=r"running mean $%+.4f$" % run[-1])
        axS.set_xlabel("sample"); axS.set_ylabel(r"$p_{xz}$")
        axS.set_title("Sheet 3: shear stress (running mean)")
        axS.legend(fontsize=7.5, frameon=False); axS.grid(alpha=0.25)
    else:
        _placeholder(axS, "Sheet 3: shear stress")

    # Sheet 4: temperature T(z) if the profile is in, else the live bath-heat trace
    T = read_chunk("conductance/day1_Tz.profile")
    h = read_series("conductance/day1_heat.dat", 3)
    if T is not None and ((T[1] > 0.5) & (T[2] > 1e-6)).any():
        z, n, t = T; m = (n > 0.5) & (t > 1e-6)
        axC.scatter(z[m], t[m], s=12, color=GREY, alpha=0.7)
        axC.set_xlabel(r"$z\ (\sigma)$"); axC.set_ylabel(r"$T(z)$")
        axC.set_title("Sheet 4: temperature"); axC.grid(alpha=0.25)
    elif h is not None:
        axC.plot(h[0], h[1], color=RED, lw=1.3, label=r"$q_{\mathrm{bot}}$")
        axC.plot(h[0], h[2], color=BLUE, lw=1.3, label=r"$q_{\mathrm{top}}$")
        axC.set_xlabel("timestep"); axC.set_ylabel("cumulative bath heat")
        axC.set_title("Sheet 4: bath heat (-> steady state)")
        axC.legend(fontsize=7.5, frameon=False); axC.grid(alpha=0.25)
    else:
        _placeholder(axC, "Sheet 4: temperature / heat")


def main():
    ap = argparse.ArgumentParser(description="Day-1 instructor overview dashboard.")
    ap.add_argument("--watch", action="store_true",
                    help="refresh live until Ctrl-C (default: render once and exit)")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between refreshes in --watch mode (default 1.0)")
    ap.add_argument("--out", default="day1_overview.png", help="figure file to save")
    args = ap.parse_args()

    try:
        import matplotlib
        if not args.watch and not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")          # headless: just save the PNG
        import matplotlib.pyplot as plt
    except ImportError:
        text_summary()
        return

    def new_fig():
        fig, axes2d = plt.subplots(2, 2, figsize=(9.5, 7.0))
        fig.suptitle("NEMD Summer School 2026 - Day 1 overview (instructor view)",
                     color=BLUE, fontsize=12)
        return fig, list(axes2d.flat)

    if not args.watch:
        fig, axes = new_fig()
        draw(axes)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(args.out, dpi=150)
        print(f"overview -> {args.out}")
        if os.environ.get("DISPLAY"):  # ssh -X: also pop the overview up on screen
            try:
                plt.show()
            except KeyboardInterrupt:  # Ctrl-C with the window up: fine, the file is already saved
                pass
        return

    # live mode: needs an interactive backend; if there is no display, degrade to
    # a single static render + save rather than crashing.
    try:
        plt.ion()
        fig, axes = new_fig()
        print(f"watching day1_* (every {args.interval:g}s); Ctrl-C to save {args.out} and exit.")
        while True:
            draw(axes)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.canvas.draw_idle()
            plt.pause(args.interval)
    except KeyboardInterrupt:
        pass
    except Exception as e:                 # no GUI backend, closed window, etc.
        print(f"    (live display unavailable: {e}; saving a static overview instead)")
        matplotlib.use("Agg")
        fig, axes = new_fig()
        draw(axes)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=150)
    print(f"\noverview -> {args.out}")


if __name__ == "__main__":
    main()
