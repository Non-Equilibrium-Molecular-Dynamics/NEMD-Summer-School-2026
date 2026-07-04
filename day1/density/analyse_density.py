#!/usr/bin/env python3
"""Day 1 - interfacial density / layering.

Reads the fine density profile written by density.in, reports the bulk density
and the near-wall layering, and saves a plot of rho(z) to day1_density.png.
Run after `lmp_serial -in density.in`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lammps_io import read_profile

def plot(z, rho, rho_bulk, rho_peak, out="day1_density.png"):
    """Save rho(z) with z vertical, so it reads like the channel side-on."""
    try:
        import matplotlib
        if not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")     # headless: just save the PNG
        import matplotlib.pyplot as plt
    except ImportError:
        print("    (matplotlib not found - skipping the plot)")
        return
    plt.plot(rho, z)                       # density on x, height z up the y-axis
    # mark the two values the sheet reads off the profile
    plt.axvline(rho_bulk, color="#999", ls="--", lw=0.9,
                label=r"$\rho_{\mathrm{bulk}} \approx %.2f$" % rho_bulk)
    ipk = rho.argmax()
    plt.plot(rho[ipk], z[ipk], "o", color="#B23A2E", ms=6,
             label=r"$\rho_{\mathrm{peak}} \approx %.2f$" % rho_peak)
    plt.legend(fontsize=8, frameon=False, loc="center left")
    plt.xlabel(r"number density $\rho(z)$")
    plt.ylabel(r"$z\ (\sigma)$")
    plt.title("density profile across the channel")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"    plot -> {out}")
    if os.environ.get("DISPLAY"):     # ssh -X: also pop the figure up on screen
        try:
            plt.show()
        except KeyboardInterrupt:      # Ctrl-C with the window up: fine, the file is already saved
            pass

def main():
    if not os.path.exists("day1_density.profile"):
        sys.exit("day1_density.profile not found. If you submitted with ../submit.sh, wait for "
                 "the job (squeue --me) to finish; otherwise run `lmp_serial -in density.in` first.")
    z, n, rho = read_profile("day1_density.profile")
    m = n > 0.5                        # keep only bins that actually hold atoms
    z, rho = z[m], rho[m]

    W = z.max() - z.min()
    bulk = (z > z.min() + 0.3 * W) & (z < z.max() - 0.3 * W)
    rho_bulk = rho[bulk].mean()
    rho_peak = rho.max()
    contrast = rho_peak / rho_bulk

    print("\nSheet 1: density")
    print(f"    bulk number density  rho_bulk ~ {rho_bulk:.3f}")
    print(f"    near-wall peak rho    ~ {rho_peak:.3f}  (layering contrast {contrast:.2f}x)")
    if contrast > 1.2:
        print("    -> the liquid is NOT uniform: it stacks into layers against the wall.")
    else:
        print("    -> no layering left: the liquid meets the wall essentially uniform.")
    plot(z, rho, rho_bulk, rho_peak)

if __name__ == "__main__":
    main()
