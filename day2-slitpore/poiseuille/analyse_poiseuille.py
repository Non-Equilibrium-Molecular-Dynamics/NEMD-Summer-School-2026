#!/usr/bin/env python3
"""Day 2 (Part 1, Sheet 1) - local viscosity in a confined slit-pore.

Reads the velocity profile vx(z) and mass-density profile rho_m(z) written by
poiseuille.in and forms, on one z-grid across the pore:

  - the shear stress P_xz(z) by momentum balance from the centreline,
        P_xz(z) = +g * integral_0^z rho_m(z') dz'      (g = body force, mass = 1);
  - the local shear rate dvx/dz;
  - the local viscosity eta(z) = -P_xz(z) / (dvx/dz).

WIDE pore (e.g. -var width 10): vx(z) is a parabola, so the shear rate is
taken from a parabola fit and eta(z) is a flat plateau slightly above the
unconfined bulk LJ value eta_0 ~ 2.13 (near-wall layering raises the
pore-averaged value). A local Newtonian viscosity describes the liquid.

NARROW pore (e.g. -var width 4): the density forms wall-induced layers that fill
the whole pore - there is no bulk region. The shear stress P_xz(z) follows that
layering, and dividing by the (noisy, layered) shear rate no longer gives a single
local value: eta(z) has no plateau. Newton's local viscosity law breaks down
(Travis, Todd & Evans, Phys. Rev. E 55, 4288, 1997).

The centreline is a 0/0 point (P_xz and dvx/dz both vanish there by symmetry), so a
window around it is masked.

Run after `lmp_serial -in poiseuille.in` (and `-var width 4` for the narrow Push),
from inside this folder. It analyses every width-tagged profile present.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lammps_io import read_vx_rho, read_params

NARROW_W = 6.0       # below this measured width (sigma) the pore is layer-dominated, no bulk plateau
SMOOTH_SIGMA = 0.5   # boxcar smoothing of vx(z) before finite differencing, in sigma

BLUE, RED, GREY, AMBER = "#1F3A5F", "#B23A2E", "#888888", "#C77F1A"


def smooth(y, win):
    """Reflect-padded boxcar moving average (odd window); win<=1 returns y."""
    if win <= 1:
        return y.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    yp = np.concatenate([y[pad:0:-1], y, y[-2:-pad - 2:-1]])
    return np.convolve(yp, np.ones(win) / win, mode="valid")


def pxz_from_centre(zc, rho, g):
    """Shear stress by momentum balance: +g * integral of rho_m from the centreline."""
    seg = 0.5 * (rho[:-1] + rho[1:]) * np.diff(zc)
    I = np.concatenate([[0.0], np.cumsum(seg)])
    return g * (I - np.interp(0.0, zc, I))


def analyse_one(prof_path, par_path, tag):
    z, n, vx, rho = read_vx_rho(prof_path)
    P = read_params(par_path)
    required = {"z_bot_inner", "z_top_inner", "gforce", "binw", "w_meas"}
    missing = required - P.keys()
    if missing:
        sys.exit(f"{par_path}: missing keys {sorted(missing)} (truncated, or from an earlier run?) - "
                 f"rerun `lmp_serial -in poiseuille.in` from inside poiseuille/.")
    zbi, zti = P["z_bot_inner"], P["z_top_inner"]
    g, binw, wmeas = P["gforce"], P["binw"], P["w_meas"]

    m = (z > zbi) & (z < zti) & (n > 0.5)
    z, vx, rho = z[m], vx[m], rho[m]
    if len(z) < 7:
        print(f"  [{tag}] only {len(z)} fluid bins at w={wmeas:.2f} sigma; run longer or lower binw.")
        return None
    zc = z - 0.5 * (zbi + zti)                          # centreline at zc = 0
    half = 0.5 * wmeas
    maskhw = max(2 * binw, min(0.6, 0.15 * half))
    masked = np.abs(zc) < maskhw

    Pxz = pxz_from_centre(zc, rho, g)

    # densities (the physically meaningful in-pore values, reported not assumed)
    rho_mean = float(np.mean(rho))
    centre = np.abs(zc) < min(1.0, 0.3 * half)
    rho_centre = float(np.mean(rho[centre])) if centre.any() else float("nan")

    wide = wmeas >= NARROW_W
    print(f"\n[{tag}] LOCAL VISCOSITY  (measured pore width w = {wmeas:.2f} sigma, "
          f"{'WIDE' if wide else 'NARROW'})")
    print(f"    in-pore density: mean {rho_mean:.3f}, pore-centre {rho_centre:.3f} "
          f"(target rho* = 0.8)")

    if wide:
        a, b, c = np.polyfit(zc, vx, 2)
        dvdz = 2 * a * zc + b
        eta = np.where(np.abs(dvdz) > 1e-6, -Pxz / dvdz, np.nan)
        eta[masked] = np.nan
        plateau = float(np.nanmedian(eta[(~masked) & (np.abs(zc) < 0.6 * half)]))
        print(f"    parabola fit -> eta(z) plateau (mid-pore) = {plateau:.3f}")
        print(f"    -> compare to the single bulk viscosity from Day-1's sheared channel;")
        print(f"       both are slightly above the unconfined bulk LJ value eta_0 ~ 2.13 (near-wall layering).")
        plot(zc, vx, a * zc**2 + b * zc + c, rho, Pxz, eta, maskhw, plateau, wmeas, tag, wide=True)
        return plateau

    vxs = smooth(vx, max(1, round(SMOOTH_SIGMA / binw)))
    dvdz = np.gradient(vxs, zc)
    eta = np.where(np.abs(dvdz) > 1e-9, -Pxz / dvdz, np.nan)
    eta[masked] = np.nan
    npeak = int((np.diff(np.sign(np.diff(rho))) < 0).sum())   # interior density maxima
    print(f"    rho_m(z) shows {npeak} wall-induced layer(s) across the pore - no bulk region.")
    print(f"    eta(z) = -P_xz/(dvx/dz) has NO single plateau here: a local Newtonian viscosity")
    print(f"    no longer describes the liquid (Travis, Todd & Evans 1997).")
    plot(zc, vx, vxs, rho, Pxz, eta, maskhw, float("nan"), wmeas, tag, wide=False)
    return None


def plot(zc, vx, vxfit, rho, Pxz, eta, maskhw, plateau, wmeas, tag, wide, out=None):
    out = out or f"poiseuille_w{tag}.png"
    try:
        import matplotlib
        if not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")     # headless: just save the PNG
        import matplotlib.pyplot as plt
    except ImportError:
        print("    (matplotlib not found - skipping the plot)")
        return
    # 2x2 panels (velocity, shear stress, density, local viscosity) in the same order
    # as the handout figure for both the wide and narrow pore, so the student plot and
    # the reference figure line up panel-for-panel.
    fig, ax = plt.subplots(2, 2, figsize=(7.6, 7.4), sharey=True)
    ax = ax.ravel()
    # velocity
    ax[0].scatter(vx, zc, s=12, color=GREY, alpha=0.6)
    ax[0].plot(vxfit, zc, color=BLUE, lw=1.8, label="parabola fit" if wide else "smoothed")
    ax[0].set_xlabel(r"$v_x(z)$"); ax[0].set_ylabel(r"$z'\ (\sigma)$"); ax[0].set_title("velocity")
    ax[0].legend(fontsize=7, frameon=False)
    # shear stress
    ax[1].plot(Pxz, zc, color=RED, lw=1.6); ax[1].axvline(0, color=GREY, lw=0.6)
    ax[1].set_xlabel(r"$P_{xz}(z)$"); ax[1].set_title("shear stress")
    # density
    ax[2].plot(rho, zc, color=AMBER, lw=1.8)
    ax[2].set_xlabel(r"$\rho_m(z)$"); ax[2].set_ylabel(r"$z'\ (\sigma)$"); ax[2].set_title("density")
    # local viscosity
    ax[3].axhspan(-maskhw, maskhw, color=GREY, alpha=0.12)
    if wide:
        ax[3].plot(eta, zc, color=BLUE, lw=1.8)
        if not np.isnan(plateau):
            ax[3].axvline(plateau, color=GREY, lw=0.9, ls="--", label=r"mid-pore $\eta \approx %.2f$" % plateau)
            ax[3].legend(fontsize=7, frameon=False)
        ax[3].set_xlim(0, 3)
    else:
        ax[3].plot(eta, zc, color=GREY, lw=1.4)
        ax[3].set_xlim(-6, 10)
    ax[3].set_title("local viscosity")
    ax[3].set_xlabel(r"$\eta(z)$")
    fig.suptitle(f"slit-pore w = {wmeas:.1f} sigma", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=150)
    print(f"    plot -> {out}")


def main():
    profs = sorted(glob.glob("poiseuille_w*.profile"))
    if not profs:
        sys.exit("no poiseuille_w*.profile here. Run `lmp_serial -in poiseuille.in` "
                 "(and `-var width 4`) first, from inside this folder.")
    for prof in profs:
        tag = prof[len("poiseuille_w"):-len(".profile")]
        par = f"poiseuille_w{tag}.params.txt"
        if not os.path.exists(par):
            print(f"  ({par} missing - skipping {prof})")
            continue
        analyse_one(prof, par, tag)
    if os.environ.get("DISPLAY"):      # ssh -X: pop every width's figure up together at the end
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        if plt.get_fignums():
            print("\n(close the plot windows to finish)")
            try:
                plt.show()
            except KeyboardInterrupt:  # Ctrl-C with the windows up: fine, the PNGs are already saved
                pass


if __name__ == "__main__":
    main()
