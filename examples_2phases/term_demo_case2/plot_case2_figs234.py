"""Reproduce the LAYOUT of Figs. 2, 3, 4 of trans_v116_n1_pp1251-1254.pdf (Shaver et al.,
Second Test Case): velocities (continuous/dispersed/mixture), volume fractions
(continuous/dispersed), and mixture pressure, each vs the flow direction z (this case's x).

*** "Mixture velocity" has two different definitions -- use the mass-weighted one ***
fluidSolver_t's own "mixture" field (AT-2 eq 81) is the VOLUME-fraction-weighted average
u_bar = Sum_k alpha_k*u_k -- for this constant-density-per-phase, no-phase-change case, that
is exactly the total volumetric flux, which is trivially conserved (=1 everywhere) and is
NOT a real result. Table III's u_m=0.9473 (and Fig. 2's "Mixture" curve) is the MASS-weighted
velocity instead (AT-2 eq 85), rho_m*u_m = Sum_k alpha_k*rho_k*u_k -- recomputed here from
u_c/u_d/alpha_d directly, not from the sampled "um" field.

*** Known model mismatch (read before comparing numbers) *** -- case2.udf's drag law
substitutes a Schiller-Naumann C_D(Re_d) correlation (with an under-sized reference
viscosity, NU_C=1e-4) in place of the paper's own drag closure; this materially changes the
fully-developed balance point (see README.md's "AT-2 drag law" section and the session notes
on back-solving Table III's own implied C_D/Re). So the curves below have the RIGHT
qualitative shape/direction as Figs. 2-4 but do NOT quantitatively land on Table III -- a
drag-law-corrected re-run (case2_at2drag.*, see dbg/) is the one to compare quantitatively.

Mixture pressure has no absolute reference in this model (only dP/dz is physically
meaningful, eq 19) -- plotted shifted so p(z=1)=0, matching the paper's own Fig. 4 convention
(pressure decreasing from inlet to outlet, outlet near 0).

*** BD1 curve (matching the paper's own Nek-2P/BD1/FD convention) ***
Shaver's own Figs. 2-4 plot Nek-2P as discrete markers, the independent 1D model (their own
"BD1", a first-order-backward-difference-scheme solve of eqs 14-19) as a continuous curve,
and the fully-developed force-balance point ("FD") as a dashed reference line. We have our
OWN BD1 equivalent: wk_claude/scratch/steady_1d_profile*.py solves the SAME eqs 14-19 (mass-
flux conservation + momentum balance with pressure eliminated into a scalar ODE for u_d(z)),
using whichever drag law matches the nekRS run being plotted -- NOT Shaver's own (genuinely
unpublished) BD1 curve, which we cannot reproduce without their C_D(Re) correlation. This
script plots nekRS as "x" markers and our own BD1-equivalent as a solid curve, same as the
paper's layout; see README.md's "AT-2 drag law" section for why our BD1-equivalent and
Table III agree closely for the AT-2 drag law but not for the production Schiller-Naumann-
only one.
"""
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fe_read import load

# Table I
RHO_C, RHO_D = 1.0, 0.2

# Table II (inlet, z=0)
UC_IN, UD_IN, ALPHAD_IN = 1.0, 1.0, 0.2083
ALPHAC_IN = 1.0 - ALPHAD_IN
UM_IN = (ALPHAC_IN * RHO_C * UC_IN + ALPHAD_IN * RHO_D * UD_IN) / (ALPHAC_IN * RHO_C + ALPHAD_IN * RHO_D)

# Table III (fully-developed / outlet reference)
UC_FD, UD_FD, ALPHAD_FD, UM_FD = 0.9318, 1.386, 0.1504, 0.9473
ALPHAC_FD = 1.0 - ALPHAD_FD

BD1_PRODUCTION = "/Disk_K/src/nekrs_yslan_2phase/wk_claude/scratch/steady_1d_profile.npz"


def latest_vts(vts_dir, prefix):
    files = sorted(glob.glob(f"{vts_dir}/{prefix}[0-9][0-9][0-9][0-9][0-9].vts"))
    if not files:
        raise RuntimeError(f"no {prefix}*.vts dumps found in {vts_dir!r} -- run the case first")
    return files[-1]


def main():
    # Usage: python3 plot_case2_figs234.py [vts_dir] [out_dir] [bd1_npz] [prefix]
    # vts_dir (optional): directory to glob <prefix>*.vts from (default ".", production run).
    # out_dir (optional): directory to write case2_fig{2,3,4}_*.png into (default ".",
    #   production run) -- created if missing. Each variant gets its OWN folder (e.g.
    #   dbg/figs/at2drag/) with plain filenames, instead of suffixing the production
    #   filenames, so comparisons don't clutter term_demo_case2/'s own top level.
    # bd1_npz (optional): our own BD1-equivalent reference profile (see module docstring),
    #   matching whichever drag law the vts_dir run actually used -- default is the
    #   production (Schiller-Naumann) one; pass the AT-2 one for case2_at2drag* runs.
    # prefix (optional): fieldExtract's own sampler label (its casename arg), default "case2"
    #   (production). Each dbg/case2_at2drag* variant now samples under its OWN casename
    #   (e.g. "case2_at2drag_visc1e4") instead of the shared "case2" -- multiple variants'
    #   dumps can coexist in the same dbg/ dir without one clobbering another's .vts files.
    vts_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    bd1_npz = sys.argv[3] if len(sys.argv) > 3 else BD1_PRODUCTION
    prefix = sys.argv[4] if len(sys.argv) > 4 else "case2"
    os.makedirs(out_dir, exist_ok=True)

    bd1 = np.load(bd1_npz)
    zb, ucb, udb, umb = bd1["z"], bd1["uc"], bd1["ud"], bd1["um"]
    acb, adb, pb = bd1["alphac"], bd1["alphad"], bd1["p"]

    fname = latest_vts(vts_dir, prefix)
    d = load(fname)
    x = d["X"]
    order = np.argsort(x)
    x = x[order]
    uc = d["uc"][order]
    ud = d["ud"][order]
    p = d["p"][order]
    alphad = d["alphad"][order]
    alphac = 1.0 - alphad
    t = d["time"]

    # Mass-weighted mixture velocity (AT-2 eq 85), NOT the sampled "um" field (AT-2 eq 81's
    # volume-weighted u_bar -- see module docstring).
    rho_m = alphac * RHO_C + alphad * RHO_D
    um = (alphac * RHO_C * uc + alphad * RHO_D * ud) / rho_m

    p = p - p[-1]  # shift so outlet (z=1) is 0, matching Fig. 4's own convention

    print(f"file = {fname}, time = {t}")
    print(f"{'':16s}{'u_c':>10s}{'u_d':>10s}{'u_m':>10s}{'alpha_c':>10s}{'alpha_d':>10s}")
    print(f"{'Table II (in)':16s}{UC_IN:10.4f}{UD_IN:10.4f}{UM_IN:10.4f}{ALPHAC_IN:10.4f}{ALPHAD_IN:10.4f}")
    print(f"{'nekRS (z~0)':16s}{uc[0]:10.4f}{ud[0]:10.4f}{um[0]:10.4f}{alphac[0]:10.4f}{alphad[0]:10.4f}")
    print(f"{'Table III (FD)':16s}{UC_FD:10.4f}{UD_FD:10.4f}{UM_FD:10.4f}{ALPHAC_FD:10.4f}{ALPHAD_FD:10.4f}")
    print(f"{'nekRS (z~1)':16s}{uc[-1]:10.4f}{ud[-1]:10.4f}{um[-1]:10.4f}{alphac[-1]:10.4f}{alphad[-1]:10.4f}")

    # Fig. 2: velocities
    fig2, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(x[::2], uc[::2], "x", color="tab:blue", label="Continuous - nekRS")
    ax.plot(x[::2], ud[::2], "x", color="tab:red", label="Dispersed - nekRS")
    ax.plot(x[::2], um[::2], "x", color="k", label="Mixture - nekRS")
    ax.plot(zb, ucb, "-", color="tab:blue", lw=1.2, label="Continuous - BD1")
    ax.plot(zb, udb, "-", color="tab:red", lw=1.2, label="Dispersed - BD1")
    ax.plot(zb, umb, "-", color="k", lw=1.2, label="Mixture - BD1")
    for val, color in [(UC_FD, "tab:blue"), (UD_FD, "tab:red"), (UM_FD, "k")]:
        ax.axhline(val, color=color, ls="--", lw=1)
    ax.plot([], [], "--", color="gray", label="Table III (FD)")
    ax.set_xlabel("z")
    ax.set_ylabel("velocity")
    ax.set_title(f"Fig. 2 analog: velocities (t = {t:.3g})")
    ax.legend(fontsize=7, ncol=2)
    fig2.tight_layout()
    outname = os.path.join(out_dir, "case2_fig2_velocities.png")
    fig2.savefig(outname, dpi=150)
    print(f"wrote {outname}")

    # Fig. 3: volume fractions
    fig3, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(x[::2], alphac[::2], "x", color="tab:blue", label="Continuous - nekRS")
    ax.plot(x[::2], alphad[::2], "x", color="tab:red", label="Dispersed - nekRS")
    ax.plot(zb, acb, "-", color="tab:blue", lw=1.2, label="Continuous - BD1")
    ax.plot(zb, adb, "-", color="tab:red", lw=1.2, label="Dispersed - BD1")
    ax.axhline(ALPHAC_FD, color="tab:blue", ls="--", lw=1)
    ax.axhline(ALPHAD_FD, color="tab:red", ls="--", lw=1)
    ax.plot([], [], "--", color="gray", label="Table III (FD)")
    ax.set_xlabel("z")
    ax.set_ylabel("volume fraction")
    ax.set_title(f"Fig. 3 analog: volume fractions (t = {t:.3g})")
    ax.legend(fontsize=8)
    fig3.tight_layout()
    outname = os.path.join(out_dir, "case2_fig3_volfrac.png")
    fig3.savefig(outname, dpi=150)
    print(f"wrote {outname}")

    # Fig. 4: pressure
    fig4, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(x[::2], p[::2], "x", color="k", label="nekRS")
    ax.plot(zb, pb, "-", color="k", lw=1.2, label="BD1")
    ax.set_xlabel("z")
    ax.set_ylabel("pressure (shifted, p(z=1)=0)")
    ax.set_title(f"Fig. 4 analog: pressure (t = {t:.3g})")
    ax.legend(fontsize=8)
    fig4.tight_layout()
    outname = os.path.join(out_dir, "case2_fig4_pressure.png")
    fig4.savefig(outname, dpi=150)
    print(f"wrote {outname}")


if __name__ == "__main__":
    main()
