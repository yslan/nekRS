"""term_demo_case1 vs. Shaver et al. 2017 (First Test Case) -- eq 21. See
wk_claude/plans/H_two_term_demo_examples.md for the full history; short version:

This session fixed a real core bug (fluidSolver_t advecting every phase with the shared
MIXTURE velocity instead of its own -- see wk_claude/notes/05_scheme_vs_pnpn.md section
2.2). Before the fix, case1's alpha_d==0/u_c==0 setup made the mixture velocity exactly
zero, so phase1 saw no convection and the problem degenerated into a purely temporal
relaxation (no spatial development at all). After the fix, phase1 genuinely self-advects,
and at steady state (confirmed bit-identical across t=3,4.5,6) the PDE's time-derivative
vanishes, leaving exactly the paper's own eq 20: u_d*du_d/dz = A - B*u_d^2. So THIS script
now compares the steady spatial profile u_d(x) directly against eq 21(z) -- x really is z.

A small residual vs. eq 21 is expected, not a bug: eq 21 assumes no diffusion ("diffusion
terms neglected as insignificant" -- paper's own text), while this sim uses a small but
nonzero viscosity=1e-4 as a numerical regularizer. The residual is largest near the inlet
(steepest gradient) and vanishes downstream -- see plot_case1_figs.py for the plots.
"""
import glob
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fe_read import load

RHO_C, RHO_D, D_D, GRAV, CD0 = 1.0, 0.2, 0.5, 0.9807, 0.4

# eq 21 (self-advection): u_d(z) = sqrt(A/B * (1 - exp(-2*B*z)))
A = GRAV * (RHO_C / RHO_D - 1.0)
B = (RHO_C / RHO_D) * 0.75 * (CD0 / D_D)


def u_d_eq21(z):
    return np.sqrt((A / B) * (1.0 - np.exp(-2.0 * B * z)))


def main():
    files = sorted(glob.glob("case1[0-9][0-9][0-9][0-9][0-9].vts"),
                    key=lambda f: int(re.search(r"(\d+)\.vts$", f).group(1)))
    d = load(files[-1])  # final (steady) snapshot
    order = np.argsort(d["X"])
    x = d["X"][order]
    ud = d["ud"][order]
    uc = d["uc"][order]
    ud_ref = u_d_eq21(x)
    abs_err = np.abs(ud - ud_ref)

    print(f"{len(files)} snapshots, final time = {d['time']:.3f}")
    print(f"u_c: range [{uc.min():.4e}, {uc.max():.4e}], expect exactly 0 (undriven, trivial per the paper)")
    print(f"u_d(x=1) sim = {ud[-1]:.6f}, eq21(z=1) = {ud_ref[-1]:.6f}, "
          f"terminal (z->inf) = {np.sqrt(A/B):.6f}")
    print(f"max abs error over the profile = {abs_err.max():.4e} at x={x[np.argmax(abs_err)]:.3f} "
          f"(expected near the inlet, finite-viscosity correction to the inviscid eq 21)")
    print(f"abs error at outlet (x=1) = {abs_err[-1]:.4e}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(x, ud, "x", color="tab:red", markersize=4, label="nekRS u_d(x) (steady)")
    z_fine = np.linspace(0, 1, 200)
    ax.plot(z_fine, u_d_eq21(z_fine), "-", color="k", label="eq 21 (self-advection)")
    ax.plot(x, uc, ".", color="tab:blue", markersize=3, label="nekRS u_c(x)")
    ax.set_xlabel("x (= paper's z)")
    ax.set_ylabel("velocity")
    ax.set_title("term_demo_case1 steady profile vs. Fig. 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig("case1_fig1_comparison.png", dpi=150)
    print("wrote case1_fig1_comparison.png")


if __name__ == "__main__":
    main()
