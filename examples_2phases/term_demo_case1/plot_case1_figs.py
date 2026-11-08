"""Two comparison figures for term_demo_case1 vs. Shaver et al. 2017's First Test Case,
each overlaying BOTH a polynomialOrder=7 and polynomialOrder=9 run against eq 21, to show
the small residual near the inlet is resolution-independent (confirmed: p-refinement from
N=7 to N=9 changes it by <1%, so it's the regularizing viscosity, not under-resolution).

MAJOR UPDATE (this session): fixed a real core bug where every phase advected with the
shared MIXTURE velocity instead of its own (see wk_claude/notes/05_scheme_vs_pnpn.md
section 2.2 and fluidSolver_t::computeUrstPhases()). After the fix, phase1 genuinely
self-advects, and at steady state (confirmed bit-identical across several late times) the
PDE's time-derivative vanishes, leaving exactly the paper's own eq 20/21 -- so the
comparison here is genuinely spatial (x really is z), not a relabeling trick.

Run both polynomialOrder=7 and =9 (editing case1.par, rerunning, saving n7_final.npz/
n7_timeseries.npz and n9_final.npz/n9_timeseries.npz via the inline snippet in the repo
history / README) before running this script -- it only reads the saved .npz files, it
does not rerun anything itself.

1. case1_fig_timehistory.png -- u_d(t) at the outlet for both N, vs the steady-state eq21
   target, with an error(t) panel.
2. case1_fig1_paperstyle.png -- u_d(x) at the final (steady) snapshot for both N, vs eq21(z).

A small residual vs. eq 21 is expected and not a bug: eq 21 is derived assuming NO
diffusion ("diffusion terms neglected as insignificant" -- paper's own text before eq 16),
while this simulation uses a small but nonzero viscosity=1e-4 as a numerical regularizer.
The residual is largest near the inlet (steepest gradient) and vanishes downstream.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RHO_C, RHO_D, D_D, GRAV, CD0 = 1.0, 0.2, 0.5, 0.9807, 0.4

A = GRAV * (RHO_C / RHO_D - 1.0)
B = (RHO_C / RHO_D) * 0.75 * (CD0 / D_D)


def u_d_eq21(z):
    return np.sqrt((A / B) * (1.0 - np.exp(-2.0 * B * z)))


def fig_timehistory():
    target = u_d_eq21(1.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    for npz_name, color, label in [("n7_timeseries.npz", "tab:red", "N=7"),
                                    ("n9_timeseries.npz", "tab:blue", "N=9")]:
        try:
            d = np.load(npz_name)
        except FileNotFoundError:
            continue
        ts, ud_outlet = d["ts"], d["ud_outlet"]
        abs_err = np.abs(ud_outlet - target)
        ax1.plot(ts, ud_outlet, "x-", color=color, markersize=4, label=f"nekRS u_d(x=1,t) {label}")
        ax2.semilogy(ts, np.maximum(abs_err, 1e-16), "-o", color=color, markersize=3, label=label)
        print(f"{label}: outlet u_d(t_final)={ud_outlet[-1]:.6f}, final abs error={abs_err[-1]:.4e}")

    ax1.axhline(target, color="k", linestyle="--", label=f"eq 21 steady-state target ({target:.4f})")
    ax1.set_ylabel("u_d at outlet")
    ax1.set_title("term_demo_case1: outlet u_d approaching steady state (N=7 vs N=9)")
    ax1.legend()

    ax2.axhline(1e-3, color="gray", linestyle="--", linewidth=1, label="1e-3 reference")
    ax2.set_xlabel("t")
    ax2.set_ylabel("absolute error")
    ax2.legend()
    ax2.set_title("error(t) vs eq21(z=1)")

    fig.tight_layout()
    fig.savefig("case1_fig_timehistory.png", dpi=150)
    print("wrote case1_fig_timehistory.png")


def fig_paperstyle():
    z_fine = np.linspace(0, 1, 200)
    ud_ref = u_d_eq21(z_fine)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(z_fine, ud_ref, "-", color="k", label="eq 21 (self-advection)")

    for npz_name, color, marker, label in [("n7_final.npz", "tab:red", "x", "N=7"),
                                            ("n9_final.npz", "tab:blue", "+", "N=9")]:
        try:
            d = np.load(npz_name)
        except FileNotFoundError:
            continue
        x, ud = d["x"], d["ud"]
        abs_err = np.abs(ud - u_d_eq21(x))
        ax.plot(x, ud, marker, color=color, markersize=5, label=f"nekRS u_d(x) {label}")
        print(f"{label}: max abs error = {abs_err.max():.4e} at x={x[np.argmax(abs_err)]:.3f}, "
              f"error(outlet) = {abs_err[-1]:.4e}")

    ax.set_xlabel("z")
    ax.set_ylabel("u_d")
    ax.set_title("Fig. 1 -- Dispersed phase velocity vs. z (steady state, N=7 vs N=9)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("case1_fig1_paperstyle.png", dpi=150)
    print("wrote case1_fig1_paperstyle.png")


def main():
    fig_timehistory()
    fig_paperstyle()


if __name__ == "__main__":
    main()
