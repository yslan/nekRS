"""Reproduce the spirit of Fig. 2/3 of trans_v116_n1_pp1251-1254.pdf (Shaver et al.,
Second Test Case): continuous/dispersed velocity (u_c, u_d) and dispersed volume fraction
(alpha_d) along the flow direction, compared against Table II (inlet, z=0) and Table III
(fully-developed / outlet, z->large) values.

*** Schiller-Naumann drag TEMPLATE SUBSTITUTION (see case2.udf header comment) *** -- the
paper's own variable-C_D(Re_d) correlation isn't given on the 4 available pages, so this is
a loose sanity check ("does it converge to something physically plausible"), not a tight
quantitative verification.

All physics (the comparison, the plot) lives here -- the UDF only sampled.
"""
import glob

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fe_read import load

# Table II: inlet conditions (z=0)
UC_IN, UD_IN, ALPHAD_IN = 1.0, 1.0, 0.2083

# Table III: fully-developed conditions (independent BD1 model's answer)
UC_FD, UD_FD, ALPHAD_FD, UM_FD = 0.9318, 1.386, 0.1504, 0.9473


def latest_vts():
    files = sorted(glob.glob("case2*.vts"))
    if not files:
        raise RuntimeError("no case2*.vts dumps found -- run the case first")
    return files[-1]


def main():
    fname = latest_vts()
    d = load(fname)
    x = d["X"]
    order = np.argsort(x)
    x = x[order]
    uc = d["uc"][order]
    ud = d["ud"][order]
    alphad = d["alphad"][order]
    t = d["time"]

    i0, i1 = 0, -1  # inlet (x~0), outlet (x~1)

    print(f"file = {fname}, time = {t}")
    print("")
    print(f"{'':12s}{'u_c':>10s}{'u_d':>10s}{'alpha_d':>10s}")
    print(f"{'Table II (in)':12s}{UC_IN:10.4f}{UD_IN:10.4f}{ALPHAD_IN:10.4f}")
    print(f"{'nekRS (x~0)':12s}{uc[i0]:10.4f}{ud[i0]:10.4f}{alphad[i0]:10.4f}")
    print("")
    print(f"{'Table III (FD)':14s}{UC_FD:10.4f}{UD_FD:10.4f}{ALPHAD_FD:10.4f}")
    print(f"{'nekRS (x~1)':14s}{uc[i1]:10.4f}{ud[i1]:10.4f}{alphad[i1]:10.4f}")
    print("")
    for name, sim, ref in [("u_c", uc[i1], UC_FD), ("u_d", ud[i1], UD_FD), ("alpha_d", alphad[i1], ALPHAD_FD)]:
        rel_err = abs(sim - ref) / abs(ref)
        print(f"outlet {name}: nekRS={sim:.4f}  Table III={ref:.4f}  rel.err={rel_err:.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(x, uc, "-", color="tab:blue", label="nekRS u_c")
    ax.plot(x, ud, "-", color="tab:red", label="nekRS u_d")
    ax.plot([0], [UC_IN], "o", color="tab:blue", mfc="none", label="Table II (inlet)")
    ax.plot([0], [UD_IN], "o", color="tab:red", mfc="none")
    ax.plot([1], [UC_FD], "s", color="tab:blue", mfc="none", label="Table III (fully-dev.)")
    ax.plot([1], [UD_FD], "s", color="tab:red", mfc="none")
    ax.set_xlabel("x (= paper's z)")
    ax.set_ylabel("velocity")
    ax.legend(fontsize=8)
    ax.set_title(f"u_c, u_d  (t = {t:.3g})")

    ax = axes[1]
    ax.plot(x, alphad, "-", color="tab:green", label="nekRS alpha_d")
    ax.plot([0], [ALPHAD_IN], "o", color="tab:green", mfc="none", label="Table II (inlet)")
    ax.plot([1], [ALPHAD_FD], "s", color="tab:green", mfc="none", label="Table III (fully-dev.)")
    ax.set_xlabel("x (= paper's z)")
    ax.set_ylabel("alpha_d")
    ax.legend(fontsize=8)
    ax.set_title(f"alpha_d  (t = {t:.3g})")

    fig.suptitle(f"term_demo_case2 vs. Table II/III (Shaver et al. 2017) -- t = {t:.3g}")
    fig.tight_layout()
    fig.savefig("case2_fig_comparison.png", dpi=150)
    print("wrote case2_fig_comparison.png")


if __name__ == "__main__":
    main()
