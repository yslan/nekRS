"""term_demo_case2 time-history plot: u_c, u_d, alpha_d at the outlet (x~1) vs time, plus
each phase's own mass flux (alpha_k*u_k) vs time against its (conserved, per the paper's own
eq 14/15, d(alpha_k*u_k)/dz=0 in the STEADY state) inlet value -- see wk_claude/notes/
07_algorithm_pseudocode.md's "Open issues" and README.md's T=32 retry writeup.

Reads whatever case2*.vts dumps are present (works whether the run completed or aborted
partway through, e.g. run_T32_blowup.log's T=32 attempt, which aborted around t~8-9.6).

Usage: python3 plot_case2_timehistory.py [label]
  label (optional): appended to every output filename, e.g. "solve_alphad" ->
  case2_fig_timehistory_solve_alphad.png + case2_timehistory_solve_alphad.npz. Also saves
  the extracted arrays (ts, uc_out, ud_out, ad_out, flux_c, flux_d) to .npz either way, for
  exact numeric comparison between retries without needing to keep every .vts dump.
"""
import glob
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fe_read import load

UC_IN, UD_IN, ALPHAD_IN = 1.0, 1.0, 0.2083
ALPHAC_IN = 1.0 - ALPHAD_IN
FLUX_C_IN = ALPHAC_IN * UC_IN
FLUX_D_IN = ALPHAD_IN * UD_IN

UC_FD, UD_FD, ALPHAD_FD = 0.9318, 1.386, 0.1504


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else ""
    suffix = f"_{label}" if label else ""

    files = sorted(glob.glob("case2[0-9][0-9][0-9][0-9][0-9].vts"),
                    key=lambda f: int(re.search(r"(\d+)\.vts$", f).group(1)))
    if not files:
        raise RuntimeError("no case2*.vts dumps found -- run the case first")

    ts, uc_out, ud_out, ad_out = [], [], [], []
    for f in files:
        d = load(f)
        order = np.argsort(d["X"])
        uc = d["uc"][order]
        ud = d["ud"][order]
        ad = d["alphad"][order]
        ts.append(d["time"])
        uc_out.append(uc[-1])
        ud_out.append(ud[-1])
        ad_out.append(ad[-1])

    ts = np.array(ts)
    uc_out = np.array(uc_out)
    ud_out = np.array(ud_out)
    ad_out = np.array(ad_out)
    ac_out = 1.0 - ad_out
    flux_c = ac_out * uc_out
    flux_d = ad_out * ud_out

    print(f"{len(files)} dumps, t in [{ts[0]:.3f}, {ts[-1]:.3f}]")
    print(f"outlet at t_final: u_c={uc_out[-1]:.4f} u_d={ud_out[-1]:.4f} alpha_d={ad_out[-1]:.4f}")
    print(f"mass flux at t_final: alpha_c*u_c={flux_c[-1]:.4f} (inlet {FLUX_C_IN:.4f}), "
          f"alpha_d*u_d={flux_d[-1]:.4f} (inlet {FLUX_D_IN:.4f})")

    np.savez(f"case2_timehistory{suffix}.npz",
             ts=ts, uc_out=uc_out, ud_out=ud_out, ad_out=ad_out, flux_c=flux_c, flux_d=flux_d)
    print(f"wrote case2_timehistory{suffix}.npz")

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    ax = axes[0]
    ax.plot(ts, uc_out, "-o", color="tab:blue", markersize=3, label="u_c(x=1,t)")
    ax.plot(ts, ud_out, "-o", color="tab:red", markersize=3, label="u_d(x=1,t)")
    ax.plot(ts, ad_out, "-o", color="tab:green", markersize=3, label="alpha_d(x=1,t)")
    ax.axhline(UC_FD, color="tab:blue", ls="--", lw=1, label=f"u_c Table III ({UC_FD})")
    ax.axhline(UD_FD, color="tab:red", ls="--", lw=1, label=f"u_d Table III ({UD_FD})")
    ax.axhline(ALPHAD_FD, color="tab:green", ls="--", lw=1, label=f"alpha_d Table III ({ALPHAD_FD})")
    ax.axhline(0.0, color="k", ls=":", lw=0.8)
    ax.set_ylabel("u_c, u_d, alpha_d at outlet")
    ax.set_title("term_demo_case2: outlet time history (eq 90 dilation term, retry #2 / Plan K)")
    ax.legend(fontsize=8, loc="center left")

    ax = axes[1]
    ax.plot(ts, flux_c, "-o", color="tab:blue", markersize=3, label="alpha_c*u_c (outlet)")
    ax.plot(ts, flux_d, "-o", color="tab:red", markersize=3, label="alpha_d*u_d (outlet)")
    ax.axhline(FLUX_C_IN, color="tab:blue", ls="--", lw=1, label=f"alpha_c*u_c inlet ({FLUX_C_IN:.4f})")
    ax.axhline(FLUX_D_IN, color="tab:red", ls="--", lw=1, label=f"alpha_d*u_d inlet ({FLUX_D_IN:.4f})")
    ax.set_xlabel("t")
    ax.set_ylabel("phase mass flux alpha_k*u_k")
    ax.set_title("Mass-flux conservation check (eq 14/15: d(alpha_k*u_k)/dz=0 at steady state)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    outname = f"case2_fig_timehistory{suffix}.png"
    fig.savefig(outname, dpi=150)
    print(f"wrote {outname}")


if __name__ == "__main__":
    main()
