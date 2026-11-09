"""Save n{N}_final.npz / n{N}_timeseries.npz from the current directory's case1*.vts dumps,
for plot_case1_figs.py's N=7-vs-N=9 overlay. Run once per polynomialOrder (edit case1.par,
rebuild, run, then this script with that N) -- does not rerun the simulation itself.

Usage: python3 save_case1_npz.py <N>   (N = 7 or 9, whatever polynomialOrder the dumps in
this directory were just produced with)
"""
import glob
import re
import sys

import numpy as np

from fe_read import load


def main():
    N = sys.argv[1] if len(sys.argv) > 1 else "7"

    files = sorted(glob.glob("case1[0-9][0-9][0-9][0-9][0-9].vts"),
                    key=lambda f: int(re.search(r"(\d+)\.vts$", f).group(1)))
    if not files:
        raise RuntimeError("no case1*.vts dumps found -- run the case first")

    ts, ud_outlet = [], []
    for f in files:
        d = load(f)
        order = np.argsort(d["X"])
        x = d["X"][order]
        ud = d["ud"][order]
        ts.append(d["time"])
        ud_outlet.append(ud[-1])
    ts = np.array(ts)
    ud_outlet = np.array(ud_outlet)

    np.savez(f"n{N}_timeseries.npz", ts=ts, ud_outlet=ud_outlet)
    print(f"wrote n{N}_timeseries.npz ({len(ts)} points, t in [{ts[0]:.3f}, {ts[-1]:.3f}])")

    d = load(files[-1])
    order = np.argsort(d["X"])
    x = d["X"][order]
    ud = d["ud"][order]
    np.savez(f"n{N}_final.npz", x=x, ud=ud)
    print(f"wrote n{N}_final.npz (final time={d['time']:.3f}, ud(outlet)={ud[-1]:.6f})")


if __name__ == "__main__":
    main()
