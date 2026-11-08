"""Shared loader/validator for fieldExtract .vts output.

Analytic reference fields mirror turbPipe.udf::test_sol:
  vz   = 6/5 (1 - (r/R)^6) cos(t)          (r = sqrt(x^2+y^2), R = 0.5)
  temp = sin(pi x) cos(2.7 pi z + 0.2)

Each per-config test script (test_box1d.py, ...) imports check() from here.
"""
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

R = 0.5  # pipe radius


def _parse_adios_value(meta):
    """Parse an ADIOS attribute metadata dict {'Type','Value','Elements'} into a
    python scalar or list. The adios2 2.10 python read_attribute() mis-reads array
    attributes (broadcasts the first element), so we parse the reliable 'Value'
    string instead, e.g. '{ 11, 7, 21 }' -> [11, 7, 21], '1617' -> 1617."""
    val = meta["Value"].strip()
    is_float = "double" in meta["Type"] or "float" in meta["Type"]
    cast = float if is_float else int
    if val.startswith("{"):
        return [cast(x) for x in val.strip("{} ").split(",")]
    return cast(val)


def load_bp(fname):
    """Load a fieldExtract ADIOS .bp (Plan F) into the same dict shape as the .vts
    loader: dims, pointDist, time, X/Y/Z, and one entry per field. Uses the latest
    step (matching the per-step .vts semantics). Coordinates come from the
    'vertices' variable; box metadata from the attributes' Value strings."""
    import adios2  # optional dep; only needed for .bp

    with adios2.FileReader(fname) as f:
        attrs = f.available_attributes()
        variables = set(f.available_variables().keys())
        ns = f.num_steps()
        last = [ns - 1, 1]  # step_selection: start, count

        nx, ny, nz = _parse_adios_value(attrs["boxDims"])
        dist = tuple(_parse_adios_value(attrs["pointDist"])) if "pointDist" in attrs else (0, 0, 0)

        verts = np.asarray(f.read("vertices", step_selection=last)).reshape(-1, 3)
        out = {
            "dims": (int(nx), int(ny), int(nz)),
            "pointDist": dist,
            "time": float(np.asarray(f.read("time", step_selection=last)).ravel()[0]),
            "X": verts[:, 0].astype(np.float64),
            "Y": verts[:, 1].astype(np.float64),
            "Z": verts[:, 2].astype(np.float64),
        }
        # field variables = everything that is not mesh/metadata
        reserved = {"vertices", "connectivity", "types", "time", "CYCLE"}
        for name in sorted(variables - reserved):
            arr = np.asarray(f.read(name, step_selection=last))
            out[name] = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.ravel()
    return out


def load(fname):
    """Return dict with dims, time, X/Y/Z arrays, and one entry per PointData field.
    Dispatches on extension: .bp -> ADIOS reader (Plan F), else VTK .vts reader."""
    if fname.rstrip("/").endswith(".bp"):
        return load_bp(fname)

    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(fname)
    reader.Update()
    grid = reader.GetOutput()

    pts = vtk_to_numpy(grid.GetPoints().GetData())
    fd = grid.GetFieldData()
    pd = grid.GetPointData()

    nx, ny, nz = vtk_to_numpy(fd.GetArray("boxDims")).ravel().astype(int)
    pd_arr = fd.GetArray("pointDist")  # absent in pre-Plan-D files -> uniform
    dist = tuple(vtk_to_numpy(pd_arr).ravel().astype(int)) if pd_arr is not None else (0, 0, 0)
    out = {
        "dims": (int(nx), int(ny), int(nz)),
        "pointDist": dist,  # per-axis code: 0 uniform, 1 GLL
        "time": vtk_to_numpy(fd.GetArray("TimeValue")).ravel().item(),
        "X": pts[:, 0],
        "Y": pts[:, 1],
        "Z": pts[:, 2],
    }
    for i in range(pd.GetNumberOfArrays()):
        arr = pd.GetArray(i)
        out[arr.GetName()] = vtk_to_numpy(arr)
    return out


def ref_vz(X, Y, Z, time):
    rr = np.sqrt(X * X + Y * Y) / R
    W = 6 / 5.0 * (1 - rr**6) * np.cos(time)
    W[rr > 1.0] = 0.0
    return W


def ref_T(X, Y, Z):
    rr = np.sqrt(X * X + Y * Y) / R
    T = np.sin(np.pi * X) * np.cos(2.7 * np.pi * Z + 0.2)
    T[rr > 1.0] = 0.0
    return T


def relerr(a, b):
    denom = np.abs(b).max()
    return np.abs(a - b).max() / denom if denom > 0 else np.abs(a - b).max()


def check(fname, tol=1e-2):
    """Load fname, compare vz/temp to the analytic refs, print + return PASS/FAIL."""
    d = load(fname)
    X, Y, Z, t = d["X"], d["Y"], d["Z"], d["time"]
    ev = relerr(d["vz"], ref_vz(X, Y, Z, t))
    eT = relerr(d["temp"], ref_T(X, Y, Z))
    ok = (ev < tol) and (eT < tol)
    print(
        f"{fname}: dims={d['dims']} time={t:g} "
        f"relerr(vz)={ev:.2e} relerr(temp)={eT:.2e} -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def zwgll(n):
    """GLL nodes/weights on [-1,1] (n points). Ported from the sem_core
    quadrature.py template (zwgll): zeros of L'_{n-1} plus the endpoints,
    Newton iteration from Chebyshev initial guesses."""
    N = n - 1
    tol = 1e-15

    z = np.zeros(n)
    w = np.zeros(n)

    w0 = 2.0 / (N * n)
    z[0], z[N] = -1.0, 1.0
    w[0], w[N] = w0, w0

    # initial guess = Chebyshev nodes
    cheby = np.array([np.cos((2 * k + 1) * np.pi / (2 * n)) for k in range(n)])
    cheby = cheby[::-1]

    for i in range(1, N):
        x = cheby[i]
        dx = np.inf
        it = 0
        while np.abs(dx) > tol and it < 10:
            it += 1

            # compute p = L(x), pp = L'(x), ppp = L''(x) via the recurrence
            pk1, pk0 = 1.0, x  # p_{k}, p_{k-1}
            ppk1, ppk0 = 0.0, 1.0
            pppk1, pppk0 = 0.0, 0.0
            for k in range(1, n - 1):
                k1, k2 = k + 1, 2 * k + 1
                p = (k2 * x * pk0 - k * pk1) / k1
                pp = (k2 * (x * ppk0 + pk0) - k * ppk1) / k1
                ppp = (k2 * (x * pppk0 + 2 * ppk0) - k * pppk1) / k1

                pk1, ppk1, pppk1 = pk0, ppk0, pppk0
                pk0, ppk0, pppk0 = p, pp, ppp

            # newton
            dx = -pp / ppp
            x = x + dx

        z[i] = x
        w[i] = 2.0 / ((n - 1) * n * p**2)
    return z, w


def axis_weights(n, code):
    """Normalized (sum=1) per-axis averaging weights, matching fieldExtract:
    code 0 -> trapezoid/(n-1); code 1 -> GLL quadrature w/2."""
    if n == 1:
        return np.array([1.0])
    if code == 1:
        return 0.5 * zwgll(n)[1]
    w = np.ones(n)
    w[0] = w[-1] = 0.5
    return w / (n - 1)


def axis_nodes_weights(n, code, a, b):
    """Per-axis node coordinates on [a,b] + normalized weights, matching
    fieldExtract::setupAxes (code 0 uniform, 1 GLL mapped from [-1,1])."""
    if n == 1:
        return np.array([a]), np.array([1.0])
    if code == 1:
        z, w = zwgll(n)
        return a + 0.5 * (z + 1.0) * (b - a), 0.5 * w
    return np.linspace(a, b, n), axis_weights(n, 0)


def grid_avg(F, dims, axes, dist=(0, 0, 0)):
    """Weighted average of flat field F (dims=(nx,ny,nz)) over axes in 'xyz',
    using the per-axis weights implied by dist (0 uniform/trapezoid, 1 GLL).

    Mirrors fieldExtract::doAvg. Returns shape (nz', ny', nx') with averaged
    axes kept as 1.
    """
    nx, ny, nz = dims
    A = np.asarray(F, dtype=np.float64).reshape((nz, ny, nx))
    for ax in axes.lower():
        ia = {"x": 0, "y": 1, "z": 2}[ax]
        axis = 2 - ia  # numpy axis in the (nz, ny, nx) layout
        n = A.shape[axis]
        w = axis_weights(n, dist[ia])
        shape = [1, 1, 1]
        shape[axis] = n
        A = (A * w.reshape(shape)).sum(axis=axis, keepdims=True)
    return A


def trap_avg(F, dims, axes):
    """Trapezoidal average (uniform grids) — grid_avg with all-uniform dist."""
    return grid_avg(F, dims, axes, dist=(0, 0, 0))


def azimuthal_avg(F, dims):
    """Uniform-periodic average of a cyl tensor grid over the theta axis (axis 1
    in the (nr,nt,nz) dims = the 'y' slot). For a periodic [0,2pi) grid the
    quadrature weight is uniform 1/nt, i.e. a plain mean over theta -- matching
    turbPipe_cyl.udf's setQuadrature. Returns shape (nz, 1, nr)."""
    nr, nt, nz = dims
    A = np.asarray(F, dtype=np.float64).reshape((nz, nt, nr))
    return A.mean(axis=1, keepdims=True)


def check_avg(avg_fname, src_fname, axes, mode, tol=1e-5):
    """Compare an _avg .vts against the numpy weighted average of its source box
    .vts (weights from the source file's pointDist: trapezoid or GLL).

    Exercises the exact contract of fieldExtract::doAvg (same grid, same weights),
    so tol is tight -- limited only by float32 storage / summation order.
    """
    da = load(avg_fname)
    ds = load(src_fname)
    nx, ny, nz = ds["dims"]
    ok = True
    errs = []
    for key in ("vz", "temp"):
        ref = grid_avg(ds[key], ds["dims"], axes, ds["pointDist"])
        if mode == "gather-scatter":
            ref = np.broadcast_to(ref, (nz, ny, nx))
        got = np.asarray(da[key], dtype=np.float64).reshape(ref.shape)
        # normalize by the UNaveraged source-field scale: averages can be ~0
        # (odd symmetry), while the float32 storage noise is absolute
        e = np.abs(got - ref).max() / np.abs(ds[key]).max()
        errs.append(e)
        ok = ok and (e < tol)
    print(
        f"{avg_fname}: dims={da['dims']} avg={axes} mode={mode} "
        f"relerr(vz)={errs[0]:.2e} relerr(temp)={errs[1]:.2e} -> {'PASS' if ok else 'FAIL'}"
    )
    return ok
