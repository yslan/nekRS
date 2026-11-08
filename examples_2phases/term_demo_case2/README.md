# term_demo_case2

Two-phase (mixture-model) "term demo" example reproducing the **Second Test Case** of:

> B. W. Shaver, R. Rahman, R. C. Martineau, "Verification of the Two-Fluid Model in the
> Spectral Element Code Nek-2P for Adiabatic Flow Conditions," *Transactions of the American
> Nuclear Society*, Vol. 116, No. 1, pp. 1251-1254 (2017).

The 4-page excerpt used while building this case is kept locally at
`wk_claude/trans_v116_n1_pp1251-1254.pdf` (that directory is gitignored/local-only and is not
part of this example's own tracked files -- obtain the paper from ANS/OSTI if you don't have
a local copy).

## What this demos

Both phases enter at `u=1` (co-moving), with a genuinely nonzero dispersed volume fraction
at the inlet (`alpha_d = 0.2083`, Table II). Unlike `term_demo_case1` (where `alpha_d==0` is
hardcoded, trivially decoupling the phases), here `alpha_d` is actually transported by a
scalar (`[SCALAR alphad]`, Plan G mechanism) and fed back into the fluid solver's
`o_alpha_phases` every step via `nrs->preFluid`. The mixture should accelerate/decelerate
from the inlet state toward the paper's independently-computed fully-developed state
(Table III) under gravity + drag.

**Template substitution (read before trusting any C_D-dependent number here):** the paper's
text says this case "includes the effect of... a variable drag coefficient" but the 4 pages
available do not give the actual `C_D(Re_d)` correlation used. `case2.udf`'s
`dragGravityKernel` substitutes the standard **Schiller-Naumann** correlation,
`C_D = (24/Re_d)(1 + 0.15 Re_d^0.687)`, clipped to Stokes drag at very small `Re_d` and
capped at `C_D=400`. `Re_d` is formed against `[FLUID VELOCITY] viscosity = 1e-4`, which is
itself only a numerical regularizer (the paper's own momentum equations have no diffusion
term). This is a demonstration of how a variable-C_D drag law gets wired into `userSource`,
**not** a reproduction of the paper's own (unpublished-in-these-4-pages) curve.

## Files

- `case2.box` -- genbox input, 40x1x1 elements (2x case1's streamwise resolution, since alpha
  genuinely evolves from inlet to fully-developed here and needs resolving).
- `case2.usr` -- classic Nek5000 user file; `usrdat2()` assigns `boundaryID` from `cbc`.
  **Important lesson learned here** (see the long comment at the top of `case2.box`): do NOT
  code the inlet/outlet faces `"E"` when `nelx > 1` -- genbox writes that same code for every
  genuine internal element-element connection along the chain, so `usrdat2` can never tell
  the true boundary face apart from an ordinary interior one, and `boundaryID` for that face
  never gets set anywhere (`"boundary ID needs to be one-based index but min(ID) is 2!"`).
  Use distinct, never-reused-for-internal-faces codes instead -- `"v"` for inlet, `"O"` for
  outlet (the standard Nek5000 convention, same as `examples/gabls1`/`examples/lowMach`).
- `case2.par`, `case2.udf` -- the nekRS case.
  - `[FLUID PHASE phase0]`/`[phase1]` each carry an **explicit** `boundaryTypeMap` that must
    match `[FLUID VELOCITY]`'s exactly, word for word -- `fluidSolver_t::setupEllipticSolver()`
    hard-refuses on a mismatch (2-phase mixed-BC-per-phase isn't supported), and relying on
    this par file's phase-inherits-from-mixture auto-merge did not reliably propagate a
    3-entry map in testing here.
  - `[CASEDATA] ioSample` controls the fieldExtract sampling cadence (steps between `.vts`
    dumps) -- read via `platform->par->extract("casedata", "iosample", ioSample)` in
    `UDF_Setup()` and used as the `tstep % ioSample == 0` guard in `UDF_ExecuteStep()`. Keep
    `numSteps / ioSample` small (<=20ish) -- there is no reason to write hundreds of `.vts`
    dumps of a 1D line sample.
  - `numSteps`/`dt`: case1's `dt` (sized for 20 streamwise elements) is numerically unstable
    here at 40 elements -- it doesn't just report a large CFL number, the solution itself
    diverges (`resNorm`/`divUErr` growing without bound within ~50 steps). `dt=5e-4` was
    verified stable (flat `CFL=0.312`) over a few hundred steps, but became unstable again
    once the mixture-vs-phase convection fix (below) made self-advection genuinely active;
    retuned to `dt=2.5e-4`/`numSteps=32000` (same total simulated time `T=8`), confirmed
    stable (`CFL~0.156` throughout) for the production run.
- `case2_eq90_attempt.udf`/`.par` -- **backup, not built** (different basename, `nrsmpi`
  never picks it up). A full attempt at the eq 90 dilation term described below, kept for
  reference. See "alpha_d: known limitation" below.
- `case2_eq90_instability.png` -- diagnostic plot of that attempt's blowup (see below).
- `fieldExtract.hpp`, `fe_read.py` -- vendored spatial-line sampler (writes `.vts`) and its
  Python reader (copied from `term_demo_case1`, identical).
- `read_case2.py` -- all physics validation/plotting (the UDF only samples).

## Running

```
source /Disk_K/src/nekrs_yslan_2phase/note
export PATH=$NEKRS_HOME/bin:$PATH
cd examples_2phases/term_demo_case2
NEKRS_JITC_NTHREADS=8 nrsmpi case2 1 --build-only   # JIT-compile (parallel kernel build)
nrsmpi case2 4 --backend serial                     # actual run, CPU backend, 4 ranks
```

## Getting the PNG from the .py script

After a run completes, `case2.box`'s fieldExtract sampler will have written `ioSample`-paced
`.vts` dumps (`case2NNNNN.vts`) along the centerline. Regenerate the comparison plot
(`case2_fig_comparison.png`) with:

```
cd examples_2phases/term_demo_case2
python3 read_case2.py
```

This loads the **latest** `.vts` dump (globs `case2*.vts` and sorts, so it always picks up
the most recent run without hardcoding a filename), prints a numeric comparison against
Table II (inlet) and Table III (fully-developed/outlet), and writes
`case2_fig_comparison.png` (two panels: velocities, and `alpha_d`; both titled with the
simulation time `t` the dump was taken at).

## Current status / open issues (IMPORTANT, read before trusting the plot)

### Velocities: two core bugs fixed, overshoot vs. Table III remains

Two real core bugs (not specific to this example -- both in `src/`, fixed this session):

1. `fluidSolver_t::solveVelocity()` was passing the bare MIXTURE-combined `o_JwF` (eq 99's
   alpha-weighted sum, meant only for the pressure equation) into every phase's own velocity
   RHS, instead of that phase's own `o_JwF_phases[phase]` (eq 97). Whenever a phase's
   `alpha_k` was small at some point, the mixture sum's weighting zeroed that phase's own
   forcing out of its own equation.
2. Every phase's own momentum advection, and any phase-tagged scalar's own transport (e.g.
   `alphad` here), convected with the shared MIXTURE velocity instead of its own phase
   velocity (eq 92/97 need `(u_k.grad)u_k`, not `(u_bar.grad)u_k`). Fixed via
   `fluidSolver_t::computeUrstPhases()` and `scalar_t::o_relUrst_byScalar`, wired from each
   scalar's `phase=` tag.

```
                     u_c       u_d   alpha_d
Before both fixes: 0.9999    1.0006   0.2083   (frozen at inlet)
After fix #1 only: 0.7509    1.9467   0.2083   (moved, but alphad still advected by u_bar)
After BOTH fixes:  0.7630    1.9009   0.2083   (phase1 now genuinely self-advects too)
Table III (target):0.9318    1.3860   0.1504
```

`u_c`/`u_d` reach a converged steady state well before `t=8` and **overshoot** Table III in
both directions (u_c too low, u_d too high). Not yet root-caused whether this is the
Schiller-Naumann template substitution (see above) or something else in the eq 90 gap below
-- flagged as a follow-up, not urgent.

### alpha_d: known limitation (frozen at inlet value)

`alphad`'s inlet Dirichlet value (0.2083, Table II) is set exactly equal to its own uniform
IC everywhere. `scalarSolver_t` only implements the standard **non-conservative**
advection-diffusion form, `d(alpha)/dt + u.grad(alpha) = D*lap(alpha)` -- a spatially
uniform field with a matching boundary value is an *exact* fixed point of that equation for
**any** `diffusionCoeff` and **any** advecting velocity (both `u.grad(alpha)` and
`D*lap(alpha)` vanish identically when `grad(alpha)=0`). This holds regardless of the two
fixes above, which is why `alpha_d` stays exactly `0.2083` at every point, every dump, even
now that `u_c`/`u_d` genuinely evolve.

The paper's own alpha equation (eq 14/15) and the more general AT2 formulation (eq 90,
`d(alpha_k)/dt + div(alpha_k*u_k) = Gamma_k/rho_k`) both have an extra dilation term,
`alpha_k*div(u_k)`, that a plain advection-diffusion equation cannot provide -- this is the
actual missing piece, confirmed by direct derivation (not a guess): diffusion-coefficient
tuning cannot fix a frozen-uniform-field fixed point at any value of `diffusionCoeff`.

**A real attempt to add this term was made and reverted** (backed up in
`case2_eq90_attempt.udf`/`.par`, not built by default). Tried, in order: an explicit
`userSource` term; the same term moved to the LHS as an implicit reaction coefficient via
`scalar_t::userImplicitLinearTerm` (since `alpha_k*div(u_k)` is linear in `alpha_k`); AVM
modal-decay regularization (`scalarSolver.cpp`'s `applyAVM()`) on top of the implicit form;
a `[0,1]` clamp on `alpha_d` (motivated by `fluidSolver.cpp`'s own "no clipping yet" comment
on `setAlphaPhase`); and under-relaxation (`relax=0.1`) of the push into
`fluid->o_alpha_phases[1]`. **Every variant blew up**, at a time that moved with the
mitigation but was never eliminated by it:

| Variant                                   | Blowup time |
|--------------------------------------------|-------------|
| Explicit source                             | t ~ 0.223   |
| Implicit reaction term alone                | t ~ 0.227   |
| + `[0,1]` clamp                             | t ~ 0.235   |
| + AVM regularization                        | t ~ 0.327   |
| + under-relaxation (0.1) instead of clamp   | t ~ 0.32    |

A direct diagnostic (`div(u_d)` and `alpha_d` min/max vs. time, every 5 steps) showed
**exponential, not sudden, growth** from early in the run (`div(u_d)` already O(7) by
t=0.16, O(100) by t=0.21, O(1e4) by t=0.32) -- see `case2_eq90_instability.png`:

![eq 90 dilation-term instability](case2_eq90_instability.png)

Every mitigation only delayed this, never cured it -- the signature of a genuinely unstable
feedback loop (`alpha_d` -> mixture `rho_bar`/`nu_bar` (eq 83/84) -> shared pressure/momentum
solve -> `u_d` -> `div(u_d)` -> back into `alpha_d`), not a fixable discretization choice.
Likely root cause: this architecture enforces only the **mixture's** divergence-free
condition (one shared pressure, eq 91's `Q_bar`); nothing constrains each phase's own
`div(u_k)` individually, and until this attempt `div(u_k)` had never been used for anything
load-bearing in the codebase. More fundamentally, the AT2 equations as excerpted use a
single shared pressure between phases (eq 92/97, confirmed) with no interfacial/differential
pressure or virtual-mass closure -- the classical "equal-pressure, drag-only" two-fluid
model is a well-known case of an **ill-posed** initial value problem (complex characteristic
roots; Ramshaw & Trapp 1978, Stewart & Wendroff 1984) unless such a closure is added. That
would explain why every numerical mitigation tried here only delayed, never fixed, the
blowup -- real two-fluid codes (presumably including the paper's own Nek-2P) add one of
these closures specifically to avoid this.

**Reverted to the stable, frozen-alpha state** (this directory's live `case2.udf`/`.par`)
rather than ship a case that crashes partway through. Properly fixing this would mean
implementing a genuine interfacial-pressure or virtual-mass closure, not another numerical
patch -- out of scope for this pass.

![term_demo_case2: u_c, u_d, alpha_d vs. Table II/III](case2_fig_comparison.png)
