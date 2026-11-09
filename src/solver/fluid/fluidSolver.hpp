#if !defined(nrs_hydro_hpp_)
#define nrs_hydro_hpp_

#include "solver.hpp"
#include "geomSolver.hpp"

class fluidSolverCfg_t : public solverCfg_t
{
public:
  std::vector<std::string> velocityName;
  std::string pressureName;
  dlong fieldOffset;
  dlong cubatureOffset;
};

// per-(phase, velocity component) snapshot of an elliptic solve's convergence stats, taken
// right after that solve() call -- needed because ellipticSolver is a SHARED set reused
// across phases (see setupEllipticSolver): its live Niter()/residual() state gets
// overwritten by the next phase's solve before nrs_t::printStepInfo can read it.
struct velocityConvergence_t {
  int Niter = 0;
  dfloat initialGuessResidual = 0;
  dfloat finalResidual = 0;
};

class fluidSolver_t : public solver_t
{
private:
  void solvePressure(double time, int stage);
  void solveVelocity(double time, int stage, int phase);

  void advectionSubcycling(int nEXT, double time, int phase); // TODO

  // Sum_{s=0}^{nLevels-1} o_coeff[s] * o_field[level s] -- o_field is a multi-level buffer
  // (levels strided by fieldOffsetSum, i.e. mesh->dim stacked fieldOffset-sized components
  // per level); o_coeff is read back to host (nLevels is tiny, O(2-3), negligible cost) so
  // plain axpby can be used level-by-level over the whole fieldOffsetSum-sized block at once
  // (component-agnostic -- axpby has no notion of fieldOffset/stride, so this works
  // identically whether o_field represents 1 or mesh->dim components). Writes a single
  // level's worth (fieldOffsetSum) into o_out. Used by updateMixtureForcing() (Plan J) for
  // both the BDF-of-o_U and EXT-of-o_ADV_phases[k] combinations, which need the SAME
  // "weighted sum over lag levels" arithmetic sumMakef.okl already does internally but never
  // exposes on its own.
  void combineLagLevels(const occa::memory &o_coeff,
                         const occa::memory &o_field,
                         int nLevels,
                         occa::memory &o_out,
                         bool applyJacobianWeight = true);

  occa::memory o_zeroNormalMask;
  occa::memory o_filterRT;
  int Nsubsteps;

  std::vector<occa::memory> o_ADV_phases;
  std::vector<occa::memory> o_JwF_phases; // phase k's own eq-97 forcing (feeds solveVelocity)

  occa::memory o_U0; // mixture
  occa::memory o_P0;
  occa::memory o_relUrst0;
  occa::memory o_prop0;
  occa::memory o_EXT0;

  occa::memory o_coeffEXTP;

public:
  fluidSolver_t(const fluidSolverCfg_t &cfg, const std::unique_ptr<geomSolver_t> &geom);

  // key resolution shared by o_solution/o_explicitTerms/o_transportCoeff/o_diffusionCoeff:
  // "" or "mixture" -> the mixture field; velocityName[i] (e.g. "phase0", or "fluid
  // velocity" in the classic case) -> phase i's own field. Returns -1 for no phase match
  // (caller falls through to its own x/y/z/p handling). Note the differing "shape" by key
  // kind: a phase key returns the FULL vector field (like the mixture/empty-key case), not
  // a single-component slice the way "x"/"y"/"z" do.
  int phaseKeyIndex(const std::string &key) const
  {
    for (int i = 0; i < Nphase; i++) {
      if (lowerCase(key) == lowerCase(velocityName[i])) {
        return i;
      }
    }
    return -1;
  }

  deviceMemory<dfloat> o_solution(std::string key = "") override
  {
    if (key.empty() || lowerCase(key) == "mixture") {
      return deviceMemory<dfloat>(o_U);
    }

    if (const auto i = phaseKeyIndex(key); i >= 0) {
      return deviceMemory<dfloat>(o_U_phases[i]);
    }

    if (lowerCase(key) == "p" || lowerCase(key) == "pressure") {
      return deviceMemory<dfloat>(o_P);
    }

    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? deviceMemory<dfloat>(o_U.slice(idx * fieldOffset, fieldOffset))
                      : deviceMemory<dfloat>(o_NULL);
  };

  deviceMemory<dfloat> o_explicitTerms(std::string key = "") override
  {
    if (key.empty() || lowerCase(key) == "mixture") {
      return deviceMemory<dfloat>(o_EXT);
    }

    if (const auto i = phaseKeyIndex(key); i >= 0) {
      return deviceMemory<dfloat>(o_EXT_phases[i]);
    }

    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? deviceMemory<dfloat>(o_EXT.slice(idx * fieldOffset, fieldOffset))
                      : deviceMemory<dfloat>(o_NULL);
  };

  deviceMemory<dfloat> o_diffusionCoeff(std::string key = "") override
  {
    if (const auto i = phaseKeyIndex(key); i >= 0) {
      return deviceMemory<dfloat>(o_mue_phases[i]);
    }
    return deviceMemory<dfloat>(o_mue);
  };

  deviceMemory<dfloat> o_transportCoeff(std::string key = "") override
  {
    if (const auto i = phaseKeyIndex(key); i >= 0) {
      return deviceMemory<dfloat>(o_rho_phases[i]);
    }
    return deviceMemory<dfloat>(o_rho);
  }

  void setTimeIntegrationCoeffs(int tstep) override;

  void lagSolution() override;
  void extrapolateSolution() override;

  void saveSolutionState() override;
  void restoreSolutionState() override;

  void applyDirichlet(double time) override;
  void setupEllipticSolver() override;

  // mixture o_rho/o_mue from phase properties + volume fraction (eq 83/84 of
  // two_phase_flow_equations_AT-1.pdf); no-op when no phases are declared. Must be called
  // before solvePressure (hence before solve()) with up-to-date o_alpha_phases.
  void updateMixtureProperties();

  // Plan G (alpha transport): NOT called automatically anywhere -- the UDF drives these,
  // typically from preFluid, right after scalar->solve() updates the independent alpha
  // scalar(s) and before fluid->solve() (updateMixtureVelocity/Properties/Forcing all
  // consume o_alpha_phases, so this must run before those, same "solve alpha first" order).
  //
  // Copies o_alpha (e.g. a scalar's solution) into o_alpha_phases[phase]. Not yet clipped to
  // [0,1] -- no elementwise clamp primitive exists in linAlg currently; add one if sharp
  // alpha fronts overshoot (the alpha transport equation alone doesn't guarantee
  // boundedness under spectral-element advection).
  void setAlphaPhase(int phase, const occa::memory &o_alpha);

  // o_alpha_phases[phase] = 1 - sum of every OTHER o_alpha_phases[k] (general for any
  // Nphase: summing N independently-discretized transport equations does not exactly
  // preserve sum_k alpha_k = 1 under discretization error; deriving one phase by
  // subtraction makes it an algebraic identity instead). Caller picks which phase is
  // "derived" -- not auto-detected, for determinism.
  void deriveLastAlphaPhase(int phase);

  void makeAdvection(double time, int tstep, int phase);
  void makeExplicit(double time, int tstep, int phase);
  void makeForcing(int phase);

  // (v2, 81): u_bar = sum_k alpha_k*u_k -- writes ONLY o_U's CURRENT (level 0) entry, using
  // TODAY's alpha_k^{n+1} and o_U_phases[k]'s own level-0 (at this call site: last step's
  // converged interior values, but THIS step's freshly-applyDirichlet()-enforced boundary
  // values -- solveVelocity() for this step hasn't run yet). Older levels are left untouched:
  // lagSolution() already shifted them down at the top of this timestep, and each was itself
  // written by a PAST call to this same function using THAT step's own alpha -- so o_U's lag
  // history carries the TRUE historical mixture velocity at every level, with no separate
  // alpha-lag array needed. Plan J (fixed a bug where this used to rebuild ALL levels every
  // call with today's alpha, corrupting the historical ones): see wk_claude/plans/J_*.md.
  // MUST stay called before solvePressure() (same spot as before this fix) -- solvePressure's
  // boundary flux term reads o_U's level-0 boundary nodes directly, which need this step's
  // fresh Dirichlet values recombined in; moving this call to run after solveVelocity()
  // instead (tried, reverted) left solvePressure reading a stale boundary value and blew up
  // the very next step.
  void updateMixtureVelocity();

  // (v2, 108)'s forcing bracket: Sum_q a_q*ubar^{n-q}/dt - Sum_k alpha_k^{n+1}*[Sum_q
  // b_q*(u_k.grad(u_k))^{n-q}]. The BDF part comes directly from o_U's OWN (correct, see
  // updateMixtureVelocity) lag history -- NOT alpha-reweighted. Only the nonlinear
  // self-advection sum is alpha-weighted, phase by phase, using each phase's own
  // o_ADV_phases[k] lag history (already built in initInnerStep) BEFORE it was combined with
  // the BDF part into o_JwF_phases[k] -- so this does NOT reuse o_JwF_phases[k] wholesale
  // (Plan J fixed a bug where it used to: Sum_k alpha_k^{n+1}*o_JwF_phases[k], which
  // incorrectly applied today's alpha to the BDF part too). Must run before solvePressure.
  void updateMixtureForcing();

  void solve(double time, int stage) override
  {
    updateMixtureVelocity();
    updateMixtureForcing();
    solvePressure(time, stage);
    for (int i=0;i<this->Nphase;i++) {
      solveVelocity(time, stage, i);
    }
  };

  void updateZeroNormalMask()
  {
    if (platform->app->bc->hasUnalignedMixed(name)) {
      o_zeroNormalMask = mesh->createZeroNormalMask(fieldOffset, ellipticSolver[0]->o_EToB());
    }
  };

  mesh_t *mesh = nullptr;

  elliptic *ellipticSolverP = nullptr;

  const std::unique_ptr<geomSolver_t> &geom;

  dlong fieldOffset = -1;
  dlong cubatureOffset = -1;

  int Nphase = 1;

  // name of the shared/mixture-level velocity solve (e.g. "fluid velocity"), used for option
  // lookups and BC/solver setup that still operate on the single shared ellipticSolver set,
  // as opposed to velocityName[k] (per-phase). Set once in the constructor.
  std::string mixtureVelocityName;

  std::vector<std::string> velocityName; // TODO, phases
  std::vector<occa::memory> o_velocityName;

  std::string pressureName;
  occa::memory o_pressureName;

  std::function<occa::memory(double)> userImplicitLinearTerm = nullptr;
  std::function<occa::memory(double, int)> userAdvectionTerm = nullptr;

  // (v2, 105)'s RED term -- the per-phase diffusion-velocity viscous stress,
  // -div(sum_k alpha_k*div(nu_k*(grad(Ubar_k)+grad(Ubar_k)^T))), Ubar_k = u_k - u_bar (eq
  // 86). Not implemented anywhere by default (same documented gap as always); a case's UDF
  // may set this to inject it into solvePressure()'s RHS. Returns the RAW (positive-sign)
  // div(sum_k alpha_k*div(nu_k*grad(Ubar_k))) -- solvePressure() applies the PDF's own
  // leading minus sign itself (via pressureAddQtl's lambda), matching the convention Plan K
  // Part A already uses for the lagged-Q_bar term, so this hook's return value should NOT
  // be pre-negated by the UDF.
  std::function<occa::memory(double)> userRedTerm = nullptr;

  occa::memory o_U;   // mixture
  occa::memory o_Ue;  // mixture
  occa::memory o_div; // mixture

  occa::memory o_P;
  occa::memory o_Pe;

  std::vector<occa::memory> o_U_phases;
  std::vector<occa::memory> o_Ue_phases; // TODO

  // o_EXT_phases[0] always aliases the base-class o_EXT (external consumers -- nrs.cpp's
  // userSource/dt-heuristic -- keep working against fluid->o_EXT unchanged, always hitting
  // phase 0's slot); o_EXT_phases[1..] are real per-phase storage when phases are declared.
  std::vector<occa::memory> o_EXT_phases;

  dfloat rho0 = NAN;
  occa::memory o_rho; // mixture
  occa::memory o_mue; // mixture
  std::vector<occa::memory> o_rho_phases;
  std::vector<occa::memory> o_mue_phases;
  std::vector<occa::memory> o_alpha_phases; // vol frac

  // [phase][component] -- component count matches ellipticSolver.size() (1 for block solver,
  // 3 for separate x/y/z). Populated by solveVelocity(); read by nrs_t::printStepInfo.
  std::vector<std::vector<velocityConvergence_t>> velocityConvergence;

  occa::memory o_relUrst; // mixture-level (classic case; also used by any scalar NOT tagged
                           // with a phase= key, e.g. a shared temperature field)

  // phase k's own convecting velocity (eq 92/97: (u_k.grad)u_k, not (u_bar.grad)u_k) --
  // computed by computeUrstPhases(). Classic case: o_relUrst_phases[0] aliases o_relUrst
  // (same pattern as o_U_phases etc.).
  std::vector<occa::memory> o_relUrst_phases;
  void computeUrstPhases();

  void finalize() override
  {
    for (auto &entry : ellipticSolver) {
      delete entry;
      entry = nullptr;
    }
    delete ellipticSolverP;
    ellipticSolverP = nullptr;
  };
};

#endif
