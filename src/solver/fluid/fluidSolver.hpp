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

class fluidSolver_t : public solver_t
{
private:
  void solvePressure(double time, int stage);
  void solveVelocity(double time, int stage, int phase);

  void advectionSubcycling(int nEXT, double time, int phase); // TODO

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

  deviceMemory<dfloat> o_solution(std::string key = "") override
  {
    if (key.empty()) {
      return deviceMemory<dfloat>(o_U);
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
    if (key.empty()) {
      return deviceMemory<dfloat>(o_EXT);
    }
    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? deviceMemory<dfloat>(o_EXT.slice(idx * fieldOffset, fieldOffset))
                      : deviceMemory<dfloat>(o_NULL);
  };

  deviceMemory<dfloat> o_diffusionCoeff(std::string key = "") override
  {
    return deviceMemory<dfloat>(o_mue);
  };

  deviceMemory<dfloat> o_transportCoeff(std::string key = "") override
  {
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

  void makeAdvection(double time, int tstep, int phase);
  void makeExplicit(double time, int tstep, int phase);
  void makeForcing(int phase);

  // eq 81: u_bar = sum_k alpha_k*u_k (+ re-derives o_div from the new u_bar). Must run before
  // solvePressure, with phases' velocities/alpha as of the end of the previous iteration.
  void updateMixtureVelocity();

  // eq 99: combines the per-phase o_ADV_phases[k]/o_JwF_phases[k] (already built earlier in
  // initInnerStep, before alpha was current) with the now-current o_alpha_phases into the
  // mixture forcing solvePressure's RHS consumes. Must run after updateMixtureVelocity(),
  // still before solvePressure.
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

  occa::memory o_relUrst; // TODO no 2p

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
