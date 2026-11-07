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

  occa::memory o_ADV; // TODO phases

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

  void solve(double time, int stage) override
  {
    solvePressure(time, stage);
    for (int i=0;i<this->Nphase;i++) {
      solveVelocity(time, stage, i);
    }
  };

  void makeForcing();

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
