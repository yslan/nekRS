#ifndef CDS_H
#define CDS_H

#include "platform.hpp"
#include "solver.hpp"
#include "geomSolver.hpp"
#include "cvode.hpp"

struct scalarConfig_t : public solverCfg_t {
public:
  int Nscalar;
  std::vector<mesh_t *> mesh;
  mesh_t *meshV;
  dlong vFieldOffset;
  dlong vCubatureOffset;
  dlong fieldOffset;
  occa::memory o_U;
  occa::memory o_Ue;
  occa::memory o_relUrst;

  // Per-scalar override of the convecting velocity above (eq 90: a phase-tagged scalar like
  // alpha_k should advect with THAT phase's own u_k, not the shared mixture velocity). Index
  // by scalar number; an uninitialized/missing entry falls back to o_relUrst above (the
  // existing default -- correct for an ordinary, non-phase-tagged scalar like temperature).
  // Populated in nrs.cpp from each scalar's "phase=" par key, using fluid->o_relUrst_phases.
  std::vector<occa::memory> o_relUrst_byScalar;

  bool dpdt = false;
  dfloat *dp0thdt = nullptr;
  dfloat *alpha0Ref = nullptr;
};

class scalar_t : public solver_t
{
public:
  static constexpr double targetTimeBenchmark{0.2};

  scalar_t(scalarConfig_t &cfg, const std::unique_ptr<geomSolver_t> &geom);
  void makeExplicit(int is, double time, int tstep);
  void makeAdvection(int is, double time, int tstep);
  void makeForcing();

  void solve(double time, int stage) override;

  void saveSolutionState() override;
  void restoreSolutionState() override;
  void lagSolution() override;

  void setTimeIntegrationCoeffs(int tstep) override {};

  void extrapolateSolution() override;

  void applyDirichlet(double time) override;
  void setupEllipticSolver() override;

  void finalize() override;

  void applyAVM();

  std::function<occa::memory(double, int)> userImplicitLinearTerm = nullptr;

  void computeUrst();

  dlong fieldOffset() const
  {
    return _fieldOffset;
  };

  mesh_t* mesh(std::string key)
  {
    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? _mesh[idx] : nullptr;
  };

  mesh_t* mesh(int idx)
  {
    return _mesh.at(idx);
  };

  deviceMemory<dfloat> o_solution(std::string key = "") override
  {
    if (key.empty()) return deviceMemory<dfloat>(o_S);
    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? deviceMemory<dfloat>(o_S.slice(fieldOffsetScan[idx], _fieldOffset)) : deviceMemory<dfloat>(o_NULL);
  };

  deviceMemory<dfloat> o_explicitTerms(std::string key = "") override
  {
    if (key.empty()) return deviceMemory<dfloat>(o_EXT);
    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? deviceMemory<dfloat>(o_EXT.slice(fieldOffsetScan[idx], _fieldOffset)) : deviceMemory<dfloat>(o_NULL);
  };

  deviceMemory<dfloat> o_diffusionCoeff(std::string key = "") override
  {
    if (key.empty()) return deviceMemory<dfloat>(o_prop.slice(0, fieldOffsetSum));
    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? deviceMemory<dfloat>(o_prop.slice(fieldOffsetScan[idx], _fieldOffset)) : deviceMemory<dfloat>(o_NULL);
  };

  deviceMemory<dfloat> o_transportCoeff(std::string key = "") override
  {
    if (key.empty()) return deviceMemory<dfloat>(o_prop.slice(fieldOffsetSum, fieldOffsetSum));
    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? deviceMemory<dfloat>(o_prop.slice(fieldOffsetSum + fieldOffsetScan[idx], _fieldOffset)) : deviceMemory<dfloat>(o_NULL);
  }

  mesh_t *meshV;

  occa::memory o_fieldOffsetScan;

  dlong vFieldOffset;
  dlong vCubatureOffset;

  std::vector<dlong> fieldOffsetScan; /* exclusive */

  cvode_t *cvode = nullptr;

  bool anyCvodeSolver = false;
  bool anyEllipticSolver = false;

  int NSfields = 0;

  std::vector<QQt *> qqt;

  std::vector<int> compute;
  std::vector<int> cvodeSolve;
  occa::memory o_compute;
  occa::memory o_cvodeSolve;

  occa::memory o_applyFilterRT;
  occa::memory o_filterS;
  occa::memory o_filterRT;

  int Nsubsteps = 0;

  bool dpdt = false;
  dfloat *dp0thdt = nullptr;
  dfloat *alpha0Ref = nullptr;

  dlong EToBOffset = -1;

  occa::memory o_U;
  occa::memory o_Ue;
  occa::memory o_relUrst;

  // see scalarConfig_t::o_relUrst_byScalar -- resolved once at construction (falls back to
  // o_relUrst above for any scalar without its own entry), read by makeAdvection/
  // advectionSubcycling instead of the bare o_relUrst.
  std::vector<occa::memory> o_relUrst_byScalar;

  occa::memory o_S;
  occa::memory o_Se;

  occa::memory o_rho;
  occa::memory o_diff;

  const std::unique_ptr<geomSolver_t> &geom;

  std::vector<std::string> name;

private:
  void advectionSubcycling(int nEXT, double time, int scalarIdx);
  std::vector<mesh_t *> _mesh;

  std::vector<occa::memory> o_name;

  dlong _fieldOffset = -1; // all scalar fields share the same offset 

  occa::memory o_ADV;

  occa::memory o_S0;
  occa::memory o_EXT0;
  occa::memory o_ADV0;
  occa::memory o_prop0;

};

#endif
