#include "nrs.hpp"
#include "gjp.hpp"
#include "lowPassFilter.hpp"
#include "advectionSubCycling.hpp"
#include "registerKernels.hpp"
#include "nekInterfaceAdapter.hpp"

fluidSolver_t::fluidSolver_t(const fluidSolverCfg_t &cfg, const std::unique_ptr<geomSolver_t> &_geom)
    : geom(_geom)
{
  name = cfg.name;

  mesh = cfg.mesh;
  fieldOffset = cfg.fieldOffset;
  fieldOffsetSum = mesh->dim * fieldOffset;
  cubatureOffset = cfg.cubatureOffset;

  const char nullChar[] = {'\0'};

  // mixture-level velocity: option lookups below (SOLVER, BC, EXTRAPOLATION, ...) for the
  // mixture fields (o_U, o_Ue, o_EToB, ...) stay keyed on this name, independent of whether
  // phases are declared.
  mixtureVelocityName = name + " velocity";

  int nPhasesDeclared = 0;
  platform->options.getArgs("NUMBER OF PHASES", nPhasesDeclared);
  Nphase = std::max(nPhasesDeclared, 1);

  velocityName = (nPhasesDeclared > 0) ? [&]() {
    std::vector<std::string> names;
    for (int i = 0; i < Nphase; i++) {
      names.push_back("phase" + phaseDigitStr(i));
    }
    return names;
  }() : cfg.velocityName;

  if (platform->comm.mpiRank() == 0) {
    std::cout << "================ "
              << "SETUP " + upperCase(name) << " ================" << std::endl;
    for (int i = 0; i < nPhasesDeclared; i++) {
      std::cout << "================ "
                << "SETUP " + upperCase(velocityName[i]) << " ================" << std::endl;
      std::cout << "  inherit settings from " << mixtureVelocityName << std::endl;
    }
  }

  o_velocityName.resize(Nphase);
  for (int i = 0; i < Nphase; i++) {
    o_velocityName[i] = platform->device.malloc<char>(velocityName[i].size() + 1);
    o_velocityName[i].copyFrom(velocityName[i].data());
    o_velocityName[i].copyFrom(nullChar, 1, velocityName[i].size());
  }

  pressureName = cfg.pressureName;
  o_pressureName = platform->device.malloc<char>(pressureName.size() + 1);
  o_pressureName.copyFrom(pressureName.data());
  o_pressureName.copyFrom(nullChar, 1, pressureName.size());

  g0 = cfg.g0;
  dt = cfg.dt;
  o_coeffEXT = cfg.o_coeffEXT;
  o_coeffBDF = cfg.o_coeffBDF;

  nameToIndex["x"] = 0;
  nameToIndex["y"] = 1;
  nameToIndex["z"] = 2;

  Nsubsteps = 0;
  platform->options.getArgs("SUBCYCLING STEPS", Nsubsteps);

  o_U = platform->device.malloc<dfloat>(fieldOffsetSum * std::max(o_coeffBDF.size(), o_coeffEXT.size()));
  // Plan J: updateMixtureVelocity() now only ever writes o_U's level 0 (not a full rebuild
  // every call, see its own comment), so levels 1..n-1 are never touched until enough real
  // steps have passed to fill them via lagSolution()'s shift. Zero them up front -- same
  // "avoid solving from garbage" reasoning as o_U_phases[i]'s fill below -- since their BDF/
  // EXT coefficients are correctly zeroed during the startup order-ramp, but 0*garbage can
  // still be NaN if the garbage happens to contain a NaN bit pattern.
  platform->linAlg->fill(o_U.size(), 0.0, o_U);

  if (!platform->options.compareArgs(upperCase(mixtureVelocityName) + " EXTRAPOLATION", "FALSE")) {
    o_Ue = platform->device.malloc<dfloat>(fieldOffsetSum);
  } else {
    o_Ue = o_U.slice(0, fieldOffsetSum);
  }

  o_U_phases.resize(Nphase);
  o_Ue_phases.resize(Nphase);
  if (nPhasesDeclared > 0) {
    for (int i = 0; i < Nphase; i++) {
      o_U_phases[i] =
          platform->device.malloc<dfloat>(fieldOffsetSum * std::max(o_coeffBDF.size(), o_coeffEXT.size()));
      // zero default: nothing extrapolates into o_Ue_phases yet (TODO), and a case's UDF
      // only sets the mixture o_U IC today, not o_U_phases -- avoid solving from garbage.
      platform->linAlg->fill(o_U_phases[i].size(), 0.0, o_U_phases[i]);

      if (!platform->options.compareArgs(upperCase(velocityName[i]) + " EXTRAPOLATION", "FALSE")) {
        o_Ue_phases[i] = platform->device.malloc<dfloat>(fieldOffsetSum);
        platform->linAlg->fill(o_Ue_phases[i].size(), 0.0, o_Ue_phases[i]);
      } else {
        o_Ue_phases[i] = o_U_phases[i].slice(0, fieldOffsetSum);
      }
    }
  } else {
    // classic case: Nphase==1 and solveVelocity/applyDirichlet always write through
    // o_U_phases[0]/o_Ue_phases[0] -- alias them to the real o_U/o_Ue (handle assignment,
    // same underlying buffer) rather than allocating separate storage the rest of the
    // solver (checkpoint, solvePressure's o_Ue-based terms, ...) would never see updated.
    o_U_phases = {o_U};
    o_Ue_phases = {o_Ue};
  }

  if (platform->options.compareArgs(upperCase(pressureName) + " RHO SPLITTING", "TRUE")) {
    int nEXT = 2;
    const auto key = upperCase(pressureName) + " EXT ORDER";
    if (platform->options.getArgs(key).empty()) {
      platform->options.setArgs(key, std::to_string(nEXT));
    }
    platform->options.getArgs(key, nEXT);
    o_Pe = platform->device.malloc<dfloat>(fieldOffset);
    o_coeffEXTP = platform->device.malloc<dfloat>(std::min(nEXT, static_cast<int>(o_coeffBDF.size())));
  }
  o_P = platform->device.malloc<dfloat>(fieldOffset * std::max(static_cast<int>(o_coeffEXTP.size()), 1));

  o_div = platform->device.malloc<dfloat>(fieldOffset);

  o_EXT = platform->device.malloc<dfloat>(o_coeffEXT.size() * fieldOffsetSum);

  // bare o_JwF is now exclusively "the combined mixture forcing for the pressure RHS" (eq 99),
  // written by updateMixtureForcing() from the per-phase pieces below; solvePressure's
  // reference to it is otherwise unchanged. o_ADV is fully per-phase (no external consumers).
  // o_EXT stays per-phase too but phase 0 aliases the bare o_EXT above so userSource/the
  // dt-heuristic in nrs.cpp keep working against fluid->o_EXT unchanged.
  o_JwF = platform->device.malloc<dfloat>(fieldOffsetSum);

  o_ADV_phases.resize(Nphase);
  o_JwF_phases.resize(Nphase);
  o_EXT_phases.resize(Nphase);
  o_EXT_phases[0] = o_EXT;
  for (int i = 0; i < Nphase; i++) {
    o_ADV_phases[i] = platform->device.malloc<dfloat>(o_coeffEXT.size() * fieldOffsetSum);
    o_JwF_phases[i] = platform->device.malloc<dfloat>(fieldOffsetSum);
    if (i > 0) {
      o_EXT_phases[i] = platform->device.malloc<dfloat>(o_coeffEXT.size() * fieldOffsetSum);
    }
  }

  {
    const dlong Nstates = Nsubsteps ? std::max(o_coeffBDF.size(), o_coeffEXT.size()) : 1;
    o_relUrst = platform->device.malloc<dfloat>(Nstates * mesh->dim * cubatureOffset);

    o_relUrst_phases.resize(Nphase);
    if (nPhasesDeclared > 0) {
      for (int i = 0; i < Nphase; i++) {
        o_relUrst_phases[i] = platform->device.malloc<dfloat>(Nstates * mesh->dim * cubatureOffset);
      }
    } else {
      o_relUrst_phases = {o_relUrst};
    }
  }

  o_prop = [&]() {
    dfloat mue = 1;
    dfloat rho = 1;
    platform->options.getArgs(upperCase(name) + " VISCOSITY", mue);
    platform->options.getArgs(upperCase(name) + " DENSITY", rho);
    auto o_u = platform->device.malloc<dfloat>(2 * fieldOffset);
    auto o_u0 = o_u + 0 * fieldOffset;
    auto o_u1 = o_u + 1 * fieldOffset;
    platform->linAlg->fill(mesh->Nlocal, mue, o_u0);
    platform->linAlg->fill(mesh->Nlocal, rho, o_u1);
    return o_u;
  }();
  o_mue = o_prop.slice(0 * fieldOffset, mesh->Nlocal);
  o_rho = o_prop.slice(1 * fieldOffset, mesh->Nlocal);

  if (nPhasesDeclared > 0) {
    o_rho_phases.resize(Nphase);
    o_mue_phases.resize(Nphase);
    o_alpha_phases.resize(Nphase);
    for (int i = 0; i < Nphase; i++) {
      auto o_propPhase = [&]() {
        dfloat mue = 1;
        dfloat rho = 1;
        platform->options.getArgs(upperCase(velocityName[i]) + " VISCOSITY", mue);
        platform->options.getArgs(upperCase(velocityName[i]) + " DENSITY", rho);
        auto o_u = platform->device.malloc<dfloat>(2 * fieldOffset);
        auto o_u0 = o_u + 0 * fieldOffset;
        auto o_u1 = o_u + 1 * fieldOffset;
        platform->linAlg->fill(mesh->Nlocal, mue, o_u0);
        platform->linAlg->fill(mesh->Nlocal, rho, o_u1);
        return o_u;
      }();
      o_mue_phases[i] = o_propPhase.slice(0 * fieldOffset, mesh->Nlocal);
      o_rho_phases[i] = o_propPhase.slice(1 * fieldOffset, mesh->Nlocal);

      o_alpha_phases[i] = platform->device.malloc<dfloat>(fieldOffset);
    }
  } else {
    // classic case: Nphase==1 and solveVelocity(phase=0) always indexes o_rho_phases[0]/
    // o_mue_phases[0] -- alias them to the mixture fields so that indexing stays valid
    // without solveVelocity needing to special-case "no phases declared".
    o_rho_phases = {o_rho};
    o_mue_phases = {o_mue};
  }

  if (!platform->options.compareArgs(upperCase(mixtureVelocityName) + " SOLVER", "NONE")) {
    platform->app->bc->printBcTypeMapping(mixtureVelocityName);

    if (platform->options.compareArgs(upperCase(mixtureVelocityName) + " REGULARIZATION METHOD", "HPFRT")) {
      int nModes = -1;
      platform->options.getArgs(upperCase(mixtureVelocityName) + " HPFRT MODES", nModes);
      o_filterRT = lowPassFilterSetup(mesh, nModes);
    }

    o_EToB = [&]() {
      auto u =
          mesh->createEToB([&](int bID) -> int { return platform->app->bc->typeId(bID, mixtureVelocityName); });
      auto o_u = platform->device.malloc<int>(u.size());
      o_u.copyFrom(u.data());
      return o_u;
    }();

    auto verifyBC = [&]() {
      nekrsCheck(mesh->Nbid != platform->app->bc->size(mixtureVelocityName),
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "Size of %s boundaryTypeMap (%d) does not match number of boundary IDs in mesh (%d)!\n",
                 mixtureVelocityName.c_str(),
                 platform->app->bc->size(mixtureVelocityName),
                 mesh->Nbid);

      platform->app->bc->checkAlignment(mesh);
    };

    verifyBC();
  }
}

// Plan G: not called anywhere in this codebase automatically -- the UDF drives this (e.g.
// from preFluid, right after scalar->solve() has updated the independent alpha scalar(s),
// before fluid->solve() consumes o_alpha_phases). No [0,1] clipping yet -- no elementwise
// clamp primitive exists in linAlg currently; add one if sharp alpha fronts overshoot.
void fluidSolver_t::setAlphaPhase(int phase, const occa::memory &o_alpha)
{
  o_alpha_phases[phase].copyFrom(o_alpha, mesh->Nlocal);
}

// o_alpha_phases[phase] = 1 - sum_{k != phase} o_alpha_phases[k]. General for any Nphase:
// summing N independently-discretized alpha transport equations does not exactly preserve
// sum_k alpha_k = 1 under discretization error; deriving one phase by subtraction makes it
// an algebraic identity instead of something that has to emerge numerically.
void fluidSolver_t::deriveLastAlphaPhase(int phase)
{
  platform->linAlg->fill(mesh->Nlocal, 1.0, o_alpha_phases[phase]);
  for (int k = 0; k < Nphase; k++) {
    if (k == phase) {
      continue;
    }
    platform->linAlg->axpby(mesh->Nlocal, -1.0, o_alpha_phases[k], 1.0, o_alpha_phases[phase]);
  }
}

void fluidSolver_t::updateMixtureProperties()
{
  if (o_alpha_phases.empty()) {
    return; // classic (no phases declared) case: o_rho_phases[0]/o_mue_phases[0] just alias
            // the mixture fields (see constructor), nothing to recompute from.
  }

  // rho_bar = (sum_k alpha_k / rho_k)^-1   (eq 83)
  auto o_invRhoSum = platform->deviceMemoryPool.reserve<dfloat>(fieldOffset);
  platform->linAlg->fill(mesh->Nlocal, 0.0, o_invRhoSum);
  for (int k = 0; k < Nphase; k++) {
    auto o_term = platform->deviceMemoryPool.reserve<dfloat>(fieldOffset);
    platform->linAlg->axdyz(mesh->Nlocal, 1.0, o_alpha_phases[k], o_rho_phases[k], o_term);
    platform->linAlg->axpby(mesh->Nlocal, 1.0, o_term, 1.0, o_invRhoSum);
  }
  platform->linAlg->ady(mesh->Nlocal, 1.0, o_invRhoSum); // in-place reciprocal -> rho_bar
  o_rho.copyFrom(o_invRhoSum, mesh->Nlocal);

  // nu_bar = sum_k alpha_k * (mue_k / rho_k)   (eq 84)
  auto o_nuSum = platform->deviceMemoryPool.reserve<dfloat>(fieldOffset);
  platform->linAlg->fill(mesh->Nlocal, 0.0, o_nuSum);
  for (int k = 0; k < Nphase; k++) {
    auto o_nuK = platform->deviceMemoryPool.reserve<dfloat>(fieldOffset);
    platform->linAlg->axdyz(mesh->Nlocal, 1.0, o_mue_phases[k], o_rho_phases[k], o_nuK); // nu_k

    auto o_term = platform->deviceMemoryPool.reserve<dfloat>(fieldOffset);
    platform->linAlg->axmyz(mesh->Nlocal, 1.0, o_alpha_phases[k], o_nuK, o_term); // alpha_k*nu_k
    platform->linAlg->axpby(mesh->Nlocal, 1.0, o_term, 1.0, o_nuSum);
  }

  // mue_bar = rho_bar * nu_bar
  platform->linAlg->axmyz(mesh->Nlocal, 1.0, o_rho, o_nuSum, o_mue);
}

// Plan J helper: Sum_{s=0}^{nLevels-1} o_coeff[s]*[JW*]o_field[level s] -> o_out (one level's
// worth, fieldOffsetSum long). The JW (mesh->o_Jw) factor matters for a Jacobian-WEIGHTED
// target like "o_JwF" (matching sumMakef.okl's own "bdfx += coeffBDF[s]*JW*U[...]"), but NOT
// for a plain (unweighted) velocity combination whose divergence will be taken afterward and
// fed through a kernel (pressureAddQtl) that applies its own mass-matrix weighting --
// applyJacobianWeight selects which (Plan K: the new Q_bar BDF-lag term needs false; Plan
// J's o_JwF/ADV/"+g" combinations need true, unchanged). Missing this weighting (when it IS
// needed) was a real bug caught by case1 blowing up during Plan J (bit-identical-to-old-code
// arithmetic only works if the weighting matches too). Not handling the p_MovingMesh
// per-level-JW variant (not needed for any current 2-phase case; sumMakef itself only varies
// JW per-level under that define). o_coeff is read back to host (nLevels is O(2-3),
// negligible) so plain axpby can combine whole fieldOffsetSum-sized blocks at once -- axpby
// has no notion of fieldOffset/component stride, so this is correct whether o_field
// represents 1 or mesh->dim stacked components.
void fluidSolver_t::combineLagLevels(const occa::memory &o_coeff,
                                      const occa::memory &o_field,
                                      int nLevels,
                                      occa::memory &o_out,
                                      bool applyJacobianWeight)
{
  std::vector<dfloat> coeff(nLevels);
  o_coeff.copyTo(coeff.data(), nLevels);

  platform->linAlg->fill(fieldOffsetSum, 0.0, o_out);
  for (int s = 0; s < nLevels; s++) {
    auto o_weighted = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);
    o_weighted.copyFrom(o_field.slice(s * fieldOffsetSum, fieldOffsetSum), fieldOffsetSum);
    if (applyJacobianWeight) {
      platform->linAlg->axmyVector(mesh->Nlocal, fieldOffset, 0, 1.0, mesh->o_Jw, o_weighted); // *= JW
    }
    platform->linAlg->axpby(fieldOffsetSum, coeff[s], o_weighted, 1.0, o_out);
  }
}

// (v2, 81): u_bar = sum_k alpha_k*u_k -- writes ONLY o_U's level-0 (current) entry, using
// TODAY's alpha_k^{n+1} and o_U_phases[k]'s own level-0 (see fluidSolver.hpp's comment: at
// this call site solveVelocity() for THIS step hasn't run yet, so this is last step's
// interior values + this step's fresh Dirichlet boundary values -- same as what the old,
// buggy full-rebuild wrote for level 0 specifically; only levels 1..n-1 change, see Plan J,
// wk_claude/plans/J_*.md). Classic case: o_U_phases[0] already aliases o_U directly
// (solveVelocity wrote it there), nothing to do.
void fluidSolver_t::updateMixtureVelocity()
{
  if (o_alpha_phases.empty()) {
    return;
  }

  auto o_level0 = o_U.slice(0, fieldOffsetSum);
  platform->linAlg->fill(fieldOffsetSum, 0.0, o_level0);
  for (int k = 0; k < Nphase; k++) {
    auto o_term = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);
    o_term.copyFrom(o_U_phases[k].slice(0, fieldOffsetSum), fieldOffsetSum);
    platform->linAlg->axmyVector(mesh->Nlocal, fieldOffset, 0, 1.0, o_alpha_phases[k], o_term);
    platform->linAlg->axpbyMany(mesh->Nlocal, mesh->dim, fieldOffset, 1.0, o_term, 1.0, o_level0);
  }
}

// (v2, 108)'s forcing bracket: Sum_q a_q*ubar^{n-q}/dt - Sum_k alpha_k^{n+1}*[Sum_q
// b_q*(u_k.grad(u_k))^{n-q}] + g (the "+g" is Sum_k alpha_k^{n+1}*[EXT-combined
// o_EXT_phases[k]] -- userSource's gravity/drag/etc, ADDED like sumMakef's own FU term, not
// subtracted like the ADV term; a first attempt at this fix dropped this term entirely,
// caught by case1 converging to a visibly worse residual against eq 21 -- 2.7e-4 vs the
// previously-established 7e-5 -- instead of being bit-identical, which it must be: case1's
// alpha is hardcoded constant from t=0, so this fix is a pure algebraic rearrangement with
// zero behavior change for any constant-alpha case). Both pieces density-free, matching the
// existing convention (confirmed by reading sumMakef.okl: RHO is passed in but never
// referenced in the kernel body; density is applied downstream at the point of use). Classic
// case is just a copy (unchanged); unlike updateMixtureVelocity(), this always runs --
// o_JwF_phases[0] is never aliased to bare o_JwF (phase 0's own forcing must survive past
// this call, for solveVelocity(0) to use).
void fluidSolver_t::updateMixtureForcing()
{
  if (o_alpha_phases.empty()) {
    o_JwF.copyFrom(o_JwF_phases[0], fieldOffsetSum);
    return;
  }

  // BDF part: from o_U's OWN (correctly historical, see updateMixtureVelocity) lag history --
  // NOT alpha-reweighted. Plan J fixed a bug here: this used to come from
  // Sum_k alpha_k^{n+1}*o_JwF_phases[k] (today's alpha applied to each phase's ALREADY
  // BDF+ADV-combined forcing), which wrongly re-weighted the BDF part with today's alpha too.
  combineLagLevels(o_coeffBDF, o_U, static_cast<int>(o_coeffBDF.size()), o_JwF);
  platform->linAlg->scale(fieldOffsetSum, 1.0 / dt[0], o_JwF);

  // "+g" part: Sum_k alpha_k^{n+1} * [EXT-combined o_EXT_phases[k]] (userSource's own
  // gravity/drag/forcing history, built in makeExplicit()/initInnerStep, same EXT-coefficient
  // treatment as the ADV part below but ADDED, matching sumMakef.okl's own
  // "nltx += JW*coeffEXT[s]*FU[...]" convention for this term).
  {
    const auto nEXT = static_cast<int>(o_coeffEXT.size());
    for (int k = 0; k < Nphase; k++) {
      auto o_extK = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);
      combineLagLevels(o_coeffEXT, o_EXT_phases[k], nEXT, o_extK);
      platform->linAlg->axmyVector(mesh->Nlocal, fieldOffset, 0, 1.0, o_alpha_phases[k], o_extK); // *= alpha_k
      platform->linAlg->axpbyMany(mesh->Nlocal, mesh->dim, fieldOffset, 1.0, o_extK, 1.0, o_JwF);
    }
  }

  // ADV part: Sum_k alpha_k^{n+1} * [Sum_q b_q*(u_k.grad(u_k))^{n-q}] -- each phase's OWN
  // o_ADV_phases[k] lag history (built in makeForcing()/initInnerStep), EXT-combined BEFORE
  // alpha-weighting, not the already-BDF+ADV-combined o_JwF_phases[k].
  const auto nEXT = static_cast<int>(o_coeffEXT.size());
  for (int k = 0; k < Nphase; k++) {
    auto o_advK = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);
    combineLagLevels(o_coeffEXT, o_ADV_phases[k], nEXT, o_advK);
    platform->linAlg->axmyVector(mesh->Nlocal, fieldOffset, 0, 1.0, o_alpha_phases[k], o_advK); // *= alpha_k
    platform->linAlg->axpbyMany(mesh->Nlocal, mesh->dim, fieldOffset, -1.0, o_advK, 1.0, o_JwF);
  }
}

void fluidSolver_t::solvePressure(double time, int stage)
{
  if (!ellipticSolverP) {
    return;
  }

  platform->timer.tic(pressureName + "Solve");

  const auto g0idt = *g0 / dt[0];

  double flopCount = 0.0;
  platform->timer.tic(pressureName + " rhs");

  auto o_lambda0 = [&](bool variable = true) {
    auto o_lambda = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
    if (platform->options.compareArgs(upperCase(pressureName) + " RHO SPLITTING", "TRUE") && !variable) {
      platform->linAlg->fill(mesh->Nlocal, 1 / rho0, o_lambda);
    } else {
      platform->linAlg->adyz(mesh->Nlocal, 1.0, o_rho, o_lambda);
    }
    return o_lambda;
  };

  const auto o_rhoSplitTerm = [&]() {
    occa::memory o_del;
    if (platform->options.compareArgs(upperCase(pressureName) + " RHO SPLITTING", "TRUE")) {
      // 1/rho - 1/rho0
      auto o_lambda = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
      platform->linAlg->adyz(mesh->Nlocal, 1.0, o_rho, o_lambda);
      platform->linAlg->add(mesh->Nlocal, -1 / rho0, o_lambda);

      o_del = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);

      const auto valSave = ellipticSolverP->options().getArgs("ELLIPTIC COEFF FIELD");
      ellipticSolverP->options().setArgs("ELLIPTIC COEFF FIELD", "TRUE");
      ellipticSolverP->Ax(o_lambda, o_NULL, o_Pe, o_del);
      ellipticSolverP->options().setArgs("ELLIPTIC COEFF FIELD", valSave);
    }
    return o_del;
  }();

  // TODO 2phase (eq 101, 2nd line): this lambda only builds the mixture term
  // nu_bar*(2*grad(Q_bar) - curl(omega_bar)) from the mixture o_Ue/o_mue/o_div. It is MISSING
  // the per-phase diffusion-velocity viscous term
  //   -div( sum_k alpha_k * div( nu_k*(grad(Ubar_k) + grad(Ubar_k)^T) ) )
  // (Ubar_k = u_k - u_bar, eq 86) entirely -- there is no code anywhere computing Ubar_k or
  // this sum. Also missing: the phase-change/interfacial source term
  // -div( sum_k [(u_ki-u_k)*Gamma_k/rho_k + M_k/rho_k] ) (eq 101 last line); currently assumed
  // zero (no phase-change / no explicit interfacial exchange model exists).
  const auto o_stressTerm = [&]() {
    auto o_curl = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);

    launchKernel("core-curlHex3D", mesh->Nelements, 1, mesh->o_vgeo, mesh->o_D, fieldOffset, o_Ue, o_curl);
    flopCount += static_cast<double>(mesh->Nelements) * (18 * mesh->Np * mesh->Nq + 36 * mesh->Np);

    oogs::startFinish(o_curl, mesh->dim, fieldOffset, ogsDfloat, ogsAdd, mesh->oogs3);

    platform->linAlg->axmyVector(mesh->Nlocal, fieldOffset, 0, 1.0, mesh->o_invLMM, o_curl);
    flopCount += mesh->Nlocal;

    auto o_stressTerm = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);
    launchKernel("core-curlHex3D",
                 mesh->Nelements,
                 1,
                 mesh->o_vgeo,
                 mesh->o_D,
                 fieldOffset,
                 o_curl,
                 o_stressTerm);
    flopCount += static_cast<double>(mesh->Nelements) * (18 * mesh->Np * mesh->Nq + 36 * mesh->Np);

    if (platform->options.compareArgs(upperCase(name) + " STRESSFORMULATION", "TRUE")) {
      launchKernel("fluidSolver_t::pressureStressHex3D",
                   mesh->Nelements,
                   mesh->o_vgeo,
                   mesh->o_D,
                   fieldOffset,
                   o_mue,
                   o_Ue,
                   o_div,
                   o_stressTerm);
      flopCount += static_cast<double>(mesh->Nelements) * (18 * mesh->Nq * mesh->Np + 100 * mesh->Np);
    }
    return o_stressTerm;
  }();

  // o_JwF here is the alpha-weighted combination of each phase's own (density-free) forcing
  // (eq 99), built by updateMixtureForcing() -- called at the top of solve(), before
  // solvePressure.
  auto o_rhs = [&]() {
    auto o_gradDiv = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);

    launchKernel("core-gradientVolumeHex3D",
                 mesh->Nelements,
                 mesh->o_vgeo,
                 mesh->o_D,
                 fieldOffset,
                 o_div,
                 o_gradDiv);
    flopCount += static_cast<double>(mesh->Nelements) * (6 * mesh->Np * mesh->Nq + 18 * mesh->Np);

    auto o_rhs = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);

#if 1
    launchKernel("fluidSolver_t::pressureRhsHex3D",
                 mesh->Nlocal,
                 fieldOffset,
                 o_mue,
                 o_lambda0(),
                 o_JwF,
                 o_stressTerm,
                 o_gradDiv,
                 o_rhs);
    flopCount += 12 * static_cast<double>(mesh->Nlocal);
#else
    o_rhs.copyFrom(this->o_JwF);
#endif

    oogs::startFinish(o_rhs, mesh->dim, fieldOffset, ogsDfloat, ogsAdd, mesh->oogs3);
    platform->linAlg->axmyVector(mesh->Nlocal, fieldOffset, 0, 1.0, mesh->o_invLMM, o_rhs);

    return o_rhs;
  }();

  auto o_pRhs = [&]() {
    auto o_pRhs = platform->deviceMemoryPool.reserve<dfloat>(fieldOffset);

    launchKernel("core-wDivergenceVolumeHex3D",
                 mesh->Nelements,
                 mesh->o_vgeo,
                 mesh->o_D,
                 fieldOffset,
                 o_rhs,
                 o_pRhs);
    flopCount += static_cast<double>(mesh->Nelements) * (6 * mesh->Np * mesh->Nq + 18 * mesh->Np);

    launchKernel("fluidSolver_t::pressureAddQtl", mesh->Nlocal, mesh->o_LMM, g0idt, o_div, o_pRhs);
    flopCount += 3 * mesh->Nlocal;

    // Plan K's (v2,105) "lagged Q_bar" term -- REMOVED (Plan M finding, wk_claude/plans/
    // M_manufactured_profile_staged_freeze_debug.md). Re-deriving eq 105->106->108: the
    // OCR'd "Sum_q a_q*Q_bar_k^{n-q}/dt" in eq 105 is a mis-read of "Sum_q a_q*u_bar^{n-q}
    // /dt" (bold u-bar, the VELOCITY vector, confirmed unambiguously by eq 106's own
    // rendering of the identical bracket using u_bar). That term is NOT an independent
    // scalar source (f,q)_Omega needing its own Qtl-style addition -- it is a VECTOR folded
    // into the gradient-type weak form (f,grad(q))_Omega, the same bucket as advection/
    // gravity, whose boundary-flux counterterm cancels out via the pressure-BC substitution
    // (eq 107) in the final eq 108, leaving only the existing (n.u_bar^{n+1}) boundary term
    // (divergenceSurfaceHex3D below, using TODAY's o_U, already correct).
    // updateMixtureForcing() (Plan J, pre-existing) ALREADY computes this exact quantity
    // correctly via that same gradient route: its "BDF part" is
    // combineLagLevels(o_coeffBDF, o_U, ..., o_JwF); scale(1/dt[0], o_JwF) -- i.e.
    // Sum_q a_q*u_bar^{n-q}/dt, fed into o_JwF -> pressureRhsHex3D ->
    // core-wDivergenceVolumeHex3D, the correct (f,grad(q))_Omega assembly with boundary
    // handling already covered by divergenceSurfaceHex3D.
    // This block instead took the DIVERGENCE of that same quantity and injected it as a
    // second, separately-discretized scalar source via pressureAddQtl (missing its own
    // boundary term) -- a duplicate of a term already present, not a missing one. case1's
    // bit-identical regression never caught this because case1 has alpha_d=0 exactly, so
    // u_bar=u_c has near-zero divergence throughout (ordinary single-phase guarantee),
    // making this term's own divergence numerically negligible there regardless of sign --
    // not because it was correct. Confirmed via Plan M's L1a/L0 staged-freeze tests:
    // divUErr grew linearly (not flat) once alpha had real spatial structure AND velocity
    // genuinely evolved, consistent with this redundant term's own magnitude scaling with
    // however much Q_bar had already drifted (self-reinforcing bias, not exponential
    // instability).

    launchKernel("fluidSolver_t::divergenceSurfaceHex3D",
                 mesh->Nelements,
                 mesh->o_sgeo,
                 mesh->o_vmapM,
                 o_EToB,
                 g0idt,
                 fieldOffset,
                 o_rhs,
                 o_U,
                 o_pRhs);
    flopCount += 25 * static_cast<double>(mesh->Nelements) * mesh->Nq * mesh->Nq;

    // (v2, 105)'s RED term, UDF-opt-in (see fluidSolver.hpp's own comment on userRedTerm --
    // not implemented by default, same documented gap as always).
    if (userRedTerm) {
      auto o_redTerm = userRedTerm(time);
      if (o_redTerm.isInitialized()) {
        launchKernel("fluidSolver_t::pressureAddQtl", mesh->Nlocal, mesh->o_LMM, -1.0, o_redTerm, o_pRhs);
        flopCount += 3 * mesh->Nlocal;
      }
    }

    if (o_rhoSplitTerm.isInitialized()) {
      platform->linAlg->axpby(mesh->Nlocal, -1.0, o_rhoSplitTerm, 1.0, o_pRhs);
    }

    return o_pRhs;
  }();

  platform->timer.toc(pressureName + " rhs");
  platform->flopCounter->add(pressureName + " rhs", flopCount);

  ellipticSolverP->solve(o_lambda0(false), o_NULL, o_pRhs, o_P.slice(0, mesh->Nlocal));

  if (platform->verbose()) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
                                                                 1,
                                                                 fieldOffset,
                                                                 mesh->ogs->o_invDegree,
                                                                 o_P,
                                                                 platform->comm.mpiComm());
    if (platform->comm.mpiRank() == 0) {
      printf("p norm: %.15e\n", debugNorm);
    }
  }

  platform->timer.toc(pressureName + "Solve");
}

// Solves phase `phase`'s own momentum equation (eq 97): rho_k/mue_k are phase-specific,
// p/grad(p) is the SHARED mixture pressure (correct as-is, eq 97's -grad(p) term uses the
// same p for every phase). o_div is still the mixture divergence (no per-phase divergence
// field exists -- a remaining approximation, see note below) and the forcing o_JwF is still
// mixture-level (advection/BDF not yet split per phase, see makeForcing/makeAdvection) --
// both documented, bounded simplifications, not silently assumed correct.
// ellipticSolver stays ONE shared set (see setupEllipticSolver); only lambda0/lambda1 passed
// into .solve() below are phase-specific, which is enough to get phase_k's correct operator
// (the shared set's own preconditioner is still built from mixture coefficients).
void fluidSolver_t::solveVelocity(double time, int stage, int phase)
{
  if (ellipticSolver.size() == 0) {
    return;
  }

  auto &o_Uphase = o_U_phases[phase];
  auto &o_rho_k = o_rho_phases[phase];
  auto &o_mue_k = o_mue_phases[phase];

  platform->timer.tic(velocityName[phase] + "Solve");

  const auto g0idt = *g0 / dt[0];

  double flopCount = 0.0;
  platform->timer.tic(velocityName[phase] + " rhs");

  // TODO 2phase (eq 97): o_div should be phase k's own divergence div(u_k); there is no
  // per-phase divergence field, so the mixture o_div is used here as an approximation.
  const auto o_gradMueDiv = [&]() {
    dfloat scale = 1. / 3;
    if (platform->options.compareArgs(upperCase(name) + " STRESSFORMULATION", "TRUE")) {
      scale = -2 * scale;
    }

    auto o_mueDiv = platform->deviceMemoryPool.reserve<dfloat>(fieldOffset);
    platform->linAlg->axmyz(mesh->Nlocal, scale, o_mue_k, o_div, o_mueDiv);

    auto o_gradMueDiv = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);

    launchKernel("core-gradientVolumeHex3D",
                 mesh->Nelements,
                 mesh->o_vgeo,
                 mesh->o_D,
                 fieldOffset,
                 o_mueDiv,
                 o_gradMueDiv);
    flopCount += static_cast<double>(mesh->Nelements) * (6 * mesh->Np * mesh->Nq + 18 * mesh->Np);

    return o_gradMueDiv;
  }();

  // TODO: RHO SPLITTING still uses the mixture rho0/o_rho (shared reference point) rather
  // than a phase-specific rho0_k -- see note in [[02_momentum_pressure_phase_gaps]], not
  // obviously the right generalization for very different phase densities.
  const auto o_rhoSplitTerm = [&]() {
    occa::memory o_del;
    if (platform->options.compareArgs(upperCase(pressureName) + " RHO SPLITTING", "TRUE")) {
      o_del = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);
      auto o_delta = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);

      platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_P, -1.0, o_Pe, o_delta);
      launchKernel("core-gradientVolumeHex3D",
                   mesh->Nelements,
                   mesh->o_vgeo,
                   mesh->o_D,
                   fieldOffset,
                   o_delta,
                   o_del);

      flopCount += static_cast<double>(mesh->Nelements) * (6 * mesh->Np * mesh->Nq + 18 * mesh->Np);

      // o_del * rho_k / rho0
      platform->linAlg->axmyVector(mesh->Nlocal, fieldOffset, 0, 1 / rho0, o_rho_k, o_del);
    }
    return o_del;
  }();

  const auto o_gradP = [&]() {
    occa::memory o_gradP = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);

    auto o_P = this->o_P;
    if (platform->options.compareArgs(upperCase(pressureName) + " RHO SPLITTING", "TRUE")) {
      o_P = o_Pe;
    }

#if 1
    // use weak formulation to allow for non-zero Dirichlet pressure BC when using stressFormulation
    launchKernel("core-wGradientVolumeHex3D",
                 mesh->Nelements,
                 mesh->o_vgeo,
                 mesh->o_D,
                 fieldOffset,
                 o_P,
                 o_gradP);
#else
    launchKernel("core-gradientVolumeHex3D",
                 mesh->Nelements,
                 mesh->o_vgeo,
                 mesh->o_D,
                 fieldOffset,
                 o_P,
                 o_gradP);
    platform->linAlg->scale(o_gradP.size(), -1.0, o_gradP);
#endif
    flopCount += static_cast<double>(mesh->Nelements) * 18 * (mesh->Np * mesh->Nq + mesh->Np);

    return o_gradP;
  }();

  // BUG FIX (eq 97), found+fixed this session: this used to pass the bare, MIXTURE-combined
  // o_JwF (eq 99's alpha-weighted sum, meant for the PRESSURE equation) into every phase's
  // own velocity RHS, instead of that phase's own o_JwF_phases[phase] (eq 97's rho_k*(BDF_k -
  // EXT_k), phase k's own time-derivative + advection history). For any case with alpha_k==0
  // for some phase (e.g. term_demo_case1's dispersed phase), the mixture sum's alpha_k
  // weighting zeroed that phase's own forcing out of its OWN velocity equation entirely,
  // leaving only the shared pressure gradient's effect -- explained the "jumps once then
  // freezes, never accumulates" symptom exactly (confirmed via DBGCORE printf cross-checked
  // against a hand-derived expected RHS). Still missing (unrelated, not fixed here): the
  // interfacial/phase-change exchange term (u_ki-u_k)*Gamma_k/alpha_k + M_k/alpha_k (eq
  // 94/97) -- not modeled anywhere, assumed zero (left to the UDF's own userSource).
  const auto o_rhs = [&]() {
    auto o_rhs = platform->deviceMemoryPool.reserve<dfloat>(fieldOffsetSum);

    // o_rho_k * o_JwF_phases[phase] + o_gradP + o_gradMueDiv
    launchKernel("fluidSolver_t::velocityRhsHex3D",
                 mesh->Nlocal,
                 fieldOffset,
                 o_rho_k,
                 o_JwF_phases[phase],
                 o_gradMueDiv,
                 o_gradP,
                 o_rhs);
    flopCount += 9 * mesh->Nlocal;

    // o_Uphase stands in for the extrapolated o_Ue_phases[phase], which nothing populates yet
    // (TODO) -- using the current solution instead of an extrapolated guess here is safe
    // (just a slightly less accurate linearization point for this Neumann BC term).
    launchKernel("fluidSolver_t::velocityNeumannBCHex3D",
                 o_velocityName[phase],
                 o_pressureName,
                 mesh->Nelements,
                 fieldOffset,
                 mesh->o_sgeo,
                 mesh->o_vmapM,
                 mesh->o_EToB,
                 o_EToB,
                 time,
                 mesh->o_x,
                 mesh->o_y,
                 mesh->o_z,
                 o_rho_k,
                 o_mue_k,
                 platform->app->bc->o_usrwrk,
                 o_Uphase,
                 o_rhs);
    flopCount += static_cast<double>(mesh->Nelements) * (3 * mesh->Np + 36 * mesh->Nq * mesh->Nq);

    if (o_rhoSplitTerm.isInitialized()) {
      platform->linAlg->axpbyMany(mesh->Nlocal, mesh->dim, fieldOffset, -1.0, o_rhoSplitTerm, 1.0, o_rhs);
    }

    return o_rhs;
  }();

  platform->timer.toc(velocityName[phase] + " rhs");
  platform->flopCounter->add(velocityName[phase] + " rhs", flopCount);

  const auto o_lambda0 = o_mue_k;
  const auto o_lambda1 = [&]() {
    auto o_lambda1 = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
    if (userImplicitLinearTerm) {
      auto o_implicitLT = userImplicitLinearTerm(time);
      platform->linAlg->axpbyz(mesh->Nlocal, g0idt, o_rho_k, 1.0, o_implicitLT, o_lambda1);
    } else {
      platform->linAlg->axpby(mesh->Nlocal, g0idt, o_rho_k, 0.0, o_lambda1);
    }
    return o_lambda1;
  }();

  // TODO: no per-phase extrapolated initial guess yet (o_Ue_phases[phase] is unpopulated --
  // see makeExplicit/extrapolateSolution TODOs), so the implicit solve always starts from the
  // previous solution. Safe (slower convergence at worst), not a correctness issue.

  // shared ellipticSolver set (built in setupEllipticSolver from mixture mue/rho for its
  // matrix structure/preconditioner); o_lambda0/o_lambda1 above are phase-specific, giving
  // phase_k's correct operator for the actual solve.
  //
  // Snapshot each component's convergence stats into velocityConvergence[phase] right after
  // its own solve() call -- ellipticSolver is reused across phases, so its live Niter()/
  // residual() state would otherwise be overwritten by the next phase before
  // nrs_t::printStepInfo gets a chance to print it.
  auto snapshot = [&](int component) {
    auto &c = velocityConvergence[phase][component];
    c.Niter = ellipticSolver.at(component)->Niter();
    c.initialGuessResidual = ellipticSolver.at(component)->initialGuessResidual();
    c.finalResidual = ellipticSolver.at(component)->finalResidual();
  };

  if (ellipticSolver.at(0)->Nfields() > 1) {
    ellipticSolver.at(0)->solve(o_lambda0, o_lambda1, o_rhs, o_Uphase.slice(0, fieldOffsetSum));
    snapshot(0);
  } else {
    const auto o_rhsX = o_rhs.slice(0 * fieldOffset, mesh->Nlocal);
    const auto o_rhsY = o_rhs.slice(1 * fieldOffset, mesh->Nlocal);
    const auto o_rhsZ = o_rhs.slice(2 * fieldOffset, mesh->Nlocal);
    ellipticSolver.at(0)->solve(o_lambda0, o_lambda1, o_rhsX, o_Uphase.slice(0 * fieldOffset, mesh->Nlocal));
    snapshot(0);
    ellipticSolver.at(1)->solve(o_lambda0, o_lambda1, o_rhsY, o_Uphase.slice(1 * fieldOffset, mesh->Nlocal));
    snapshot(1);
    ellipticSolver.at(2)->solve(o_lambda0, o_lambda1, o_rhsZ, o_Uphase.slice(2 * fieldOffset, mesh->Nlocal));
    snapshot(2);
  }

  if (platform->verbose()) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
                                                                 mesh->dim,
                                                                 fieldOffset,
                                                                 mesh->ogs->o_invDegree,
                                                                 o_Uphase,
                                                                 platform->comm.mpiComm());
    if (platform->comm.mpiRank() == 0) {
      printf("U norm (%s): %.15e\n", velocityName[phase].c_str(), debugNorm);
    }
  }

  platform->timer.toc(velocityName[phase] + "Solve");
}

// Builds ONE shared ellipticSolver/ellipticSolverP set from mixture mue/rho/BC (fixed to
// compile against mixtureVelocityName rather than the now-vectorized velocityName).
// Deliberately NOT one set per phase (that would need real per-phase preconditioner setup,
// a bigger change) -- solveVelocity() instead passes phase-specific lambda0/lambda1 into
// this shared solver's .solve() per call, which IS correct for the operator itself; only the
// preconditioner is still built from mixture coefficients (a known, bounded approximation).
void fluidSolver_t::setupEllipticSolver()
{
  if (platform->options.compareArgs(upperCase(mixtureVelocityName) + " SOLVER", "NONE")) {
    return;
  }

  if (Nphase > 1) {
    // Restriction: the shared ellipticSolver's preconditioner is only confirmed to rebuild
    // correctly from phase-specific lambda0/lambda1 for PRECONDITIONER=JACOBI
    // (ellipticUpdateAllJacobi runs inside elliptic::_solve on every call). MULTIGRID/AMG
    // coarse-grid setup has not been verified to do the same -- refuse rather than silently
    // reuse a preconditioner that may be frozen at the mixture's coefficients.
    for (int i = 0; i < Nphase; i++) {
      nekrsCheck(!platform->options.compareArgs(upperCase(velocityName[i]) + " PRECONDITIONER", "JACOBI"),
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "2-phase: %s shares one elliptic solver across phases, currently only "
                 "verified correct for PRECONDITIONER=JACOBI -- got %s PRECONDITIONER=%s\n",
                 velocityName[i].c_str(),
                 velocityName[i].c_str(),
                 platform->options.getArgs(upperCase(velocityName[i]) + " PRECONDITIONER").c_str());
    }

    // Restriction: o_EToB (BC application) and each elliptic object's own EToB (operator/
    // mask structure) are built ONCE from mixtureVelocityName's boundaryTypeMap only (see
    // o_EToB construction above and the EToBx/y/z lambdas below) -- a per-phase
    // boundaryTypeMap override would currently be silently ignored, not actually produce
    // mixed BC. Refuse rather than let a case believe per-phase BC is in effect: there isn't
    // enough math support for mixed BC yet (Plan C left the option open, not implemented).
    //
    // Compare phases against EACH OTHER (not against the mixture's own option): (a)
    // "FLUID VELOCITY BOUNDARY TYPE MAP" is gone by this point anyway -- nrs_t::bdry::setup()
    // (called long before fluidSolver_t even exists) removes every option whose key contains
    // "BOUNDARY TYPE MAP" once consumed, so there is nothing reliable left to compare against
    // on the mixture side; (b) a user could in principle set boundaryTypeMap only on each
    // [FLUID PHASE] section with nothing on [FLUID VELOCITY] at all -- phase-vs-phase is the
    // right invariant either way, not phase-vs-mixture. (Self-caught: the previous version
    // of this check compared against the mixture's option directly and failed unconditionally
    // for every phased case, since that option was always already empty by this point.)
    const auto referenceBcMap = platform->options.getArgs(upperCase(velocityName[0]) + " BOUNDARY TYPE MAP");
    nekrsCheck(referenceBcMap.empty(),
               platform->comm.mpiComm(),
               EXIT_FAILURE,
               "2-phase: %s boundaryTypeMap is not set (neither on the phase nor inherited "
               "from FLUID VELOCITY)\n",
               velocityName[0].c_str());
    for (int i = 1; i < Nphase; i++) {
      const auto phaseBcMap = platform->options.getArgs(upperCase(velocityName[i]) + " BOUNDARY TYPE MAP");
      nekrsCheck(phaseBcMap.empty() || phaseBcMap != referenceBcMap,
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "2-phase: %s boundaryTypeMap differs from %s's (or is unset) -- mixed BC per "
                 "phase is not supported yet (shared o_EToB/elliptic EToB)\n",
                 velocityName[i].c_str(),
                 velocityName[0].c_str());
    }
  }

  // builds ONE shared ellipticSolver set (matrix structure/preconditioner only) from the
  // mixture mue/rho; solveVelocity() passes phase-specific lambda0/lambda1 into .solve()
  // per call, same pattern solvePressure already uses for its own coefficients.
  auto o_lambda0 = o_mue;
  auto o_lambda1 = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  platform->linAlg->axpby(mesh->Nlocal, *g0 / dt[0], o_rho, 0.0, o_lambda1);

  const auto unalignedBoundary = platform->app->bc->hasUnalignedMixed(mixtureVelocityName);

  if (platform->options.compareArgs(upperCase(mixtureVelocityName) + " SOLVER", "BLOCK")) {
    if (platform->options.compareArgs(upperCase(name) + " STRESSFORMULATION", "TRUE")) {
      platform->options.setArgs(upperCase(mixtureVelocityName) + " STRESSFORMULATION", "TRUE");
    }

    ellipticSolver.push_back(new elliptic(mixtureVelocityName, mesh, fieldOffset, o_lambda0, o_lambda1));

    if (unalignedBoundary) {
      o_zeroNormalMask = mesh->createZeroNormalMask(fieldOffset, ellipticSolver[0]->o_EToB());

      auto f = [this](dlong Nelements, const occa::memory &o_elementList, occa::memory &o_x) {
        launchKernel("mesh-applyZeroNormalMask",
                     Nelements,
                     this->fieldOffset,
                     o_elementList,
                     this->mesh->o_sgeo,
                     this->o_zeroNormalMask,
                     this->mesh->o_vmapM,
                     this->ellipticSolver.at(0)->o_EToB(),
                     o_x);
      };
      ellipticSolver.at(0)->applyZeroNormalMask(f);
    }
  } else {
    nekrsCheck(unalignedBoundary,
               platform->comm.mpiComm(),
               EXIT_FAILURE,
               "unaligned mixed boundary conditions require using block solver for %s\n",
               mixtureVelocityName.c_str());
    auto EToBx = mesh->createEToB(
        [&](int bID) -> int { return platform->app->bc->typeElliptic(bID, mixtureVelocityName, "x"); });

    auto EToBy = mesh->createEToB(
        [&](int bID) -> int { return platform->app->bc->typeElliptic(bID, mixtureVelocityName, "y"); });

    auto EToBz = mesh->createEToB(
        [&](int bID) -> int { return platform->app->bc->typeElliptic(bID, mixtureVelocityName, "z"); });

    ellipticSolver.push_back(new elliptic(mixtureVelocityName, mesh, fieldOffset, EToBx, o_lambda0, o_lambda1));
    ellipticSolver.push_back(new elliptic(mixtureVelocityName, mesh, fieldOffset, EToBy, o_lambda0, o_lambda1));
    ellipticSolver.push_back(new elliptic(mixtureVelocityName, mesh, fieldOffset, EToBz, o_lambda0, o_lambda1));
  }

  velocityConvergence.assign(Nphase, std::vector<velocityConvergence_t>(ellipticSolver.size()));

  if (Nphase > 1 && platform->comm.mpiRank() == 0) {
    // one shared elliptic set was just built above (ELLIPTIC SETUP banner, named
    // mixtureVelocityName) -- each phase REUSES it (phase-specific lambda0/lambda1 passed
    // into .solve() per call, see solveVelocity), it is not rebuilt per phase from scratch.
    for (int i = 0; i < Nphase; i++) {
      std::cout << "  " << upperCase(velocityName[i]) << ": copy of " << upperCase(mixtureVelocityName)
                << " elliptic setup (not rebuilt from scratch)" << std::endl;
    }
  }

  {
    const auto o_lambda0 = [&]() {
      auto o_lambda = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
      if (platform->options.compareArgs(upperCase(pressureName) + " RHO SPLITTING", "TRUE")) {
        rho0 = platform->linAlg->min(mesh->Nlocal, o_rho, platform->comm.mpiComm());
        platform->linAlg->fill(mesh->Nlocal, 1 / rho0, o_lambda);
      } else {
        platform->linAlg->adyz(mesh->Nlocal, 1.0, o_rho, o_lambda);
      }
      return o_lambda;
    }();

    // derive from velocity
    auto EToBP = mesh->createEToB([&](int bID) -> int {
      auto bcType = platform->app->bc->typeId(bID, mixtureVelocityName);
      if (bID < 1) {
        return ellipticBcType::NO_OP;
      }
      return platform->app->bc->isOutflow(bcType) ? ellipticBcType::DIRICHLET : ellipticBcType::NEUMANN;
    });

    ellipticSolverP = new elliptic(pressureName, mesh, 0, EToBP, o_lambda0, o_NULL);
  }
}

// Per-phase velocity BC loop (pressure BC redundantly recomputed each phase -- idempotent,
// deterministic, harmless at Nphase<=2). Zero-normal-mask stays mixture-level (shared BC
// topology, same mesh/EToB for every phase).
void fluidSolver_t::applyDirichlet(double time)
{
  if (ellipticSolver.size() == 0) {
    return;
  }

  if (platform->app->bc->hasUnalignedMixed(mixtureVelocityName)) {
    launchKernel("mesh-applyZeroNormalMask",
                 mesh->Nelements,
                 fieldOffset,
                 mesh->o_elementList,
                 mesh->o_sgeo,
                 o_zeroNormalMask,
                 mesh->o_vmapM,
                 ellipticSolver.at(0)->o_EToB(),
                 o_U);
    launchKernel("mesh-applyZeroNormalMask",
                 mesh->Nelements,
                 fieldOffset,
                 mesh->o_elementList,
                 mesh->o_sgeo,
                 o_zeroNormalMask,
                 mesh->o_vmapM,
                 ellipticSolver.at(0)->o_EToB(),
                 o_Ue);
  }

  // lower than any other possible Dirichlet value
  static constexpr dfloat TINY = -1e30;
  auto &neknek = platform->app->neknek;

  for (int phase = 0; phase < Nphase; phase++) {
    occa::memory o_tmp = platform->deviceMemoryPool.reserve<dfloat>((mesh->dim + 1) * fieldOffset);
    platform->linAlg->fill(o_tmp.size(), TINY, o_tmp);

    for (int sweep = 0; sweep < 2; sweep++) {
      launchKernel("fluidSolver_t::pressureDirichletBCHex3D",
                   o_pressureName,
                   mesh->Nelements,
                   time,
                   fieldOffset,
                   mesh->o_sgeo,
                   mesh->o_x,
                   mesh->o_y,
                   mesh->o_z,
                   mesh->o_vmapM,
                   mesh->o_EToB,
                   o_EToB,
                   o_rho,
                   o_mue,
                   platform->app->bc->o_usrwrk,
                   o_Ue,
                   o_tmp);

      launchKernel("fluidSolver_t::velocityDirichletBCHex3D",
                   o_velocityName[phase],
                   mesh->Nelements,
                   fieldOffset,
                   time,
                   mesh->o_sgeo,
                   mesh->o_x,
                   mesh->o_y,
                   mesh->o_z,
                   mesh->o_vmapM,
                   mesh->o_EToB,
                   o_EToB,
                   o_rho_phases[phase],
                   o_mue_phases[phase],
                   neknek ? neknek->intValOffset() : 0,
                   neknek ? neknek->o_pointMap() : o_NULL,
                   neknek ? neknek->getField(velocityName[phase]).o_intVal : o_NULL,
                   platform->app->bc->o_usrwrk,
                   o_U_phases[phase],
                   o_tmp.slice(fieldOffset));

      oogs::startFinish(o_tmp,
                        1 + mesh->dim,
                        fieldOffset,
                        ogsDfloat,
                        (sweep == 0) ? ogsMax : ogsMin,
                        mesh->oogs3);
    }

    if (phase == 0 && ellipticSolverP->Nmasked()) {
      // pressure is shared -- apply its Dirichlet values into o_P only once
      auto o_dirichletValues = o_tmp.slice(0, fieldOffset);
      launchKernel("core-maskCopy",
                   ellipticSolverP->Nmasked(),
                   0,
                   0,
                   ellipticSolverP->o_maskIds(),
                   o_dirichletValues,
                   o_P);
    }

    int cnt = 0;
    for (auto &solver : ellipticSolver) {
      if (solver->Nmasked()) {
        const auto offset = cnt * fieldOffset;
        auto o_dirichletValues = o_tmp.slice(fieldOffset, fieldOffsetSum);
        launchKernel("core-maskCopy2",
                     solver->Nmasked(),
                     offset,
                     offset,
                     solver->o_maskIds(),
                     o_dirichletValues,
                     o_U_phases[phase],
                     o_Ue_phases[phase]);
      }
      cnt++;
    }
  }
}

// Phase k's own forcing (BDF time-derivative + advection of u_k) -- density-free, same as
// the single-phase sumMakef output always was (RHO is passed to the kernel but unused there;
// density is applied downstream at the point of use: velocityRhsHex3D multiplies by rho_k
// explicitly for solveVelocity's RHS (eq 97); updateMixtureForcing() just alpha-weights and
// sums this directly for the pressure RHS (eq 99), no rho conversion needed either way.
void fluidSolver_t::makeForcing(int phase)
{
  if (ellipticSolver.size() == 0) {
    return;
  }

  auto &o_rho_k = o_rho_phases[phase];
  auto &o_Uphase = o_U_phases[phase];
  auto &o_ADVphase = o_ADV_phases[phase];
  auto &o_EXTphase = o_EXT_phases[phase];
  auto &o_JwFphase = o_JwF_phases[phase];

  launchKernel("fluidSolver_t::sumMakef",
               mesh->Nlocal,
               mesh->o_Jw,
               1 / dt[0],
               o_coeffEXT,
               o_coeffBDF,
               fieldOffset,
               fieldOffsetSum,
               mesh->fieldOffset, /* o_Jw offset */
               o_rho_k,
               o_Uphase,
               o_ADVphase,
               o_EXTphase,
               o_JwFphase);

  dfloat flops = Nsubsteps ? (6 + 6 * o_coeffEXT.size()) : (6 * o_coeffEXT.size() + 12 * o_coeffBDF.size());
  platform->flopCounter->add(velocityName[phase] + " sumMakef", flops * static_cast<double>(mesh->Nlocal));

  if (platform->verbose()) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
                                                                 mesh->dim,
                                                                 fieldOffset,
                                                                 mesh->ogs->o_invDegree,
                                                                 o_JwFphase,
                                                                 platform->comm.mpiComm());
    if (platform->comm.mpiRank() == 0) {
      printf("%s JwF norm: %.15e\n", velocityName[phase].c_str(), debugNorm);
    }
  }

  for (int s = o_coeffEXT.size(); s > 1; s--) {
    o_EXTphase.copyFrom(o_EXTphase, fieldOffsetSum, (s - 1) * fieldOffsetSum, (s - 2) * fieldOffsetSum);
    if (o_ADVphase.isInitialized()) {
      o_ADVphase.copyFrom(o_ADVphase, fieldOffsetSum, (s - 1) * fieldOffsetSum, (s - 2) * fieldOffsetSum);
    }
  }
}

// eq 92/97: BUG FIX, found+fixed this session (same root cause/category as the o_JwF fix):
// every phase's advection used to convect with the shared MIXTURE o_relUrst (see the old
// comment this replaced -- "the mesh's own motion doesn't differ per phase, so this isn't
// split" conflated the mesh-velocity part of relUrst, which genuinely doesn't vary per
// phase, with the FLUID-velocity part, which does and was wrongly left shared). Computes
// each phase's own convecting velocity from o_U_phases[i] instead of the bare mixture o_U.
// No-op for the classic case (Nphase==1): o_relUrst_phases[0] already aliases the mixture
// o_relUrst, which nrs_t::computeUrst() already updates.
void fluidSolver_t::computeUrstPhases()
{
  if (o_alpha_phases.empty()) {
    return;
  }

  for (int i = 0; i < Nphase; i++) {
    auto &o_relUrstPhase = o_relUrst_phases[i];

    if (Nsubsteps) {
      for (int s = o_coeffEXT.size(); s > 1; s--) {
        const auto lagOffset = mesh->dim * cubatureOffset;
        o_relUrstPhase.copyFrom(o_relUrstPhase, lagOffset, (s - 1) * lagOffset, (s - 2) * lagOffset);
      }
    }

    const auto relative = static_cast<int>(static_cast<bool>(geom) && Nsubsteps);

    double flopCount = 0.0;
    if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
      launchKernel("nrs-UrstCubatureHex3D",
                   mesh->Nelements,
                   relative,
                   mesh->o_cubvgeo,
                   mesh->o_cubInterpT,
                   fieldOffset,
                   (geom) ? geom->fieldOffset : 0,
                   cubatureOffset,
                   o_U_phases[i],
                   (geom) ? geom->o_U : o_NULL,
                   o_relUrstPhase);
      flopCount += 6 * mesh->Np * mesh->cubNq;
      flopCount += 6 * mesh->Nq * mesh->Nq * mesh->cubNq * mesh->cubNq;
      flopCount += 6 * mesh->Nq * mesh->cubNp;
      flopCount += 24 * mesh->cubNp;
      flopCount *= mesh->Nelements;
    } else {
      launchKernel("nrs-UrstHex3D",
                   mesh->Nelements,
                   relative,
                   mesh->o_vgeo,
                   fieldOffset,
                   (geom) ? geom->fieldOffset : 0,
                   o_U_phases[i],
                   (geom) ? geom->o_U : o_NULL,
                   o_relUrstPhase);
      flopCount += 24 * static_cast<double>(mesh->Nlocal);
    }
    platform->flopCounter->add(velocityName[i] + " Urst", flopCount);
  }
}

void fluidSolver_t::makeAdvection(double time, int tstep, int phase)
{
  if (userAdvectionTerm) {
    userAdvectionTerm(time, tstep);
    return;
  }

  if (Nsubsteps) {
    advectionSubcycling(std::min(tstep, static_cast<int>(o_coeffEXT.size())), time, phase);
  } else {
    auto &o_Uconv = o_relUrst_phases[phase];
    auto &o_Uphase = o_U_phases[phase];
    auto &o_ADVphase = o_ADV_phases[phase];

    double flopCount = 0.0;

    if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
      launchKernel("core-strongAdvectionCubatureVolumeHex3D",
                   mesh->Nelements,
                   mesh->o_vgeo,
                   mesh->o_cubDiffInterpT,
                   mesh->o_cubInterpT,
                   mesh->o_cubProjectT,
                   fieldOffset,
                   cubatureOffset,
                   o_Uphase,
                   o_Uconv,
                   o_ADVphase);

      flopCount += 4. * mesh->Nq *
                   (mesh->cubNp + mesh->cubNq * mesh->cubNq * mesh->Nq +
                    mesh->cubNq * mesh->Nq * mesh->Nq); // interpolation
      flopCount += 6. * mesh->cubNp * mesh->cubNq;      // apply Dcub
      flopCount += 5 * mesh->cubNp;                     // compute advection term on cubature mesh
      flopCount += mesh->Np;                            // weight by inv. mass matrix
    } else {
      launchKernel("core-strongAdvectionVolumeHex3D",
                   mesh->Nelements,
                   mesh->o_vgeo,
                   mesh->o_D,
                   fieldOffset,
                   o_Uphase,
                   o_Uconv,
                   o_ADVphase);

      flopCount += 8 * (mesh->Np * mesh->Nq + mesh->Np);
    }

    flopCount *= mesh->Nelements;
    flopCount *= mesh->dim;
    platform->flopCounter->add(velocityName[phase] + " advection", flopCount);
  }
}

// TODO: regularization/mesh-advection still operate on the mixture o_U/o_EXT only (not
// per-phase PHASEk REGULARIZATION options yet -- see makeAdvection/makeForcing); only
// phase 0's call does anything, matching makeAdvection's pattern.
void fluidSolver_t::makeExplicit(double time, int tstep, int phase)
{
  if (ellipticSolver.size() == 0) {
    return;
  }

  auto &o_Uphase = o_U_phases[phase];
  auto &o_EXTphase = o_EXT_phases[phase];
  const auto &name_k = velocityName[phase]; // PHASEk REGULARIZATION, inherited from FLUID VELOCITY if unset

  if (platform->options.compareArgs(upperCase(name_k) + " REGULARIZATION METHOD", "HPFRT")) {
    dfloat strength = NAN;
    platform->options.getArgs(upperCase(name_k) + " HPFRT STRENGTH", strength);

    launchKernel("core-vectorFilterRTHex3D", mesh->Nelements, o_filterRT, strength, fieldOffset, o_Uphase, o_EXTphase);

    double flops = 24 * mesh->Np * mesh->Nq + 3 * mesh->Np;
    flops *= static_cast<double>(mesh->Nelements);
    platform->flopCounter->add(name_k + " filterRT", flops);
  }

  if (platform->options.compareArgs(upperCase(name_k) + " REGULARIZATION METHOD", "GJP")) {
    dfloat tauFactor;
    platform->options.getArgs(upperCase(name_k) + " REGULARIZATION GJP SCALING COEFF", tauFactor);

    for (int i = 0; i < mesh->dim; i++) {
      auto o_EXTi = o_EXTphase.slice(i * fieldOffset, mesh->Nlocal);
      addGJP(mesh, o_EToB, fieldOffset, o_Uphase, o_Uphase.slice(i * fieldOffset, mesh->Nlocal), o_EXTi, tauFactor);
    }
  }

  if (geom && !Nsubsteps) {
    launchKernel("fluidSolver_t::advectMeshVelocityHex3D",
                 mesh->Nelements,
                 mesh->o_vgeo,
                 mesh->o_D,
                 fieldOffset,
                 geom->fieldOffset,
                 geom->o_U,
                 o_Uphase,
                 o_EXTphase);
    double flops = 54 * mesh->Np * mesh->Nq + 63 * mesh->Np;
    flops *= static_cast<double>(mesh->Nelements);
    platform->flopCounter->add(name_k + " advectMeshVelocity", flops);
  }
}

void fluidSolver_t::saveSolutionState()
{
  if (!o_U0.isInitialized()) {
    o_U0 = platform->device.malloc<dfloat>(o_U.size());
    o_P0 = platform->device.malloc<dfloat>(o_P.size());
    o_EXT0 = platform->device.malloc<dfloat>(o_EXT.size());
    o_prop0 = platform->device.malloc<dfloat>(o_prop.size());
    o_relUrst0 = platform->device.malloc<dfloat>(o_relUrst.size());
  }

  o_U0.copyFrom(o_U);
  o_P0.copyFrom(o_P);
  o_EXT0.copyFrom(o_EXT);
  o_prop0.copyFrom(o_prop);
  o_relUrst0.copyFrom(o_relUrst);
}

void fluidSolver_t::restoreSolutionState()
{
  o_U.copyFrom(o_U0);
  o_P.copyFrom(o_P0);
  o_EXT.copyFrom(o_EXT0);
  o_prop.copyFrom(o_prop0);
  o_relUrst.copyFrom(o_relUrst0);
}

// Mixture-only (classic case: o_U_phases[0]/o_Ue_phases[0] alias o_U/o_Ue, so this covers it
// for free; 2-phase case still needs its own per-phase extrapolation loop -- TODO, currently
// unused anyway since solveVelocity doesn't consume o_Ue_phases[phase] yet).
void fluidSolver_t::extrapolateSolution()
{
  if (!platform->options.compareArgs(upperCase(mixtureVelocityName) + " EXTRAPOLATION", "FALSE")) {
    launchKernel("core-extrapolate",
                 mesh->Nlocal,
                 mesh->dim,
                 static_cast<int>(o_coeffEXT.size()),
                 fieldOffset,
                 o_coeffEXT,
                 o_U,
                 o_Ue);
  }

  if (o_coeffEXTP.size()) {
    launchKernel("core-extrapolate",
                 mesh->Nlocal,
                 1,
                 static_cast<int>(o_coeffEXTP.size()),
                 fieldOffset,
                 o_coeffEXTP,
                 o_P,
                 o_Pe);
  }
}

// TODO: saveSolutionState/restoreSolutionState still only snapshot the mixture o_U/o_prop,
// not o_U_phases -- unrelated to the lag-history fix below (outer-iteration state restore,
// not timestep history).
void fluidSolver_t::lagSolution()
{
  {
    const auto n = std::max(o_coeffEXT.size(), o_coeffBDF.size());
    for (int s = n; s > 1; s--) {
      o_U.copyFrom(o_U, fieldOffsetSum, (s - 1) * fieldOffsetSum, (s - 2) * fieldOffsetSum);
    }
  }

  // classic case: o_U_phases[0] aliases o_U (just shifted above) -- skip it here to avoid a
  // double shift. 2-phase case: o_U is reconstructed fresh each step from o_U_phases (see
  // updateMixtureVelocity), so its own shift above is redundant-but-harmless there; what
  // actually needs shifting is each phase's OWN history, used by makeForcing's BDF term.
  if (!o_alpha_phases.empty()) {
    const auto n = std::max(o_coeffEXT.size(), o_coeffBDF.size());
    for (int k = 0; k < Nphase; k++) {
      for (int s = n; s > 1; s--) {
        o_U_phases[k].copyFrom(o_U_phases[k], fieldOffsetSum, (s - 1) * fieldOffsetSum, (s - 2) * fieldOffsetSum);
      }
    }
  }

  {
    const auto n = o_coeffEXTP.size();
    for (int s = n; s > 1; s--) {
      o_P.copyFrom(o_P, fieldOffset, (s - 1) * fieldOffset, (s - 2) * fieldOffset);
    }
  }
}

void fluidSolver_t::advectionSubcycling(int nEXT, double time, int phase)
{
  const auto nFields = mesh->dim;
  const auto fieldOffsetSum = nFields * fieldOffset;

  auto &o_Uphase = o_U_phases[phase];
  auto &o_JwFphase = o_JwF_phases[phase];

  static occa::kernel kernel;
  if (!kernel.isInitialized()) {
    if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
      kernel = platform->kernelRequests.load("core-subCycleStrongCubatureVolumeHex3D");
    } else {
      kernel = platform->kernelRequests.load("core-subCycleStrongVolumeHex3D");
    }
  }

  platform->linAlg->fill(o_JwFphase.size(), 0, o_JwFphase);

  advectionSubcyclingRK(mesh,
                        mesh,
                        time,
                        dt,
                        Nsubsteps,
                        o_coeffBDF,
                        nEXT,
                        nFields,
                        kernel,
                        mesh->oogs3,
                        mesh->fieldOffset,
                        fieldOffset,
                        cubatureOffset,
                        fieldOffsetSum,
                        (geom) ? geom->o_div : o_NULL,
                        o_relUrst_phases[phase],
                        o_Uphase,
                        o_JwFphase);

  if (platform->verbose()) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
                                                                 mesh->dim,
                                                                 fieldOffset,
                                                                 mesh->ogs->o_invDegree,
                                                                 o_JwFphase,
                                                                 platform->comm.mpiComm());
    if (platform->comm.mpiRank() == 0) {
      printf("%s advSub norm: %.15e\n", velocityName[phase].c_str(), debugNorm);
    }
  }
}

void registerFluidSolverKernels(occa::properties kernelInfoBC)
{
  const bool serial = platform->serial();
  const std::string extension = serial ? ".c" : ".okl";
  const int movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");
  const std::string suffix = "Hex3D";
  const std::string oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/solver/fluid/");
  std::string section = "fluidSolver_t::";

  int N, cubN;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  platform->options.getArgs("CUBATURE POLYNOMIAL DEGREE", cubN);
  const int Nq = N + 1;
  const int cubNq = cubN + 1;
  const int Np = Nq * Nq * Nq;
  const int cubNp = cubNq * cubNq * cubNq;

  occa::properties kernelInfo = platform->kernelInfo;
  kernelInfo["defines"].asObject();
  kernelInfo["includes"].asArray();
  kernelInfo["header"].asArray();
  kernelInfo["flags"].asObject();
  kernelInfo["include_paths"].asArray();

  int nBDF = 0;
  int nEXT = 0;
  platform->options.getArgs("BDF ORDER", nBDF);
  platform->options.getArgs("EXT ORDER", nEXT);

  int Nsubsteps = 0;
  platform->options.getArgs("SUBCYCLING STEPS", Nsubsteps);
  if (Nsubsteps) {
    nEXT = nBDF;
  }

  std::string fileName, kernelName;

  occa::properties meshProps = kernelInfo;
  meshProps += meshKernelProperties(N);

  {
    occa::properties prop = kernelInfo;
    prop["defines/p_MovingMesh"] = movingMesh;
    prop["defines/p_nEXT"] = nEXT;
    prop["defines/p_nBDF"] = nBDF;
    if (Nsubsteps) {
      prop["defines/p_SUBCYCLING"] = 1;
    } else {
      prop["defines/p_SUBCYCLING"] = 0;
    }

    prop["defines/p_ADVECTION"] = 0;
    if (platform->options.compareArgs("EQUATION TYPE", "NAVIERSTOKES") && !Nsubsteps) {
      prop["defines/p_ADVECTION"] = 1;
    }

    kernelName = "sumMakef";
    fileName = oklpath + kernelName + ".okl";
    platform->kernelRequests.add(section + kernelName, fileName, prop);
  }

  kernelName = "divergenceSurface" + suffix;
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, kernelInfoBC);

  kernelName = "advectMeshVelocityHex3D";
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "pressureRhs" + suffix;
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "pressureStress" + suffix;
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "pressureDirichletBC" + suffix;
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, kernelInfoBC);

  kernelName = "velocityDirichletBC" + suffix;
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, kernelInfoBC);

  kernelName = "velocityNeumannBC" + suffix;
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, kernelInfoBC);

  kernelName = "velocityRhs" + suffix;
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);

  kernelName = "pressureAddQtl";
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(section + kernelName, fileName, meshProps);
}

void fluidSolver_t::setTimeIntegrationCoeffs(int tstep)
{
  if (o_coeffEXTP.size()) {
    std::vector<dfloat> coeff(o_coeffEXTP.size());
    const auto extOrder = std::min(tstep, static_cast<int>(o_coeffEXTP.size()));
    nek::extCoeff(coeff.data(), dt, extOrder, extOrder);
    for (int i = coeff.size(); i > extOrder; i--) {
      coeff[i - 1] = 0;
    }
    o_coeffEXTP.copyFrom(coeff.data());
  }
}
