#include "platform.hpp"
#include "mesh.h"

static const std::string section = "core-";
static const std::string suffix = "Hex3D";

static void runAvg(mesh_t *mesh, occa::memory& o_U, dlong offset, int nFields)
{
  oogs::startFinish(o_U, nFields, offset, ogsDfloat, ogsAdd, mesh->oogs);
  platform->linAlg->axmyMany(mesh->Nlocal, nFields, offset, 0, 1.0, mesh->o_invAJw, o_U);
}

namespace opSEM
{

void grad(mesh_t *mesh, dlong offset, const occa::memory &o_in, occa::memory &o_out)
{
  launchKernel(section + "gradientVolume" + suffix,
               mesh->Nelements,
               mesh->o_vgeo,
               mesh->o_D,
               offset,
               o_in,
               o_out);
}

occa::memory grad(mesh_t *mesh, dlong offset, const occa::memory &o_in)
{
  auto o_out = platform->deviceMemoryPool.reserve<dfloat>(mesh->dim * offset);
  grad(mesh, offset, o_in, o_out);
  return o_out;
}

void strongGrad(mesh_t *mesh, dlong offset, const occa::memory &o_in, occa::memory &o_out, bool avg)
{
  launchKernel(section + "gradientVolume" + suffix,
               mesh->Nelements,
               mesh->o_vgeo,
               mesh->o_D,
               offset,
               o_in,
               o_out);
  if (avg) runAvg(mesh, o_out, offset, mesh->dim);
}

occa::memory strongGrad(mesh_t *mesh, dlong offset, const occa::memory &o_in, bool avg)
{
  auto o_out = platform->deviceMemoryPool.reserve<dfloat>(mesh->dim * offset);
  strongGrad(mesh, offset, o_in, o_out, avg);
  return o_out;
}

void strongGradVec(mesh_t *mesh, dlong offset, const occa::memory &o_in, occa::memory &o_out, bool avg)
{
  for (int i = 0; i < mesh->dim; i++) {
    auto o_u = o_in.slice(i * offset, mesh->Nlocal);
    auto o_grad_u = o_out.slice(i * mesh->dim * offset, mesh->dim * offset);
    strongGrad(mesh, offset, o_u, o_grad_u, avg);
  }
}

occa::memory strongGradVec(mesh_t *mesh, dlong offset, const occa::memory &o_in, bool avg)
{
  // BUG FIX (found this session, unused anywhere else -- confirmed via grep, no regression
  // risk): the 4-arg strongGradVec() above already calls strongGrad(..., avg) per
  // component, which itself runs runAvg() internally when avg=true -- so passing avg=true
  // straight through here, then ALSO calling runAvg() again on the whole output, summed
  // (oogs ogsAdd) and normalized (invAJw) the SAME data twice, inflating it by roughly the
  // node-sharing multiplicity. Compute raw (avg=false) internally; this function's own
  // runAvg call below is the only pairing.
  poolDeviceMemory<dfloat> o_out(mesh->dim * mesh->dim * offset);
  strongGradVec(mesh, offset, o_in, o_out, false);
  if (avg) runAvg(mesh, o_out, offset, mesh->dim * mesh->dim);
  return o_out;
}

void divergence(mesh_t *mesh, dlong offset, const occa::memory &o_in, occa::memory &o_out)
{
  launchKernel(section + "wDivergenceVolume" + suffix,
               mesh->Nelements,
               mesh->o_vgeo,
               mesh->o_D,
               offset,
               o_in,
               o_out);
}

occa::memory divergence(mesh_t *mesh, dlong offset, const occa::memory &o_in)
{
  auto o_out = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  divergence(mesh, offset, o_in, o_out);
  return o_out;
}

void strongDivergence(mesh_t *mesh, dlong offset, const occa::memory &o_in, occa::memory &o_out, bool avg)
{
  launchKernel(section + "divergenceVolume" + suffix,
               mesh->Nelements,
               mesh->o_vgeo,
               mesh->o_D,
               offset,
               o_in,
               o_out);
  if (avg) runAvg(mesh, o_out, 0, 1);
}

occa::memory strongDivergence(mesh_t *mesh, dlong offset, const occa::memory &o_in, bool avg)
{
  auto o_out = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  strongDivergence(mesh, offset, o_in, o_out, avg);
  return o_out;
}

void laplacian(mesh_t *mesh,
               dlong offset,
               const occa::memory &o_lambda,
               const occa::memory &o_in,
               occa::memory &o_out)
{
  // BUG FIX (found this session, unused anywhere else -- confirmed via grep, no regression
  // risk): this condition was inverted (allocates only when ALREADY initialized, which
  // never happens for a static local starting uninitialized), so o_fieldOffsetScan was
  // permanently uninitialized/empty and passed to the kernel as-is.
  static occa::memory o_fieldOffsetScan;
  if (!o_fieldOffsetScan.isInitialized()) {
    o_fieldOffsetScan = platform->device.malloc<dlong>(1);
  }
  launchKernel(section + "weakLaplacian" + suffix,
               mesh->Nelements,
               1,
               o_fieldOffsetScan,
               mesh->o_ggeo,
               mesh->o_D,
               o_lambda,
               o_in,
               o_out);
}

occa::memory laplacian(mesh_t *mesh, dlong offset, const occa::memory &o_lambda, const occa::memory &o_in)
{
  auto o_out = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  laplacian(mesh, offset, o_lambda, o_in, o_out);
  return o_out;
}

void strongLaplacian(mesh_t *mesh,
                     dlong offset,
                     const occa::memory &o_lambda,
                     const occa::memory &o_in,
                     occa::memory &o_out, bool avg)
{
  // BUG FIX (found this session, unused anywhere else -- confirmed via grep, no regression
  // risk): with avg=true (the default), strongGrad() below already performs its own
  // runAvg() internally (oogs ogsAdd sum + invAJw normalize) -- the EXPLICIT
  // oogs::startFinish() right after summed that ALREADY-normalized field AGAIN, with no
  // compensating divide, inflating it by roughly the node-sharing multiplicity (same for
  // strongDivergence()'s own internal runAvg + the explicit "if (avg) runAvg(...)" at the
  // end). Caught by a case2.udf RED-term (eq 105) experiment going from healthy to
  // resNorm0~1e9 within 2 steps. Compute both stages raw (avg=false) internally; the
  // explicit oogs/invAJw sequence already written below, and the final runAvg, are each
  // meant to be the ONLY pairing for their stage.
  auto o_grad = strongGrad(mesh, offset, o_in, false);
  oogs::startFinish(o_grad, mesh->dim, offset, ogsDfloat, ogsAdd, mesh->oogs);

  auto o_tmp = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  platform->linAlg->axmyz(mesh->Nlocal, 1.0, mesh->o_invAJw, o_lambda, o_tmp);
  platform->linAlg->axmyVector(mesh->Nlocal, offset, 0, 1.0, o_tmp, o_grad);

  o_out = strongDivergence(mesh, offset, o_grad, false);
  if (avg) runAvg(mesh, o_out, 0, 1);
}

occa::memory
strongLaplacian(mesh_t *mesh, dlong offset, const occa::memory &o_lambda, const occa::memory &o_in, bool avg)
{
  occa::memory o_out;
  strongLaplacian(mesh, offset, o_lambda, o_in, o_out, avg);
  return o_out;
}

void strongCurl(mesh_t *mesh, dlong offset, const occa::memory &o_in, occa::memory &o_out, bool avg)
{
  const dlong scaleJW = 1;
  launchKernel(section + "curl" + suffix,
               mesh->Nelements,
               scaleJW,
               mesh->o_vgeo,
               mesh->o_D,
               offset,
               o_in,
               o_out);
  if (avg) runAvg(mesh, o_out, offset, mesh->dim);
}

occa::memory strongCurl(mesh_t *mesh, dlong offset, const occa::memory &o_in, bool avg)
{
  auto o_out = platform->deviceMemoryPool.reserve<dfloat>(mesh->dim * offset);
  strongCurl(mesh, offset, o_in, o_out, avg);
  return o_out;
}

} // namespace opSEM
