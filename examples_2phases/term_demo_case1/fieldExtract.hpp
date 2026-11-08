// fieldExtract.hpp — NekRS v26 box data sampler (header-only)
//
// Interpolates solution fields onto a uniform 1D/2D/3D grid of points (via
// pointInterpolation_t / findpts) and writes the sampled grid to disk. Two IO
// backends, selected per call by the trailing `format` argument:
//   "vts"   (default) one VTK .vts StructuredGrid per call, hand-rolled MPI-IO.
//   "adios"           one ADIOS2 .bp per stem (steps appended), carrying a VTK
//                     UnstructuredGrid (.vtu) schema so ParaView opens it
//                     directly. Requires NekRS built with NEKRS_ENABLE_ADIOS.
//
// Usage (in a .udf): construct once in UDF_Setup, then call process(time,tstep)
// on each dump step. If the sampler is a global unique_ptr, release it on the
// last step (before MPI_Finalize) so the findpts MPI comm (and any ADIOS engine)
// is freed while MPI is still live:  if (nrs->lastStep) { sampler.reset(); }
//
//   // Box mode: grid spanning the diagonal x0 -> x1
//   fieldExtract(mesh, boxDims, fldList, fname, x0, x1);
//   fieldExtract(mesh, boxDims, fldList, fname, x0, x1, {"gll"}); // GLL points
//
//   // Points mode: user-supplied coordinates (one vector per axis)
//   fieldExtract(mesh, boxDims, fldList, fname, XYZ);
//
//   boxDims   : point counts {nx, ny, nz}, endpoints included (1D/2D via ny/nz=1)
//   fldList   : vector of {name, {deviceMemory<dfloat> per component}}
//   fname     : output prefix -> <fname>[_gll]NNNNN.vts  (or <fname>[_gll].bp)
//   pointDist : optional per-axis distribution, "uniform" (default) | "gll"
//               (Gauss-Lobatto-Legendre); 1 entry = all axes. Recorded in the
//               .vts FieldData tag "pointDist" (0 uniform, 1 GLL); doAvg uses the
//               matching quadrature weights and "_gll" is appended to the name.
//
// Average mode (box only): reduce the current fields over one or more axes and
// dump one .vts. Custom integer-id gather-scatter (gslib/oogs, same pattern as
// NekRS planarAvg); one handle cached per direction, freed in the destructor.
//   sampler->doAvg(time, tstep, "xy");                  // gather: collapse x,y
//   sampler->doAvg(time, tstep, "z", "gather-scatter"); // broadcast z-avg onto box
//   -> <fname>[_gll]_avg<dir>_<g|gs>NNNNN.vts   (g=gather, gs=gather-scatter)
// Integral (quadrature) average over the SAMPLING grid, not a mass-matrix
// average over the SEM mesh.
//
// Grid points follow lexicographic (nz, ny, nx) ordering and are distributed
// evenly across MPI ranks. The .vts layout is consumed by test.py / fe_read.py.
//
// Roadmap (see plans/): cylinder / Gauss-Radau layouts, derived fields (dT/dz),
// redistribution of mismatched explicit points, in-memory data() accessor.

#if !defined(nekrs_field_extract_hpp_)
#define nekrs_field_extract_hpp_

#include <map>
#include <memory>

#include "platform.hpp"
#include "mesh3D.h"
#include "pointInterpolation.hpp"

#ifdef NEKRS_ENABLE_ADIOS
#include "adios2.h"
#endif

class fieldExtract
{
public:
  using field = std::tuple<std::string, std::vector<deviceMemory<dfloat>>>;

  // Box mode: grid spanning the diagonal x0 -> x1. pointDist selects the
  // per-axis point distribution, "uniform" (default) or "gll"
  // (Gauss-Lobatto-Legendre); one entry applies to all axes, otherwise its size
  // must match boxDims. doAvg uses the matching quadrature weights.
  fieldExtract(mesh_t *mesh,
               const std::vector<int> &boxDims,
               const std::vector<field> &fldList,
               const std::string fileName,
               const std::array<dfloat, 3> &x0,
               const std::array<dfloat, 3> &x1,
               const std::vector<std::string> &pointDist = {});

  // Points mode: user-supplied coordinates (one vector per axis).
  fieldExtract(mesh_t *mesh,
               const std::vector<int> &boxDims,
               const std::vector<field> &fldList,
               const std::string fileName,
               const std::array<std::vector<dfloat>, 3> &XYZ);

  ~fieldExtract(); // frees the cached gather-scatter handles (avgGsh)

  // Interpolate + write the box. format: "vts" (default) | "adios".
  void process(const double time, const int tstep, const std::string &format = "vts");

  // Average current fields over avgDir ("x".."xyz", order-insensitive) and dump
  // one file. mode: "gather" (collapse averaged axes) | "gather-scatter"
  // (broadcast the average back onto the box). format: "vts" (default) |
  // "adios". Box mode only.
  void doAvg(const double time,
             const int tstep,
             const std::string &avgDir,
             const std::string &mode = "gather",
             const std::string &format = "vts");

  // Even point split used internally; exposed so callers building explicit XYZ
  // (points mode) can produce exactly this rank's local share of numPoints.
  static void pointDistribution(dlong numPoints, dlong &numLocal, dlong &offset);

  // Points mode only: enable doAvg on a tensor-product grid by supplying the
  // per-axis coordinates and normalized (sum=1 per axis) quadrature weights.
  // Each coord[a]/weight[a] must have length boxDims[a] (singleton axes: length
  // 1). The explicit points passed to the points-mode ctor must be the tensor
  // product coord[0] x coord[1] x coord[2] in lexicographic (nz,ny,nx) order --
  // this is the caller's responsibility (not verified). Weights are whatever
  // rule fits the axis, e.g. uniform 1/n for a periodic [0,2pi) azimuth. See
  // turbPipe_cyl.udf for a cylinder azimuthal-average example.
  void setQuadrature(const std::array<std::vector<dfloat>, 3> &coord,
                     const std::array<std::vector<dfloat>, 3> &weight);

private:
  std::unique_ptr<pointInterpolation_t> interpolator;

  std::vector<field> userFieldList;
  int Nfields;

  std::string fileName; // includes the "_gll" marker after setup
  int stepCounter;

  int boxDim = 0;
  std::vector<int> boxDims;

  std::array<dfloat, 3> point0{0, 0, 0}; // start pt (box mode; .vts tag only)
  std::array<dfloat, 3> point1{0, 0, 0}; // end pt

  dlong numPoints;      // global number of sample points
  dlong numPointsLocal; // points owned by this rank
  dlong numPointsScan;  // global offset of this rank's first point

  std::vector<dfloat> xCoord, yCoord, zCoord;
  std::vector<size_t> fldDataOffsetScan; // per-field offset into fldData

  // Box-mode per-axis discretization (built by setupAxes):
  // distCode 0 = uniform, 1 = GLL; axisWeight is normalized (sums to 1 per
  // axis) so doAvg needs no extra normalization.
  std::array<int, 3> distCode{0, 0, 0};
  std::array<std::vector<dfloat>, 3> axisCoord;
  std::array<std::vector<dfloat>, 3> axisWeight;

  std::vector<float> fldData;

  bool boxMode = false;  // true: box ctor (x0/x1 grid); false: points ctor
  bool userAxes = false; // points mode: axisCoord/axisWeight set via setQuadrature
  bool setupCalled = false;
  bool setPointsCalled = false;
  bool avgAnnounced = false; // one-line notice on the first doAvg call

  std::map<std::string, int> avgCounter; // per-stream dump counter for doAvg

  // Cached gather-scatter machinery for one doAvg configuration: local points
  // pre-reduce into per-cell "slot" partial sums (pointSlot maps each local
  // point to its slot), the oogs exchange runs over the slots plus this rank's
  // share of the reduced grid; gather-scatter reads back through pointSlot,
  // gather reads the reduced-grid tail.
  struct avgGs_t {
    oogs_t *gsh = nullptr;
    dlong nSlots = 0;             // locally-touched reduced cells
    dlong nTotal = 0;             // nSlots + rLocal (gs vector length)
    dlong rLocal = 0;             // this rank's share of the reduced grid
    dlong rScan = 0;              // global offset of that share
    std::vector<dlong> pointSlot; // local box point -> slot
  };

  std::map<std::string, avgGs_t> avgGsh; // one entry per avg configuration

  void getIxyz(const dlong i, const int nx, const int nxy, int &ix, int &iy, int &iz);

  // <stem>NNNNN<ext> with a 5-digit zero-padded counter (ext defaults ".vts").
  std::string outName(const std::string &stem, const int step, const std::string &ext = ".vts") const;

  // Build (or fetch the cached) gslib/oogs machinery for one avg direction.
  avgGs_t &setupAvgGs(const std::array<bool, 3> &avg, const std::string &dirTag);
  void setBoxDims(const std::vector<int> &boxDims_in);

  void setupCommon(const std::vector<int> &boxDims_in,
                   const std::vector<field> &fldList,
                   const std::string &fileName_in);
  void finishSetup(mesh_t *mesh);

  // Fill axisCoord / axisWeight / distCode for box mode (see plans/D.md).
  void setupAxes(const std::array<dfloat, 3> &x0,
                 const std::array<dfloat, 3> &x1,
                 const std::vector<std::string> &pointDist);

  void setPoints(const std::array<dfloat, 3> &x0, const std::array<dfloat, 3> &x1);
  void setPoints(const std::array<std::vector<dfloat>, 3> &XYZ);

  void interpolate();

  // A grid + its interpolated data to emit as one .vts (full box or averaged).
  struct GridView {
    std::array<int, 3> dims; // nx, ny, nz (1 for collapsed axes)
    std::array<int, 3> dist; // per-axis distCode (0 uniform, 1 GLL)
    dlong nGlobal;
    dlong nLocal;
    dlong nScan;
    const std::vector<dfloat> &x;
    const std::vector<dfloat> &y;
    const std::vector<dfloat> &z;
    const std::vector<float> &data;
    const std::vector<size_t> &offsetScan;
    std::array<dfloat, 3> p0;
    std::array<dfloat, 3> p1;
  };

  void writeVts(const GridView &v, const std::string &fname, const double time, const int tstep);

  // Backend dispatch: pick the writer + extension by `format` ("vts"|"adios").
  // outStep is the per-stem dump counter (drives the .vts name / .bp Write vs
  // Append); tstep is the simulation step recorded as CYCLE metadata.
  void writeGrid(const GridView &v,
                 const std::string &stem,
                 const int outStep,
                 const double time,
                 const int tstep,
                 const std::string &format);

#ifdef NEKRS_ENABLE_ADIOS
  // ADIOS2 .bp writer: one file per stem, VTK UnstructuredGrid (.vtu) schema.
  void writeAdios(const GridView &v, const std::string &stem, const double time, const int tstep);

  // Local hex/quad/line connectivity for the box lattice (VTK order); returns
  // the cells fully owned by this rank (see writeAdios for the seam note).
  void
  generateBoxConnectivity(const GridView &v, std::vector<uint64_t> &conn, dlong &nCells, uint32_t &vtkType);

  // One ADIOS stream per output stem (kept open, steps appended).
  struct adiosStream_t {
    std::unique_ptr<adios2::ADIOS> adios;
    adios2::IO io;
    adios2::Engine engine;
    int step = 0;
  };

  std::map<std::string, adiosStream_t> adiosStreams;
#endif
};

inline void fieldExtract::getIxyz(const dlong i, const int nx, const int nxy, int &ix, int &iy, int &iz)
{
  iz = i / nxy;
  int rem = i % nxy;
  iy = rem / nx;
  ix = rem % nx;
}

inline void fieldExtract::setBoxDims(const std::vector<int> &boxDims_in)
{
  boxDims = boxDims_in;
  boxDim = boxDims_in.size();

  nekrsCheck(boxDim < 1 || boxDim > 3,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "invalid dimensions of the box %d! Support 1d/2d/3d\n",
             boxDim);

  numPoints = 1;
  for (int npt : boxDims) {
    numPoints *= npt;
  }
  nekrsCheck(numPoints < 1, MPI_COMM_SELF, EXIT_FAILURE, "invalid number of points: %d!\n", numPoints);

  pointDistribution(numPoints, numPointsLocal, numPointsScan);
}

inline void fieldExtract::pointDistribution(dlong numPoints, dlong &numLocal, dlong &offset)
{
  const int rank = platform->comm.mpiRank();
  const int Nrank = platform->comm.mpiCommSize();
  const dlong base = numPoints / Nrank; // first evenly distribute points
  const dlong left = numPoints % Nrank; // assign remaining points
  numLocal = (rank < left) ? base + 1 : base;
  offset = (rank > 0) ? rank * base + std::min(left, static_cast<dlong>(rank)) : 0;
}

inline void fieldExtract::setupAxes(const std::array<dfloat, 3> &x0,
                                    const std::array<dfloat, 3> &x1,
                                    const std::vector<std::string> &pointDist)
{
  nekrsCheck(pointDist.size() > 1 && static_cast<int>(pointDist.size()) != boxDim,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "pointDist has %zu entries; use 1 (all axes) or %d (per axis)\n",
             pointDist.size(),
             boxDim);

  for (int a = 0; a < 3; a++) {
    const int n = (a < boxDim) ? boxDims[a] : 1;

    int code = 0; // 0 = uniform
    if (!pointDist.empty() && a < boxDim) {
      const auto s = upperCase(pointDist.size() == 1 ? pointDist[0] : pointDist[a]);
      code = (s == "UNIFORM") ? 0 : (s == "GLL") ? 1 : -1;
      nekrsCheck(code < 0,
                 MPI_COMM_SELF,
                 EXIT_FAILURE,
                 "invalid pointDist '%s' (use uniform|gll)\n",
                 s.c_str());
    }
    if (n < 2) {
      code = 0; // singleton axes are trivially uniform
    }
    distCode[a] = code;

    // normalized weights (sum to 1) so doAvg needs no extra normalization
    auto &coord = axisCoord[a];
    auto &wght = axisWeight[a];
    coord.resize(n);
    wght.resize(n);
    if (n == 1) {
      coord[0] = x0[a];
      wght[0] = 1.0;
    } else if (code == 0) { // uniform + trapezoid weights
      const dfloat dx = (x1[a] - x0[a]) / (n - 1);
      for (int i = 0; i < n; i++) {
        coord[i] = x0[a] + i * dx;
        wght[i] = ((i == 0 || i == n - 1) ? 0.5 : 1.0) / (n - 1);
      }
    } else { // GLL nodes + quadrature weights, mapped from [-1,1] to [x0,x1]
      std::vector<dfloat> z(n), w(n);
      JacobiGLL(n - 1, z.data(), w.data());
      for (int i = 0; i < n; i++) {
        coord[i] = x0[a] + 0.5 * (z[i] + 1.0) * (x1[a] - x0[a]);
        wght[i] = 0.5 * w[i]; // sum(w) = 2 on [-1,1]
      }
    }
  }
}

inline void fieldExtract::setPoints(const std::array<dfloat, 3> &x0, const std::array<dfloat, 3> &x1)
{
  // grid from x0 to x1 following the per-axis coordinates from setupAxes
  if (numPointsLocal) {
    const int nx = boxDims[0];
    const int ny = (boxDim > 1) ? boxDims[1] : 1;
    const int nxy = nx * ny;

    xCoord.resize(numPointsLocal);
    yCoord.resize(numPointsLocal);
    zCoord.resize(numPointsLocal);

    for (dlong i = 0; i < numPointsLocal; i++) {
      const dlong ig = i + numPointsScan;
      int ix, iy, iz;
      getIxyz(ig, nx, nxy, ix, iy, iz);
      xCoord[i] = axisCoord[0][ix];
      yCoord[i] = axisCoord[1][iy];
      zCoord[i] = axisCoord[2][iz];
    }
  }

  setPointsCalled = true;
}

inline void fieldExtract::setPoints(const std::array<std::vector<dfloat>, 3> &XYZ)
{
  const dlong Nlocal = XYZ[0].size();

  { // chk XYZ: all three axes must have matching length
    int err = 0;
    for (int d = 1; d < 3; d++) {
      if (XYZ[d].size() != XYZ[0].size()) {
        err = d;
        break;
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_INT, MPI_MAX, platform->comm.mpiComm()); // for err-print

    nekrsCheck(err != 0, MPI_COMM_SELF, EXIT_FAILURE, "XYZ[%d] has different length as XYZ[0]!\n", err);
  }

  {                                      // chk XYZ vs boxDims
    nekrsCheck(Nlocal != numPointsLocal, // TODO(later): redistribute points
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "input points don't match the boxDims! Nlocal=%d  numPointsLocal=%d\n",
               Nlocal,
               numPointsLocal);

    int Ntotal = Nlocal;
    MPI_Allreduce(MPI_IN_PLACE, &Ntotal, 1, MPI_INT, MPI_SUM, platform->comm.mpiComm());

    nekrsCheck(Ntotal != numPoints,
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "input points don't match the boxDims! Ntotal=%d  numPoints=%d\n",
               Ntotal,
               numPoints);
  }

  xCoord.resize(numPointsLocal);
  yCoord.resize(numPointsLocal);
  zCoord.resize(numPointsLocal);

  for (dlong i = 0; i < numPointsLocal; ++i) {
    xCoord[i] = XYZ[0][i];
    yCoord[i] = XYZ[1][i];
    zCoord[i] = XYZ[2][i];
  }

  setPointsCalled = true;
}

inline void fieldExtract::setupCommon(const std::vector<int> &boxDims_in,
                                      const std::vector<field> &fldList,
                                      const std::string &fileName_in)
{
  static bool statAdded = false;
  if (!statAdded) {
    platform->timer.addUserStat("fieldExtract::");
    statAdded = true;
  }

  nekrsCheck(setupCalled, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "Attempt to call setup twice!\n");

  setBoxDims(boxDims_in);

  userFieldList = fldList;
  Nfields = userFieldList.size();

  // Each field's components must share a common length.
  int ifld = 1;
  for (auto &entry : userFieldList) {
    const auto &o_fld = std::get<1>(entry);
    const auto dim_fld = o_fld.size();
    const auto len_fld = o_fld[0].size();

    int err = 0;
    for (size_t idim = 1; idim < dim_fld; idim++) {
      if (o_fld[idim].size() != len_fld) {
        err = idim;
        break;
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_INT, MPI_MAX, platform->comm.mpiComm()); // for err-print
    nekrsCheck(err != 0,
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "userFieldList[%d][%d] has different length than its 0th array!\n",
               ifld,
               err);
    ifld++;
  }

  fileName = fileName_in;
  stepCounter = 0;
}

inline std::string
fieldExtract::outName(const std::string &stem, const int step, const std::string &ext) const
{
  std::ostringstream s;
  s << stem << std::setw(5) << std::setfill('0') << step << ext;
  return s.str();
}

inline void fieldExtract::finishSetup(mesh_t *mesh)
{
  nekrsCheck(!setPointsCalled, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "points are not setup properly!");

  const double tStart = MPI_Wtime();

  // mark GLL outputs in every filename derived from fileName
  if (distCode[0] || distCode[1] || distCode[2]) {
    fileName += "_gll";
  }

  // fldData layout: one contiguous block per field, component-major (d) then point (i).
  fldDataOffsetScan.resize(Nfields + 1);
  fldDataOffsetScan[0] = 0;
  int ifld = 1;
  for (auto &entry : userFieldList) {
    const auto dim_fld = std::get<1>(entry).size();
    fldDataOffsetScan[ifld] = fldDataOffsetScan[ifld - 1] + numPointsLocal * dim_fld;
    ifld++;
  }
  fldData.resize(fldDataOffsetScan[Nfields]);

  interpolator = std::make_unique<pointInterpolation_t>(mesh, platform->comm.mpiComm());
  interpolator->setPoints(xCoord, yCoord, zCoord);
  interpolator->find();

  setupCalled = true;

  platform->timer.set("fieldExtract::setup", MPI_Wtime() - tStart);

  // one-line summary: global count + how evenly points split across ranks
  dlong lmin = numPointsLocal, lmax = numPointsLocal, lsum = numPointsLocal;
  MPI_Allreduce(MPI_IN_PLACE, &lmin, 1, MPI_DLONG, MPI_MIN, platform->comm.mpiComm());
  MPI_Allreduce(MPI_IN_PLACE, &lmax, 1, MPI_DLONG, MPI_MAX, platform->comm.mpiComm());
  MPI_Allreduce(MPI_IN_PLACE, &lsum, 1, MPI_DLONG, MPI_SUM, platform->comm.mpiComm());
  if (platform->comm.mpiRank() == 0) {
    printf("fieldExtract(%s): nPoints=%lld  local(min/max/avg)=%lld/%lld/%.1f\n",
           fileName.c_str(),
           (long long)numPoints,
           (long long)lmin,
           (long long)lmax,
           (double)lsum / platform->comm.mpiCommSize());
    fflush(stdout);
  }
}

inline fieldExtract::fieldExtract(mesh_t *mesh,
                                  const std::vector<int> &boxDims_in,
                                  const std::vector<field> &fldList,
                                  const std::string fileName_in,
                                  const std::array<dfloat, 3> &x0,
                                  const std::array<dfloat, 3> &x1,
                                  const std::vector<std::string> &pointDist)
{
  boxMode = true;
  setupCommon(boxDims_in, fldList, fileName_in);
  point0 = x0;
  point1 = x1;
  setupAxes(x0, x1, pointDist);
  setPoints(x0, x1);
  finishSetup(mesh);
}

inline fieldExtract::fieldExtract(mesh_t *mesh,
                                  const std::vector<int> &boxDims_in,
                                  const std::vector<field> &fldList,
                                  const std::string fileName_in,
                                  const std::array<std::vector<dfloat>, 3> &XYZ)
{
  boxMode = false;
  setupCommon(boxDims_in, fldList, fileName_in);
  // point0/point1 stay {0,0,0}; they only feed the .vts x0/x1 tags.
  setPoints(XYZ);
  finishSetup(mesh);
}

inline void fieldExtract::setQuadrature(const std::array<std::vector<dfloat>, 3> &coord,
                                        const std::array<std::vector<dfloat>, 3> &weight)
{
  nekrsCheck(boxMode,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "setQuadrature is for points mode; box mode already builds its own axes!");

  for (int a = 0; a < 3; a++) {
    const int n = (a < boxDim) ? boxDims[a] : 1;
    nekrsCheck(static_cast<int>(coord[a].size()) != n || static_cast<int>(weight[a].size()) != n,
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "setQuadrature axis %d: coord/weight length must equal boxDims=%d\n",
               a,
               n);
  }

  axisCoord = coord;
  axisWeight = weight;
  userAxes = true;
}

inline fieldExtract::~fieldExtract()
{
  // Free every cached doAvg handle (one per avg configuration). Do NOT use
  // oogs::destroy(): it releases the array-new'd oogs_t with free(), which
  // glibc rejects ("free(): invalid pointer" -- the new[] array cookie offsets
  // the pointer), and it does not free the underlying ogs_t either. Instead
  // free the ogs_t via ogsFree and delete[] the handle; the occa::memory
  // members release their buffers in their destructors.
  for (auto &entry : avgGsh) {
    if (entry.second.gsh) {
      ogsFree(entry.second.gsh->ogs);
      delete[] entry.second.gsh;
      entry.second.gsh = nullptr;
    }
  }
  avgGsh.clear();

#ifdef NEKRS_ENABLE_ADIOS
  // Close every ADIOS engine before MPI_Finalize (same reason the findpts comm
  // must be released early). The ADIOS object is destroyed with the map entry.
  for (auto &entry : adiosStreams) {
    if (entry.second.engine) {
      entry.second.engine.Close();
    }
  }
  adiosStreams.clear();
#endif
}

inline fieldExtract::avgGs_t &fieldExtract::setupAvgGs(const std::array<bool, 3> &avg,
                                                       const std::string &dirTag)
{
  auto it = avgGsh.find(dirTag);
  if (it != avgGsh.end()) {
    return it->second;
  }

  const int nx = boxDims[0];
  const int ny = (boxDim > 1) ? boxDims[1] : 1;
  const int nz = (boxDim > 2) ? boxDims[2] : 1;
  const int nxy = nx * ny;

  // reduced grid: averaged axes collapse to 1
  const int rn[3] = {avg[0] ? 1 : nx, avg[1] ? 1 : ny, avg[2] ? 1 : nz};
  const dlong rN = static_cast<dlong>(rn[0]) * rn[1] * rn[2];
  const int rnxy = rn[0] * rn[1];

  avgGs_t ags;
  pointDistribution(rN, ags.rLocal, ags.rScan);

  // Local pre-reduction slots: one per reduced cell touched by this rank's box
  // points. The gs then runs over partial sums (<= 2 local entries per cell:
  // the slot and, if owned here, the reduced-grid tail entry), which keeps the
  // local rowsize tiny no matter how many points collapse into a cell -- a
  // direct point-level gs would exceed ogsSetup's per-row device blocksize
  // limit (e.g. avgDir "xyz" puts the whole box into one row).
  std::map<dlong, dlong> cellSlot; // reduced cell -> slot
  ags.pointSlot.resize(numPointsLocal);
  for (dlong i = 0; i < numPointsLocal; i++) {
    const dlong ig = i + numPointsScan;
    int ix, iy, iz;
    getIxyz(ig, nx, nxy, ix, iy, iz);
    const int rx = avg[0] ? 0 : ix, ry = avg[1] ? 0 : iy, rz = avg[2] ? 0 : iz;
    const dlong rcell = static_cast<dlong>(rz) * rnxy + static_cast<dlong>(ry) * rn[0] + rx;
    const auto ins = cellSlot.emplace(rcell, static_cast<dlong>(cellSlot.size()));
    ags.pointSlot[i] = ins.first->second;
  }
  ags.nSlots = cellSlot.size();
  ags.nTotal = ags.nSlots + ags.rLocal;

  // ids (1-based): slots carry their reduced-cell id; the tail entries carry
  // the id of the reduced point this rank owns (they contribute zero to the
  // ogsAdd sum and receive it back -- that is the gather-mode output)
  std::vector<long long int> ids(ags.nTotal);
  for (const auto &cs : cellSlot) {
    ids[cs.second] = 1 + static_cast<long long int>(cs.first);
  }
  for (dlong i = 0; i < ags.rLocal; i++) {
    ids[ags.nSlots + i] = 1 + static_cast<long long int>(ags.rScan) + i;
  }

  int nComp = 0; // total number of scalar components over all fields
  for (auto &entry : userFieldList) {
    nComp += std::get<1>(entry).size();
  }

  auto ogsh = ogsSetup(ags.nTotal, ids.data(), platform->comm.mpiComm(), 0, platform->device.occaDevice());
  ags.gsh =
      oogs::setup(ogsh, nComp, std::max(ags.nTotal, static_cast<dlong>(1)), ogsDfloat, nullptr, OOGS_AUTO);

  return avgGsh.emplace(dirTag, std::move(ags)).first->second;
}

inline void fieldExtract::interpolate()
{
  platform->timer.tic("fieldExtract::interpolate");
  auto o_wrk = platform->deviceMemoryPool.reserve<dfloat>(numPointsLocal);
  std::vector<dfloat> wrk(numPointsLocal);

  int ifld = 0;
  for (const auto &entry : userFieldList) {
    const auto &fieldVector = std::get<1>(entry);
    const auto fieldDim = fieldVector.size();

    for (int d = 0; d < fieldDim; d++) {
      interpolator->eval(1, 0, fieldVector[d], numPointsLocal, o_wrk);
      o_wrk.copyTo(wrk.data(), numPointsLocal);

      for (dlong i = 0; i < numPointsLocal; i++) {
        fldData[fldDataOffsetScan[ifld] + d * numPointsLocal + i] = static_cast<float>(wrk[i]);
      }
    }
    ifld++;
  }
  platform->timer.toc("fieldExtract::interpolate");
}

inline void
fieldExtract::writeVts(const GridView &v, const std::string &fname, const double time, const int tstep)
{
  platform->timer.tic("fieldExtract::io");

  MPI_Comm mpi_comm = platform->comm.mpiComm();
  const int rank = platform->comm.mpiRank();

  const long long int pOffset = v.nScan; // this rank's first point, global ordering

  // truncate cleanly before MPI-IO opens it
  if (rank == 0) {
    std::ofstream file(fname, std::ios::binary | std::ios::trunc);
    file.close();
  }
  MPI_Barrier(mpi_comm);

  MPI_File file_out;
  MPI_File_open(mpi_comm, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file_out);

  auto indent = [](const int n) { return std::string(2 * n, ' '); };

  auto fieldDataArrayInt3 = [](const std::string &name, const int a, const int b, const int c) {
    return "<DataArray type=\"Int32\" Name=\"" + name +
           "\" NumberOfComponents=\"3\" NumberOfTuples=\"1\" format=\"ascii\"> " + std::to_string(a) + " " +
           std::to_string(b) + " " + std::to_string(c) + " </DataArray>\n";
  };

  auto fieldDataArrayFloat3 = [](const std::string &name, const double a, const double b, const double c) {
    return "<DataArray type=\"Float64\" Name=\"" + name +
           "\" NumberOfComponents=\"3\" NumberOfTuples=\"1\" format=\"ascii\"> " + std::to_string(a) + " " +
           std::to_string(b) + " " + std::to_string(c) + " </DataArray>\n";
  };

  auto fieldDataArrayInt64 = [](const std::string &name, const long long v) {
    return "<DataArray type=\"Int64\" Name=\"" + name + "\" NumberOfTuples=\"1\" format=\"ascii\"> " +
           std::to_string(v) + " </DataArray>\n";
  };

  auto vtsDataArray = [](const std::string &fieldName, long long int nComponent, long long int offset) {
    return "<DataArray type=\"Float32\" Name=\"" + fieldName + "\" NumberOfComponents=\"" +
           std::to_string(nComponent) + "\" format=\"appended\" offset=\"" + std::to_string(offset) +
           "\"/>\n";
  };

  const int nx = v.dims[0];
  const int ny = v.dims[1];
  const int nz = v.dims[2];
  const std::string vtsExtent = "0 " + std::to_string(nx - 1) + " " + "0 " + std::to_string(ny - 1) + " " +
                                "0 " + std::to_string(nz - 1);

  constexpr long long int dim = 3;
  constexpr long long int headerBytes = sizeof(unsigned long long int);

  // appended-data offsets (relative to the byte after "_"); each block is
  // [UInt64 nbytes][raw float data]
  long long int vtsOffset = 0;

  const long long int posVtsOffset = vtsOffset;
  vtsOffset += headerBytes + dim * v.nGlobal * sizeof(float);

  std::vector<long long int> fieldVtsOffsets;
  fieldVtsOffsets.reserve(Nfields);

  for (const auto &entry : userFieldList) {
    const auto &fieldVector = std::get<1>(entry);
    const int fieldDim = fieldVector.size();

    fieldVtsOffsets.push_back(vtsOffset);
    vtsOffset += headerBytes + fieldDim * v.nGlobal * sizeof(float);
  }

  const long long int totalAppendedBytes = vtsOffset;

  // XML header
  std::string message;
  message += indent(0) + "<VTKFile type=\"StructuredGrid\" version=\"1.0\" "
                         "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
  message += indent(1) + "<StructuredGrid WholeExtent=\"" + vtsExtent + "\">\n";

  message += indent(2) + "<FieldData>\n";
  message += indent(3) +
             "<DataArray type=\"Float32\" Name=\"TimeValue\" "
             "NumberOfTuples=\"1\" format=\"ascii\"> " +
             std::to_string(time) + " </DataArray>\n";
  message += indent(3) +
             "<DataArray type=\"Int32\" Name=\"CYCLE\" "
             "NumberOfTuples=\"1\" format=\"ascii\"> " +
             std::to_string(tstep) + " </DataArray>\n";
  message += indent(3) + fieldDataArrayInt3("boxDims", nx, ny, nz);
  message += indent(3) + fieldDataArrayInt3("pointDist", v.dist[0], v.dist[1], v.dist[2]);
  message += indent(3) + fieldDataArrayInt64("numPoints", static_cast<long long>(v.nGlobal));
  message += indent(3) + fieldDataArrayFloat3("x0", v.p0[0], v.p0[1], v.p0[2]);
  message += indent(3) + fieldDataArrayFloat3("x1", v.p1[0], v.p1[1], v.p1[2]);
  message += indent(2) + "</FieldData>\n";

  message += indent(2) + "<Piece Extent=\"" + vtsExtent + "\">\n";
  message += indent(3) + "<Points>\n";
  message += indent(4) + vtsDataArray("Position", dim, posVtsOffset);
  message += indent(3) + "</Points>\n";
  message += indent(3) + "<PointData>\n";

  int ifld = 0;
  for (const auto &entry : userFieldList) {
    const std::string fieldName = std::get<0>(entry);
    const auto &fieldVector = std::get<1>(entry);
    const int fieldDim = fieldVector.size();
    message += indent(4) + vtsDataArray(fieldName, fieldDim, fieldVtsOffsets[ifld]);
    ifld++;
  }

  message += indent(3) + "</PointData>\n";
  message += indent(2) + "</Piece>\n";
  message += indent(1) + "</StructuredGrid>\n";
  message += indent(1) + "<AppendedData encoding=\"raw\">\n";
  message += "_";

  const MPI_Offset appendedStart = static_cast<MPI_Offset>(message.size());

  if (rank == 0) {
    MPI_File_write_at(file_out, 0, message.c_str(), message.size(), MPI_CHAR, MPI_STATUS_IGNORE);
  }

  MPI_Barrier(mpi_comm);

  // write one appended VTK block (vtkOffset is relative to appendedStart, i.e. after "_")
  auto writeField = [&](const MPI_Offset vtkOffset, const int nComponents, const std::vector<float> &field) {
    const unsigned long long int nbytes = static_cast<unsigned long long int>(nComponents) *
                                          static_cast<unsigned long long int>(v.nGlobal) * sizeof(float);

    const MPI_Offset blockStart = appendedStart + vtkOffset;
    const MPI_Offset dataStart = blockStart + static_cast<MPI_Offset>(sizeof(unsigned long long int));

    const MPI_Offset localDataStart = dataStart + static_cast<MPI_Offset>(sizeof(float)) *
                                                      static_cast<MPI_Offset>(nComponents) *
                                                      static_cast<MPI_Offset>(pOffset);

    if (rank == 0) {
      MPI_File_write_at(file_out, blockStart, &nbytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(mpi_comm);

    MPI_File_write_at_all(file_out, localDataStart, field.data(), field.size(), MPI_FLOAT, MPI_STATUS_IGNORE);
  };

  // Write coordinates.
  {
    std::vector<float> positions(dim * v.nLocal, 0.0f);

    for (dlong i = 0; i < v.nLocal; i++) {
      positions[dim * i + 0] = static_cast<float>(v.x[i]);
      positions[dim * i + 1] = static_cast<float>(v.y[i]);
      positions[dim * i + 2] = static_cast<float>(v.z[i]);
    }

    writeField(posVtsOffset, dim, positions);
  }

  // Write one appended block per VTK DataArray.
  ifld = 0;

  for (const auto &entry : userFieldList) {
    const auto &fieldVector = std::get<1>(entry);
    const int fieldDim = fieldVector.size();

    std::vector<float> oneField(fieldDim * v.nLocal);

    for (int d = 0; d < fieldDim; d++) {
      for (dlong i = 0; i < v.nLocal; i++) {
        oneField[d * v.nLocal + i] = v.data[v.offsetScan[ifld] + d * v.nLocal + i];
      }
    }

    writeField(fieldVtsOffsets[ifld], fieldDim, oneField);

    ifld++;
  }

  // XML footer at the exact end of appended data
  const std::string footer = "\n  </AppendedData>\n"
                             "</VTKFile>\n";

  const MPI_Offset footerOffset = appendedStart + static_cast<MPI_Offset>(totalAppendedBytes);

  if (rank == 0) {
    MPI_File_write_at(file_out, footerOffset, footer.c_str(), footer.size(), MPI_CHAR, MPI_STATUS_IGNORE);
  }

  MPI_Barrier(mpi_comm);

  MPI_File_close(&file_out);

  platform->timer.toc("fieldExtract::io");
}

inline void fieldExtract::writeGrid(const GridView &v,
                                    const std::string &stem,
                                    const int outStep,
                                    const double time,
                                    const int tstep,
                                    const std::string &format)
{
  if (format == "adios") {
#ifdef NEKRS_ENABLE_ADIOS
    writeAdios(v, stem, time, tstep);
#else
    nekrsAbort(platform->comm.mpiComm(),
               EXIT_FAILURE,
               "%s\n",
               "fieldExtract: format 'adios' requested but NEKRS_ENABLE_ADIOS is not set!");
#endif
  } else if (format == "vts") {
    writeVts(v, outName(stem, outStep, ".vts"), time, tstep);
  } else {
    nekrsAbort(platform->comm.mpiComm(),
               EXIT_FAILURE,
               "invalid fieldExtract format '%s' (use vts|adios)\n",
               format.c_str());
  }
}

#ifdef NEKRS_ENABLE_ADIOS
inline void fieldExtract::generateBoxConnectivity(const GridView &v,
                                                  std::vector<uint64_t> &conn,
                                                  dlong &nCells,
                                                  uint32_t &vtkType)
{
  // VTK cell types
  constexpr uint32_t VTK_LINE = 3;
  constexpr uint32_t VTK_QUAD = 9;
  constexpr uint32_t VTK_HEXAHEDRON = 12;

  const int nx = v.dims[0], ny = v.dims[1], nz = v.dims[2];
  const int dimBox = (nz > 1) ? 3 : (ny > 1) ? 2 : 1;

  // ADIOS .vtu uses one local-array block per rank, so cells reference LOCAL
  // point ids only. Points are split lexicographically (nz,ny,nx) across ranks,
  // so a cell whose corners straddle a rank boundary cannot be emitted by either
  // rank; that thin seam between slabs is simply not drawn (all point/field data
  // is still written). On a single rank the whole box is emitted.
  int lx0, ly0, lz0, lx1, ly1, lz1; // local (ix,iy,iz) range covered by this rank
  {
    int ax, ay, az, bx, by, bz;
    getIxyz(v.nScan, nx, nx * ny, ax, ay, az);                // first local point
    getIxyz(v.nScan + v.nLocal - 1, nx, nx * ny, bx, by, bz); // last local point
    // A rank's local points are a contiguous global range; the safe cell region
    // is the box of whole (ix,iy,iz) fully inside [first,last]. For the common
    // single-rank case this is the full box.
    lx0 = 0;
    ly0 = 0;
    lz0 = az;
    lx1 = nx - 1;
    ly1 = ny - 1;
    lz1 = bz;
    // if this rank starts/ends mid-plane, keep only whole z-planes it fully owns
    if (ax != 0) {
      lz0 = az + 1;
    }
    if (bx != nx - 1 || by != ny - 1) {
      lz1 = bz - 1;
    }
  }

  auto lid = [&](int ix, int iy, int iz) -> uint64_t {
    // local id = global id - nScan, within this rank's contiguous range
    const dlong g = static_cast<dlong>(iz) * (nx * ny) + static_cast<dlong>(iy) * nx + ix;
    return static_cast<uint64_t>(g - v.nScan);
  };

  conn.clear();
  nCells = 0;

  if (dimBox == 3) {
    vtkType = VTK_HEXAHEDRON;
    for (int iz = lz0; iz < lz1; iz++) {
      for (int iy = 0; iy < ny - 1; iy++) {
        for (int ix = 0; ix < nx - 1; ix++) {
          conn.push_back(8);
          conn.push_back(lid(ix, iy, iz));
          conn.push_back(lid(ix + 1, iy, iz));
          conn.push_back(lid(ix + 1, iy + 1, iz));
          conn.push_back(lid(ix, iy + 1, iz));
          conn.push_back(lid(ix, iy, iz + 1));
          conn.push_back(lid(ix + 1, iy, iz + 1));
          conn.push_back(lid(ix + 1, iy + 1, iz + 1));
          conn.push_back(lid(ix, iy + 1, iz + 1));
          nCells++;
        }
      }
    }
  } else if (dimBox == 2) {
    vtkType = VTK_QUAD;
    for (int iy = 0; iy < ny - 1; iy++) {
      for (int ix = 0; ix < nx - 1; ix++) {
        conn.push_back(4);
        conn.push_back(lid(ix, iy, 0));
        conn.push_back(lid(ix + 1, iy, 0));
        conn.push_back(lid(ix + 1, iy + 1, 0));
        conn.push_back(lid(ix, iy + 1, 0));
        nCells++;
      }
    }
  } else {
    vtkType = VTK_LINE;
    for (int ix = 0; ix < nx - 1; ix++) {
      conn.push_back(2);
      conn.push_back(lid(ix, 0, 0));
      conn.push_back(lid(ix + 1, 0, 0));
      nCells++;
    }
  }
}

inline void
fieldExtract::writeAdios(const GridView &v, const std::string &stem, const double time, const int tstep)
{
  platform->timer.tic("fieldExtract::io");

  MPI_Comm mpi_comm = platform->comm.mpiComm();

  // Open (once per stem) an ADIOS stream; steps are appended into one .bp.
  auto it = adiosStreams.find(stem);
  if (it == adiosStreams.end()) {
    adiosStream_t s;
    s.adios = std::make_unique<adios2::ADIOS>(mpi_comm);
    s.io = s.adios->DeclareIO("fieldExtract::" + stem);
    s.engine = s.io.Open(stem + ".bp", adios2::Mode::Write); // one <stem>.bp, steps appended
    s.step = 0;
    it = adiosStreams.emplace(stem, std::move(s)).first;
  }
  adiosStream_t &S = it->second;

  const int nx = v.dims[0], ny = v.dims[1], nz = v.dims[2];
  const size_t nLocalPts = static_cast<size_t>(v.nLocal);

  // repack coordinates: interleaved x/y/z per point (VTK "vertices")
  std::vector<float> vertices(3 * nLocalPts);
  for (dlong i = 0; i < v.nLocal; i++) {
    vertices[3 * i + 0] = static_cast<float>(v.x[i]);
    vertices[3 * i + 1] = static_cast<float>(v.y[i]);
    vertices[3 * i + 2] = static_cast<float>(v.z[i]);
  }

  // local connectivity + cell type
  std::vector<uint64_t> conn;
  dlong nCells = 0;
  uint32_t vtkType = 0;
  generateBoxConnectivity(v, conn, nCells, vtkType);
  const size_t vertsPerCell = (conn.empty()) ? 0 : (conn[0] + 1); // count word + ids

  S.engine.BeginStep();

  // ADIOS "local array" (one block per rank): global shape {} / start {} /
  // count = shape. Shape is fixed across steps for a given stem, so defining it
  // once is enough — InquireVariable returns it on later steps.
  auto putLocalArray = [&](const std::string &name, const auto *data, adios2::Dims shape) {
    using T = std::decay_t<decltype(*data)>;
    auto var = S.io.InquireVariable<T>(name) ? S.io.InquireVariable<T>(name)
                                             : S.io.DefineVariable<T>(name, {}, {}, shape);
    S.engine.Put(var, data, adios2::Mode::Sync);
  };

  // mesh (static for a fixed box, but cheap; write every step so append/restart
  // stays self-contained per step, mirroring how iofldAdios re-emits on step 0)
  if (S.step == 0) {
    putLocalArray("vertices", vertices.data(), adios2::Dims{nLocalPts, 3});
    if (nCells > 0) {
      putLocalArray("connectivity", conn.data(), adios2::Dims{static_cast<size_t>(nCells), vertsPerCell});
    }
    // cell type as a scalar per rank
    {
      auto var = S.io.InquireVariable<uint32_t>("types") ? S.io.InquireVariable<uint32_t>("types")
                                                         : S.io.DefineVariable<uint32_t>("types");
      S.engine.Put(var, vtkType, adios2::Mode::Sync);
    }
    // metadata attributes (mirror the .vts FieldData tags)
    S.io.DefineAttribute<int32_t>("boxDims", std::vector<int32_t>{nx, ny, nz}.data(), 3);
    S.io.DefineAttribute<int32_t>("pointDist",
                                  std::vector<int32_t>{v.dist[0], v.dist[1], v.dist[2]}.data(),
                                  3);
    S.io.DefineAttribute<int64_t>("numPoints", static_cast<int64_t>(v.nGlobal));
    S.io.DefineAttribute<double>("x0", std::vector<double>{v.p0[0], v.p0[1], v.p0[2]}.data(), 3);
    S.io.DefineAttribute<double>("x1", std::vector<double>{v.p1[0], v.p1[1], v.p1[2]}.data(), 3);
  } else {
    putLocalArray("vertices", vertices.data(), adios2::Dims{nLocalPts, 3});
  }

  // per-step scalars referenced by the vtk.xml TIME tag / CYCLE metadata
  {
    auto vt = S.io.InquireVariable<double>("time") ? S.io.InquireVariable<double>("time")
                                                   : S.io.DefineVariable<double>("time");
    S.engine.Put(vt, time, adios2::Mode::Sync);
    auto vc = S.io.InquireVariable<int32_t>("CYCLE") ? S.io.InquireVariable<int32_t>("CYCLE")
                                                     : S.io.DefineVariable<int32_t>("CYCLE");
    S.engine.Put(vc, static_cast<int32_t>(tstep), adios2::Mode::Sync);
  }

  // field variables: one local array per field, component-major (matches .vts)
  std::vector<std::vector<float>> scratch; // keep alive until PerformDataWrite
  scratch.reserve(userFieldList.size());
  int ifld = 0;
  for (const auto &entry : userFieldList) {
    const std::string fieldName = std::get<0>(entry);
    const int fieldDim = std::get<1>(entry).size();

    scratch.emplace_back(static_cast<size_t>(fieldDim) * nLocalPts);
    auto &buf = scratch.back();
    for (int d = 0; d < fieldDim; d++) {
      for (dlong i = 0; i < v.nLocal; i++) {
        buf[d * nLocalPts + i] = v.data[v.offsetScan[ifld] + d * v.nLocal + i];
      }
    }
    if (fieldDim > 1) {
      putLocalArray(fieldName, buf.data(), adios2::Dims{nLocalPts, static_cast<size_t>(fieldDim)});
    } else {
      putLocalArray(fieldName, buf.data(), adios2::Dims{nLocalPts});
    }
    ifld++;
  }

  // vtk.xml schema (once) so ParaView reads the .bp as an UnstructuredGrid
  if (S.step == 0) {
    std::string schema = "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
                         "  <UnstructuredGrid>\n"
                         "    <Piece NumberOfPoints=\"numOfPoints\" NumberOfCells=\"numOfCells\">\n"
                         "      <Points>\n"
                         "        <DataArray Name=\"vertices\" />\n"
                         "      </Points>\n"
                         "      <Cells>\n"
                         "        <DataArray Name=\"connectivity\" />\n"
                         "        <DataArray Name=\"types\" />\n"
                         "      </Cells>\n"
                         "      <PointData>\n";
    for (const auto &entry : userFieldList) {
      schema += "        <DataArray Name=\"" + std::get<0>(entry) + "\" />\n";
    }
    schema += "        <DataArray Name=\"TIME\"> time </DataArray>\n"
              "      </PointData>\n"
              "    </Piece>\n"
              "  </UnstructuredGrid>\n"
              "</VTKFile>\n";
    S.io.DefineAttribute<std::string>("vtk.xml", schema);
  }

  S.engine.PerformDataWrite();
  S.engine.EndStep();
  S.step++;

  platform->timer.toc("fieldExtract::io");
}
#endif // NEKRS_ENABLE_ADIOS

inline void fieldExtract::process(const double time, const int tstep, const std::string &format)
{
  nekrsCheck(!setupCalled, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "process called prior to setup!");

  interpolate();

  stepCounter++;
  const int out_step = stepCounter;

  const int nx = boxDims[0];
  const int ny = (boxDim > 1) ? boxDims[1] : 1;
  const int nz = (boxDim > 2) ? boxDims[2] : 1;

  GridView v{std::array<int, 3>{nx, ny, nz},
             distCode,
             numPoints,
             numPointsLocal,
             numPointsScan,
             xCoord,
             yCoord,
             zCoord,
             fldData,
             fldDataOffsetScan,
             point0,
             point1};

  writeGrid(v, fileName, out_step, time, tstep, format);
}

inline void fieldExtract::doAvg(const double time,
                                const int tstep,
                                const std::string &avgDir,
                                const std::string &mode,
                                const std::string &format)
{
  nekrsCheck(!setupCalled, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "doAvg called prior to setup!");
  nekrsCheck(!boxMode && !userAxes,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "doAvg requires box mode, or points mode with setQuadrature()!");
  nekrsCheck(mode != "gather" && mode != "gather-scatter",
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "invalid doAvg mode '%s' (use gather|gather-scatter)\n",
             mode.c_str());

  // parse avgDir -> which axes to average
  std::array<bool, 3> avg{false, false, false};
  {
    int bad = 0;
    for (char c : avgDir) {
      const int a = (c == 'x' || c == 'X') ? 0 : (c == 'y' || c == 'Y') ? 1 : (c == 'z' || c == 'Z') ? 2 : -1;
      if (a < 0 || avg[a]) {
        bad = 1;
        break;
      }
      avg[a] = true;
    }
    const bool any = avg[0] || avg[1] || avg[2];
    nekrsCheck(bad || !any,
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "invalid avgDir '%s' (non-repeating subset of x,y,z)\n",
               avgDir.c_str());
  }

  const int n[3] = {boxDims[0], (boxDim > 1) ? boxDims[1] : 1, (boxDim > 2) ? boxDims[2] : 1};
  for (int a = 0; a < 3; a++) {
    nekrsCheck(avg[a] && n[a] < 2,
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "cannot average singleton axis '%c'\n",
               "xyz"[a]);
  }

  const int nx = n[0], ny = n[1], nz = n[2];
  const int nxy = nx * ny;

  std::string dirTag; // averaged axes in x,y,z order
  for (int a = 0; a < 3; a++) {
    if (avg[a]) {
      dirTag += "xyz"[a];
    }
  }
  const std::string modeTag = (mode == "gather") ? "g" : "gs";

  // reduced grid: averaged axes collapse to 1
  const int rn[3] = {avg[0] ? 1 : nx, avg[1] ? 1 : ny, avg[2] ? 1 : nz};
  const dlong rN = static_cast<dlong>(rn[0]) * rn[1] * rn[2];
  const int rnxy = rn[0] * rn[1];

  auto &ags = setupAvgGs(avg, dirTag); // cached after the first call per direction
  const dlong nTotal = ags.nTotal;
  const dlong rLocal = ags.rLocal;
  const dlong rScan = ags.rScan;

  if (!avgAnnounced) {
    avgAnnounced = true;
    if (platform->comm.mpiRank() == 0) {
      printf("fieldExtract(%s): doAvg dir=%s mode=%s nReduced=%lld\n",
             fileName.c_str(),
             dirTag.c_str(),
             mode.c_str(),
             (long long)rN);
      fflush(stdout);
    }
  }

  platform->timer.tic("fieldExtract::avg");
  interpolate();

  int nComp = 0; // total scalar components over all fields
  for (auto &entry : userFieldList) {
    nComp += std::get<1>(entry).size();
  }

  // Accumulate quadrature-weighted partial sums into the per-cell slots (one gs
  // vector of length nTotal per component; tail entries start at zero). axisWeight
  // is normalized per axis, so after the gs the sum IS the average.
  std::vector<dfloat> wrk(static_cast<size_t>(nComp) * nTotal, 0.0);
  {
    int comp = 0, ifld = 0;
    for (const auto &entry : userFieldList) {
      const int fieldDim = std::get<1>(entry).size();
      for (int d = 0; d < fieldDim; d++) {
        for (dlong i = 0; i < numPointsLocal; i++) {
          const dlong ig = i + numPointsScan;
          int ix, iy, iz;
          getIxyz(ig, nx, nxy, ix, iy, iz);
          dfloat w = 1.0;
          if (avg[0]) {
            w *= axisWeight[0][ix];
          }
          if (avg[1]) {
            w *= axisWeight[1][iy];
          }
          if (avg[2]) {
            w *= axisWeight[2][iz];
          }
          wrk[static_cast<size_t>(comp) * nTotal + ags.pointSlot[i]] +=
              w * fldData[fldDataOffsetScan[ifld] + d * numPointsLocal + i];
        }
        comp++;
      }
      ifld++;
    }
  }

  // combine partial sums across ranks; the total scatters back to every member
  oogs::startFinish(wrk.data(), nComp, nTotal, ogsDfloat, ogsAdd, ags.gsh);
  platform->timer.toc("fieldExtract::avg");

  const std::string stem = fileName + "_avg" + dirTag + "_" + modeTag;
  const int out_step = ++avgCounter[stem];

  if (mode == "gather-scatter") {
    // scatter each point's cell average back through pointSlot (reuse box grid)
    std::vector<float> sdata(fldDataOffsetScan[Nfields]);
    int comp = 0, ifld = 0;
    for (const auto &entry : userFieldList) {
      const int fieldDim = std::get<1>(entry).size();
      for (int d = 0; d < fieldDim; d++) {
        for (dlong i = 0; i < numPointsLocal; i++) {
          sdata[fldDataOffsetScan[ifld] + d * numPointsLocal + i] =
              static_cast<float>(wrk[static_cast<size_t>(comp) * nTotal + ags.pointSlot[i]]);
        }
        comp++;
      }
      ifld++;
    }

    GridView v{std::array<int, 3>{nx, ny, nz},
               distCode,
               numPoints,
               numPointsLocal,
               numPointsScan,
               xCoord,
               yCoord,
               zCoord,
               sdata,
               fldDataOffsetScan,
               point0,
               point1};
    writeGrid(v, stem, out_step, time, tstep, format);
    return;
  }

  // mode == "gather": emit the collapsed grid (the wrk tail, already in the
  // reduced-grid split). Surviving axes keep their coords; collapsed axes -> midpoint.
  const dfloat mid[3] = {static_cast<dfloat>(0.5) * (point0[0] + point1[0]),
                         static_cast<dfloat>(0.5) * (point0[1] + point1[1]),
                         static_cast<dfloat>(0.5) * (point0[2] + point1[2])};

  std::vector<dfloat> rx(rLocal), ry(rLocal), rz(rLocal);
  for (dlong i = 0; i < rLocal; i++) {
    const dlong ig = i + rScan;
    int jx, jy, jz;
    getIxyz(ig, rn[0], rnxy, jx, jy, jz);
    rx[i] = avg[0] ? mid[0] : axisCoord[0][jx];
    ry[i] = avg[1] ? mid[1] : axisCoord[1][jy];
    rz[i] = avg[2] ? mid[2] : axisCoord[2][jz];
  }

  // local reduced data slice + its offset scan
  std::vector<size_t> rLocalOffset(Nfields + 1);
  rLocalOffset[0] = 0;
  {
    int ifld = 1;
    for (auto &entry : userFieldList) {
      const auto dimf = std::get<1>(entry).size();
      rLocalOffset[ifld] = rLocalOffset[ifld - 1] + static_cast<size_t>(rLocal) * dimf;
      ifld++;
    }
  }
  std::vector<float> rdata(rLocalOffset[Nfields]);
  {
    int comp = 0, ifld = 0;
    for (const auto &entry : userFieldList) {
      const int fieldDim = std::get<1>(entry).size();
      for (int d = 0; d < fieldDim; d++) {
        for (dlong i = 0; i < rLocal; i++) {
          rdata[rLocalOffset[ifld] + d * rLocal + i] =
              static_cast<float>(wrk[static_cast<size_t>(comp) * nTotal + ags.nSlots + i]);
        }
        comp++;
      }
      ifld++;
    }
  }

  // collapsed axes are trivially uniform (single point)
  const std::array<int, 3> rDist{avg[0] ? 0 : distCode[0],
                                 avg[1] ? 0 : distCode[1],
                                 avg[2] ? 0 : distCode[2]};

  GridView v{std::array<int, 3>{rn[0], rn[1], rn[2]},
             rDist,
             rN,
             rLocal,
             rScan,
             rx,
             ry,
             rz,
             rdata,
             rLocalOffset,
             point0,
             point1};
  writeGrid(v, stem, out_step, time, tstep, format);
}

#endif // nekrs_field_extract_hpp_
