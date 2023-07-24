include(config/bench.cmake)
include(config/mesh.cmake)
include(config/elliptic.cmake)
include(config/gslib.cmake)

set(SRC 
    src/lib/nekrs.cpp
    src/io/writeFld.cpp
    src/io/fileUtils.cpp
    src/utils/sha1.cpp
    src/utils/inipp.cpp
    src/utils/unifdef.c
    src/utils/mysort.cpp
    src/utils/parallelSort.cpp
    src/utils/tinyexpr.c
    src/utils/setupAide.cpp
    src/core/numberActiveFields.cpp
    src/core/printHeader.cpp
    src/navierStokes/cfl.cpp
    src/regularization/lowPassFilter.cpp
    src/regularization/avm.cpp
    src/bdry/bcMap.cpp
    src/core/compileKernels.cpp
    src/setup/setup.cpp
    src/bdry/alignment.cpp
    src/core/registerNrsKernels.cpp
    src/core/registerNekNekKernels.cpp
    src/core/registerPostProcessingKernels.cpp
    src/core/registerEllipticKernels.cpp
    src/core/registerEllipticPreconditionerKernels.cpp
    src/core/registerCdsKernels.cpp
    src/core/registerLinAlgKernels.cpp
    src/core/registerMeshKernels.cpp
    src/core/LVector.cpp
    src/bdry/createEToBV.cpp
    src/navierStokes/applyDirichlet.cpp
    src/navierStokes/timeStepper.cpp
    src/navierStokes/evaluateProperties.cpp
    src/navierStokes/subCycling.cpp
    src/navierStokes/tombo.cpp
    src/navierStokes/constantFlowRate.cpp
    src/navierStokes/Urst.cpp
    src/cds/cdsSolve.cpp
    src/setup/parReader.cpp
    src/io/re2Reader.cpp
    src/setup/configReader.cpp
    src/core/timer.cpp
    src/core/platform.cpp
    src/core/comm.cpp
    src/core/flopCounter.cpp
    src/core/kernelRequestManager.cpp
    src/core/device.cpp
    src/linAlg/linAlg.cpp
    src/linAlg/matrixConditionNumber.cpp
    src/linAlg/matrixInverse.cpp
    src/linAlg/matrixEig.cpp
    src/linAlg/matrixTranspose.cpp
    src/linAlg/matrixRightSolve.cpp
    src/plugins/tavg.cpp
    src/plugins/velRecycling.cpp
    src/plugins/RANSktau.cpp
    src/plugins/lowMach.cpp
    src/plugins/lpm.cpp
    src/pointInterpolation/findpts/findpts.cpp
    src/pointInterpolation/pointInterpolation.cpp
    src/neknek/neknek.cpp
    src/udf/udf.cpp
    src/udf/compileUDFKernels.cpp
    src/nekInterface/nekInterfaceAdapter.cpp
    src/postProcessing/planarAvg.cpp
    src/postProcessing/strainRotationRate.cpp
    src/postProcessing/viscousDrag.cpp
    src/postProcessing/Qcriterion.cpp
    src/core/registerCvodeKernels.cpp
    src/solvers/cvode/cvode.cpp
    ${BENCH_SOURCES}
    ${MESH_SOURCES}
    ${ELLIPTIC_SOURCES}
    ${OGS_SOURCES}
    ${FINDPTS_SOURCES}
)

set_property(
   SOURCE src/core/printHeader.cpp 
   APPEND PROPERTY COMPILE_DEFINITIONS
   GITCOMMITHASH="${GIT_COMMIT_HASH}"
   NEKRS_VERSION=${PROJECT_VERSION_MAJOR}
   NEKRS_SUBVERSION=${PROJECT_VERSION_MINOR}
   NEKRS_PATCHVERSION=${PROJECT_VERSION_PATCH}
)

add_library(nekrs-lib SHARED ${SRC})
set_target_properties(nekrs-lib PROPERTIES LINKER_LANGUAGE CXX OUTPUT_NAME nekrs)

target_include_directories(nekrs-lib 
  PUBLIC 
  ${CMAKE_CURRENT_BINARY_DIR}
  src
  src/setup
  src/bdry
  src/core
  src/utils
  src/lib
  src/io
  src/udf
  src/regularization
  src/linAlg
  src/navierStokes
  src/neknek
  src/cds
  src/pointInterpolation/findpts
  src/postProcessing
  src/pointInterpolation
  src/solvers/cvode
  src/lns
  ${BENCH_SOURCE_DIR}
  ${BENCH_SOURCE_DIR}/core
  ${BENCH_SOURCE_DIR}/fdm
  ${BENCH_SOURCE_DIR}/axHelm
  ${BENCH_SOURCE_DIR}/advsub
  ${MESH_SOURCE_DIR}
  ${NEKINTERFACEDIR}
  ${OGS_SOURCE_DIR}/include
  ${OGS_SOURCE_DIR}
  ${FINDPTS_SOURCE_DIR}
  ${ELLIPTIC_SOURCE_DIR}
  PRIVATE
  ${ELLIPTIC_SOURCE_DIR}/amgSolver/hypre
  ${ELLIPTIC_SOURCE_DIR}/amgSolver/amgx
  ${ELLIPTIC_SOURCE_DIR}/MG
)

if(NEKRS_USE_DFLOAT_FLOAT)
  target_compile_definitions(nekrs-lib PUBLIC -DNEKRS_USE_DFLOAT_FLOAT)
  target_compile_definitions(nekrs-lib PUBLIC -DOGS_USE_DFLOAT_FLOAT)
endif()

add_executable(nekrs-bin src/main.cpp)
target_include_directories(nekrs-bin PRIVATE src/lib src/utils)
set_target_properties(nekrs-bin PROPERTIES LINKER_LANGUAGE CXX OUTPUT_NAME nekrs)
