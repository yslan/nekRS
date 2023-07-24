/*

   The MIT License (MIT)

   Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

 */


#include "elliptic.h"
#include "timer.hpp"
#include "linAlg.hpp"

//#define DEBUG
PchebData::PchebData(elliptic_t *elliptic) {
  isConvRateReady = 0;
}

void initializePchebData(elliptic_t* elliptic) {
   PchebData *pchebData = new PchebData(elliptic);
   elliptic->pchebData = pchebData;
} 

static void ChebyshevSolver(elliptic_t* elliptic, occa::memory &o_r, occa::memory &o_x,
                            int niter, dfloat dmin, dfloat dmax, int restart) {

  mesh_t* mesh = elliptic->mesh;
  setupAide& options = elliptic->options;
  const int verbose = platform->options.compareArgs("VERBOSE", "TRUE");

  const dfloat theta = 0.5 * (dmax + dmin);
  const dfloat delta = 0.5 * (dmax - dmin);
  const dfloat invTheta = 1.0 / theta;

  const dfloat sigma = theta / delta;
  dfloat rho_prev = 1. / sigma;

  const dfloat one = 1., mone = -1., zero = 0.0;
  const dlong Nlocal = mesh->Nlocal;
  const int Nfields = elliptic->Nfields;
  const dlong offset = elliptic->fieldOffset;

  occa::memory &o_d = elliptic->o_p;
  occa::memory &o_w = elliptic->o_tmp; //FIXME
  occa::memory &o_z = (!options.compareArgs("PRECONDITIONER", "NONE")) ? elliptic->o_z : o_x;
  occa::memory &o_Ap = elliptic->o_Ap;
  occa::memory &o_weight = elliptic->o_invDegree;
  dfloat resNormFactor = elliptic->resNormFactor;
  dfloat rdotr;

  if (restart==0) {
    // r = r - A Minv z, z0 = 0
    platform->linAlg->scaleMany(Nlocal, Nfields, offset, zero, o_z); // z = 0 * z
  
    // Only compute norm in verbose mode
    if (verbose) {
      rdotr = platform->linAlg->weightedNorm2Many(
            Nlocal, Nfields, offset, o_weight, o_r, platform->comm.mpiComm)
          * sqrt(resNormFactor);
      if (platform->comm.mpiRank == 0) {
        printf("CHEB it %d r norm %.8e \n",0,rdotr);
      }
    }
    
    // d = invTheta*res
    // d = 0 * d + invTheta * r
    platform->linAlg->axpbyMany(Nlocal, Nfields, offset, invTheta, o_r, zero, o_d);
  }

  for (int k = 1; k <= niter; k++) {
    // x = x + d
    platform->linAlg->axpbyMany(Nlocal, Nfields, offset, one, o_d, one, o_z);   

    // r = r - SA d = r - z
    if(!options.compareArgs("PRECONDITIONER", "NONE")) {
      ellipticPreconditioner(elliptic, o_d, o_w);
      ellipticOperator(elliptic, o_w, o_Ap, dfloatString);
    } else {
      ellipticOperator(elliptic, o_d, o_Ap, dfloatString);
    }
    platform->linAlg->axpbyMany(Nlocal, Nfields, offset, mone, o_Ap, one, o_r); // r = r - z

    // Only compute norm in verbose mode
    if (verbose) {
      rdotr = platform->linAlg->weightedNorm2Many(
           Nlocal, Nfields, offset, o_weight, o_r, platform->comm.mpiComm)
         * sqrt(resNormFactor);
      if (platform->comm.mpiRank == 0) {
        printf("CHEB it %d r norm %.8e \n",k+restart,rdotr);
      }
    }

    const dfloat rho = 1.0 / (2.0 * sigma - rho_prev);
    const dfloat rCoeff = 2.0 * rho / delta;
    const dfloat dCoeff = rho * rho_prev;
    // d = rho*rho_prev * d + 2 * rho / delta * r
    platform->linAlg->axpbyMany(Nlocal, Nfields, offset, rCoeff, o_r, dCoeff, o_d);    
    rho_prev = rho;
  }

  // x = x + d
  platform->linAlg->axpbyMany(Nlocal, Nfields, offset, one, o_d, one, o_z);

  if(!options.compareArgs("PRECONDITIONER", "NONE")) {
    ellipticPreconditioner(elliptic, o_z, o_x);
  }
}

int chebyshev_aux(elliptic_t* elliptic, occa::memory &o_r, occa::memory &o_x,
        const dfloat tol, const int MAXIT, dfloat &rdotr, const int tstep)
{
  mesh_t* mesh = elliptic->mesh;
  setupAide& options = elliptic->options;
  const int verbose = platform->options.compareArgs("VERBOSE", "TRUE");

  int startStep = 10, iter = 10;
  elliptic->options.getArgs("CHEBYSHEV STARTSTEP", startStep);
  elliptic->options.getArgs("CHEBYSHEV ITER", iter);
  const int extra = options.compareArgs("CHEBYSHEV EXTRA ITER", "TRUE");
  const int guess = options.compareArgs("CHEBYSHEV GUESS ITER", "TRUE");
  const int isConvRateReady = elliptic->pchebData->isConvRateReady;

  if (startStep<=0) {
    nrsAbort(platform->comm.mpiComm, EXIT_FAILURE,
             "%s%d\n", "Chebyshev solver: Invalid startStep!",startStep); 
  }
  if (iter<=0) {
    nrsAbort(platform->comm.mpiComm, EXIT_FAILURE,
             "%s%d\n", "Chebyshev solver: Invalid iter!",iter); 
  }

  dfloat dmin = elliptic->pcgEigenData->dmin;
  dfloat dmax = elliptic->pcgEigenData->dmax;
  if (verbose && platform->comm.mpiRank == 0) { 
    printf("%s Chebyshev step=%d, start=%d iter=%d guess=%d extra=%d dmin %.6e dmax %.6e\n"
          ,elliptic->name.c_str(),tstep,startStep,iter,guess,extra,dmin,dmax);
  }

  if (tstep < startStep) {
    // Run PCG to estimate eigenvalues
    iter = pcg_eigen(elliptic, o_r, o_x, tol, MAXIT, rdotr, dmin, dmax);
    if (iter>=3) {
      elliptic->pcgEigenData->dmin = dmin;
      elliptic->pcgEigenData->dmax = dmax;
      elliptic->pcgEigenData->isEigenReady = 1;
    }
  }
  else {
    if (elliptic->pcgEigenData->isEigenReady==0) {
      nrsAbort(platform->comm.mpiComm, EXIT_FAILURE,
               "%s\n", "Chebyshev solver: dmin/dmax is not set!");
    }

    // Guess Iteration number based on previous conv. rate
    if (guess && isConvRateReady==1) {
      iter = ((int)  ( (log(tol) - log(elliptic->res0Norm)) / elliptic->pchebData->logConvRate ) ) + 1;
      if (verbose && platform->comm.mpiRank == 0) {
        printf("%s Chebyshev guess iter: r0 = %.6e  rate = %.6e  change iter to %d \n"
              ,elliptic->name.c_str(),elliptic->res0Norm,exp(elliptic->pchebData->logConvRate),iter);
      }
    }

    // Main solve
    if (verbose && platform->comm.mpiRank == 0) { 
      printf("CHEBYSHEV ");
      printf("%s: initial res norm %.15e WE NEED TO GET TO %e \n", elliptic->name.c_str(), rdotr, tol);
    }
    ChebyshevSolver(elliptic, o_r, o_x, iter, dmin, dmax, 0);

    // Compute res
    rdotr = platform->linAlg->weightedNorm2Many(
        mesh->Nlocal,
        elliptic->Nfields,
        elliptic->fieldOffset,
        elliptic->o_invDegree,
        o_r,
        platform->comm.mpiComm
      ) 
      * sqrt(elliptic->resNormFactor);

    const dfloat logConvRate = (log(rdotr) - log(elliptic->res0Norm)) / ((dfloat) iter);
    const dfloat conditionNumber = dmax / dmin;
    const dfloat theoConvRate = (sqrt(conditionNumber)-1.0) / (sqrt(conditionNumber) + 1.0);
    if (verbose && platform->comm.mpiRank == 0) {
      printf("%s Chebyshev stat: iter = %d, rnorm from %.6e to %.6e, Conv rate = %.6e (theo = %.6e)\n"
            ,elliptic->name.c_str(),iter,elliptic->res0Norm,rdotr,exp(logConvRate),theoConvRate);
    }    

    // Store logConvRate
    if (guess && isConvRateReady==0) {
      elliptic->pchebData->logConvRate = logConvRate; // or using theoConvRate??
      elliptic->pchebData->isConvRateReady = 1;
      if (platform->comm.mpiRank == 0) {
        printf("%s Chebyshev save ConvRate at step = %d, with rate = %.8e \n"
              ,elliptic->name.c_str(),tstep,exp(logConvRate));
      }
    }

    // Extra iterations
    if (extra && rdotr>tol) {
      const int extraIteration = ((int)  ( (log(tol) - log(rdotr)) / logConvRate ) ) + 1;
      if (extraIteration>=1) {
        ChebyshevSolver(elliptic, o_r, o_x, extraIteration, dmin, dmax, iter);
        iter = iter + extraIteration;
      }
      if (platform->comm.mpiRank == 0) {
        printf("%s Chebyshev extra: conv. rate = %.6e (theo. rate = %.6e), extraIter = %d \n"
              ,elliptic->name.c_str(),exp(logConvRate),theoConvRate,extraIteration);
      }

/*
      // if iter is off, update estimator in next step // TODO, this is kind of a corner case.
      if (guess && extraIteration>=5) {
        elliptic->pchebData->logConvRate = logConvRate; 
        elliptic->pchebData->isConvRateReady = 1;
        if (platform->comm.mpiRank == 0) {
          printf("%s Chebyshev save ConvRate at step = %d, with rate = %.8e \n"
                ,elliptic->name.c_str(),tstep,exp(logConvRate));
        }
      }
*/
    }
  }

  return iter;
}
