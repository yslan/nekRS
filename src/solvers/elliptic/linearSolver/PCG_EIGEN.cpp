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


// In the PCG, we collect the alpha/beta from PCG to form tri-diagonal matrix (Lanczo)
// then the Ritz values will be used to estimated the min/max eigenvalues to later usage

#include "elliptic.h"
#include "timer.hpp"
#include "linAlg.hpp"

//#define DEBUG
PcgEigenData::PcgEigenData(elliptic_t *elliptic)
    : maxIter([&]() {
        int _maxIter = 500;
        elliptic->options.getArgs("MAXIMUM ITERATIONS", _maxIter);
        return _maxIter;
      }()),
      diagt((dfloat *)calloc(maxIter, sizeof(dfloat))),
      upper((dfloat *)calloc(maxIter, sizeof(dfloat)))
{
   isEigenReady = 0;
   dmin = 1e31;
   dmax = -1e31;
}

void initializePcgEigenData(elliptic_t *elliptic) {
  PcgEigenData *pcgEigenData = new PcgEigenData(elliptic);
  elliptic->pcgEigenData = pcgEigenData;
}

extern "C" {
void dstevd_ (char* JOBZ, int* N, double* D, double *E, double *Z, int* LDZ, 
              double* WORK, int* LWORK, int* IWORK, int* LIWORK, int* INFO);
}

// compute right eigenvectors
void triDiagEig(int N, dfloat* D, dfloat* E, dfloat &dmin, dfloat &dmax)
{
  int verbose = platform->options.compareArgs("VERBOSE", "TRUE");

  char JOBZ = 'N';
  int LDZ = N;
  int LWORK = 1;
  int IWORK = 1;
  int LIWORK = 1;

  double* WORK  = new double[LWORK];
  double* tmpZ = NULL;

  auto invalid = 0;
  for(int i = 0; i < N; i++) {
    if(std::isnan(D[i]) || std::isinf(D[i])) invalid++;
  }
  for(int i = 0; i < N-1; i++) {
    if(std::isnan(E[i]) || std::isinf(E[i])) invalid++;
  }
  nrsCheck(invalid, platform->comm.mpiComm, EXIT_FAILURE,
           "%s\n", "invalid matrix entries!");

  int INFO = -999;
  dstevd_ (&JOBZ, &N, D, E, tmpZ, &LDZ, WORK, &LWORK, &IWORK, &LIWORK, &INFO);

  nrsCheck(INFO != 0, platform->comm.mpiComm, EXIT_FAILURE,
           "%s %d\n", "dstevd failed INFO = ", INFO);

  dmin = D[0]; // LAPACAK has sorted it?
  dmax = D[N-1];
  for (int i = 0; i < N; i++) {
    dmax = std::max(D[i],dmax);
    dmin = std::min(D[i],dmin);
  }

  int ichk = 0;
  if (dmax<0 || dmin<0) {
    ichk = 1;
    verbose = 1;
  }
  
  if (verbose && platform->comm.mpiRank == 0) {
    for(int i = 0; i < N; i++) {
      printf("Ritz Value by LAPACK %d / %d   %.6e\n", i+1, N, D[i]);
    }
  }

  nrsCheck(ichk, platform->comm.mpiComm, EXIT_FAILURE,
           "%s\n", "negative eigenvalues!");

  delete [] WORK;
    
}

static dfloat f77sgn(dfloat val1, dfloat val2) { // return val1 with sign of val2
  return (val2 < 0) ? -std::abs(val1) : std::abs(val1);
}

// Solve eigenvalue of tri-diagonal matrix
// copied from subroutine calc (diag,upper,d,e,n,dmax,dmin) (Nek5000/core/hmholtz.f)
// TODO: we can call it directly via nekInterface?
static void calc_eigen(dfloat* diagt, dfloat* upper, const int n, dfloat &dmin, dfloat &dmax)
{
  // Rayleight-quotient with QR iter and Wilkinson's shift

  const int verbose = platform->options.compareArgs("VERBOSE", "TRUE");
  dfloat* d = diagt; 
  dfloat* e = upper;
  const dfloat tol=1.e-16;

  // TODO: change index to 0-base
  uint l;
  for (l=1;l<=n;l++) { 

    // QR iteration
    uint iter = 0;
    uint iskip = 0;
    do {

      // detect m such that off-diag[m] = 0
      uint m_in = l-1;
      do {
        m_in++;

        uint m;
        const dfloat ee = std::abs(e[m_in-1]);
        if ( ee < tol ) {
           m = m_in;
        } else if(m_in==n-1) {
           m = n;
        } else {
           continue;
        }


        if (m==l) {
          iskip = 1;
        } else { // m!=l
          iter = iter + 1;
          dfloat g = (d[l]-d[l-1]) / (2.0*e[l-1]);
          dfloat r = sqrt(g*g+1.0);
    
          g = d[m-1] - d[l-1] + e[l-1]/(g + f77sgn(r,g));
          dfloat s = 1.0;
          dfloat c = 1.0;
          dfloat p = 0.0;
    
          // QR iteration using givens' rotation
          uint i;
          for (i=m-1;i>=l;i--) {
            const dfloat f = s * e[i-1];
            const dfloat b = c * e[i-1];
    
            if (std::abs(f) >= std::abs(g)) {
              c = g / f;
              r = sqrt(c*c+1.0);
              e[i] = f*r;
              s = 1.0/r;
              c = c * s;
            } else {
              s = f / g;
              r = sqrt(s*s+1.0);
              e[i] = g*r;
              c =  1.0 / r;
              s = s * c;
            }
    
            g = d[i] - p;
            r = (d[i-1]-g) * s + 2.0 * c * b;
            p = s * r;
            d[i] = g + p;
            g = c*r - b;
          } // 14
    
          d[l-1] = d[l-1] - p;
          e[l-1] = g;
          e[m-1] = 0.0;
        } // m<=l

      } // m_in loop, exit at m_in==n
      while (iskip==0 && m_in<n-1); 

    } // outer iteration
    while (iskip==0 && iter<=30);

  } // l loop

  dmax = 0.0;
  dmin = d[0];

  uint i;
  for (i=1;i<=n;i++) {
    dmax = std::abs(std::max(d[i-1],dmax));
    dmin = std::abs(std::min(d[i-1],dmin));
  }
  if(platform->comm.mpiRank == 0 && verbose) {
    for (i=1;i<=n;i++) {
      printf("PCG Ritz values %d / %d   %.6e \n", i, n, d[i-1]);
    }
  }
  if (platform->comm.mpiRank == 0) {
    nrsCheck(std::isnan(dmin), MPI_COMM_SELF, EXIT_FAILURE,
             "%s\n", "Detected invalid dmin while solving eigenvalues!");
    nrsCheck(std::isnan(dmax), MPI_COMM_SELF, EXIT_FAILURE,
             "%s\n", "Detected invalid dmax while solving eigenvalues!");
  }
}


static dfloat update(elliptic_t* elliptic,
                     occa::memory &o_p, occa::memory &o_Ap, const dfloat alpha,
                     occa::memory &o_x, occa::memory &o_r)
{
  mesh_t* mesh = elliptic->mesh;

  const bool serial = platform->serial;

  // r <= r - alpha*A*p
  // dot(r,r)
  elliptic->updatePCGKernel(mesh->Nlocal,
                            elliptic->fieldOffset,
                            elliptic->o_invDegree,
                            o_Ap,
                            alpha,
                            o_r,
                            elliptic->o_tmpNormr);

  dfloat rdotr1 = 0;
#ifdef ELLIPTIC_ENABLE_TIMER
    //platform->timer.tic("dotp",1);
#endif
  if(serial) {
    rdotr1 = *((dfloat *) elliptic->o_tmpNormr.ptr());
  } else {
    const dlong Nblock = (mesh->Nlocal + BLOCKSIZE - 1) / BLOCKSIZE;
    elliptic->o_tmpNormr.copyTo(elliptic->tmpNormr, Nblock*sizeof(dfloat));
    for(int n = 0; n < Nblock; ++n)
      rdotr1 += elliptic->tmpNormr[n];
  }

  // x <= x + alpha*p
  platform->linAlg->axpbyMany(
    mesh->Nlocal,
    elliptic->Nfields,
    elliptic->fieldOffset,
    alpha,
    o_p,
    1.0,
    o_x);

  MPI_Allreduce(MPI_IN_PLACE, &rdotr1, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);
#ifdef ELLIPTIC_ENABLE_TIMER
    //platform->timer.toc("dotp");
#endif

  platform->flopCounter->add(elliptic->name + " ellipticUpdatePC",
                             elliptic->Nfields * static_cast<double>(mesh->Nlocal) * 6 + mesh->Nlocal);

  return rdotr1;
}

int pcg_eigen(elliptic_t* elliptic, occa::memory &o_r, occa::memory &o_x,
        const dfloat tol, const int MAXIT, dfloat &rdotr, dfloat &dmin, dfloat &dmax)
{
  
  mesh_t* mesh = elliptic->mesh;
  setupAide& options = elliptic->options;

  const int flexible = options.compareArgs("SOLVER", "FLEXIBLE"); // TODO: does this work with fcg?
  const int verbose = platform->options.compareArgs("VERBOSE", "TRUE");
  const int fixedIteration = false;

  dfloat* diagt = elliptic->pcgEigenData->diagt;
  dfloat* upper = elliptic->pcgEigenData->upper;

  dfloat rdotz1;
  dfloat alpha;
  dfloat pAp;

  /*aux variables */
  occa::memory &o_p  = elliptic->o_p;
  occa::memory &o_z = (!options.compareArgs("PRECONDITIONER", "NONE")) ? elliptic->o_z : o_r;
  occa::memory &o_Ap = elliptic->o_Ap;
  occa::memory &o_weight = elliptic->o_invDegree;
  platform->linAlg->fill(elliptic->Nfields * elliptic->fieldOffset, 0.0, o_p);

  if(platform->comm.mpiRank == 0 && verbose) {
    if(flexible) 
      printf("PFCG ");	  
    else
      printf("PCG ");	  
    printf("%s: initial res norm %.15e WE NEED TO GET TO %e \n", elliptic->name.c_str(), rdotr, tol);
  }

  int iter = 0;
  do {
    iter++;
    const dfloat rdotz2 = rdotz1;
    if(!options.compareArgs("PRECONDITIONER", "NONE")) {
      ellipticPreconditioner(elliptic, o_r, o_z);

      rdotz1 = platform->linAlg->weightedInnerProdMany(
        mesh->Nlocal,
        elliptic->Nfields,
        elliptic->fieldOffset,
        o_weight,
        o_r,
        o_z,
        platform->comm.mpiComm);
    } else {
      rdotz1 = rdotr; 
    }

#ifdef DEBUG
    printf("norm rdotz1: %.15e\n", rdotz1);
#endif

    dfloat beta = 0;
    if(iter > 1) {
      beta = rdotz1/rdotz2;
      if(flexible) {
        const dfloat zdotAp = platform->linAlg->weightedInnerProdMany(
          mesh->Nlocal,
          elliptic->Nfields,
          elliptic->fieldOffset,
          o_weight,
          o_z,
          o_Ap,
          platform->comm.mpiComm);
        beta = -alpha * zdotAp/rdotz2;
#ifdef DEBUG
        printf("norm zdotAp: %.15e\n", zdotAp);
#endif
      }
    }

#ifdef DEBUG
        printf("beta: %.15e\n", beta);
#endif


    platform->linAlg->axpbyMany(
      mesh->Nlocal,
      elliptic->Nfields,
      elliptic->fieldOffset,
      1.0,
      o_z,
      beta,
      o_p);

    const dfloat pAp0 = pAp;
    ellipticOperator(elliptic, o_p, o_Ap, dfloatString);
    pAp = platform->linAlg->weightedInnerProdMany(
      mesh->Nlocal,
      elliptic->Nfields,
      elliptic->fieldOffset,
      o_weight,
      o_p,
      o_Ap,
      platform->comm.mpiComm);
    alpha = rdotz1 / (pAp + 1e-300);

#ifdef DEBUG
    printf("alpha: %.15e\n", alpha);
    printf("norm pAp: %.15e\n", pAp);
#endif

    //  x <= x + alpha*p
    //  r <= r - alpha*A*p
    //  dot(r,r)
    rdotr = sqrt(update(elliptic, o_p, o_Ap, alpha, o_x, o_r) * elliptic->resNormFactor);
#ifdef DEBUG
    printf("rdotr: %.15e\n", rdotr);
#endif

    if (iter==1) {
      diagt[iter-1] = pAp / rdotz1;
    } else {
      diagt[iter-1] = (beta*beta*pAp0 + pAp) / (rdotz1 + 1e-300);
      upper[iter-1] = -beta*pAp0 / sqrt(rdotz2*rdotz1);
    }
    if (platform->comm.mpiRank == 0)
      nrsCheck(std::isnan(rdotr), MPI_COMM_SELF, EXIT_FAILURE,
               "%s\n", "Detected invalid resiual norm while running linear solver!");

    if (verbose && (platform->comm.mpiRank == 0))
      printf("it %d r norm %.15e\n", iter, rdotr);
  }
  while (rdotr > tol && iter < MAXIT);

  if (iter>=3) {
    triDiagEig(iter, diagt, upper, dmin, dmax);
//    calc_eigen(diagt,upper,iter,dmin,dmax);
    if (verbose && platform->comm.mpiRank == 0)
      printf("%s PCG eigen: dmin %.8e dmax %.8e \n", elliptic->name.c_str(),dmin,dmax);
  }

  return iter;
}
