#ifndef PROJ_HELPER_FUNS
#define PROJ_HELPER_FUNS

#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "../include/Constants.h"
#include <cuda_runtime.h>

using namespace std;


struct PrivGlobs {
  // Array dimensions
  unsigned int numX;
  unsigned int numY;
  unsigned int numT;
  unsigned int numM; // size of the expanded matrices

  //	grid
  REAL*        myX;        // [numX]
  REAL*        dmyX;
  REAL*        myY;        // [numY]
  REAL*        dmyY;
  REAL*        myTimeline; // [numT]
  REAL*        dmyTimeline;
  unsigned            myXindex;
  unsigned            myYindex;

  //	variable
  REAL*   myResult; // [outer][numX][numY]
  REAL*   dmyResult; // [outer][numX][numY]

  //	coeffs
  REAL*   myVarX; // [outer][numX][numY]
  REAL*   dmyVarX; // [outer][numX][numY]
  REAL*   myVarY; // [outer][numX][numY]
  REAL*   dmyVarY; // [outer][numX][numY]

  //	operators
  REAL*  myDxx;  // [numX][4]
  REAL*  dmyDxx;  // [numX][4]
  REAL*  myDyy;  // [numY][4]
  REAL*  dmyDyy;  // [numY][4]

  // x and y, abc arrays

  REAL* dax;
  REAL* dbx;
  REAL* dcx;

  // u, v and uu
  REAL* du;

  PrivGlobs( ) {
    printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
    exit(0);
  }

  PrivGlobs(  const unsigned int& numX,
              const unsigned int& numY,
              const unsigned int& numT,
              const unsigned int& outer) {
    this->numX = numX;
    this->numY = numY;
    this->numT = numT;
    this->numM = numX * numY;
    this->myX = (REAL*) malloc(sizeof(REAL) * numX);
    cudaMalloc((void**)&this->dmyX, numX * sizeof(REAL));
    this->myDxx = (REAL*) malloc(sizeof(REAL) * numX * 4);
    this->myY  = (REAL*) malloc(sizeof(REAL) * numY);
    cudaMalloc((void**)&this->dmyY, numY * sizeof(REAL));
    this->myDyy = (REAL*) malloc(sizeof(REAL) * numY * 4);

    this->myTimeline = (REAL*) malloc(sizeof(REAL) * numT);
    cudaMalloc((void**)&this->dmyTimeline, numT * sizeof(REAL));

    this->  myVarX = (REAL*) malloc(sizeof(REAL) * numX * numY * outer);
    this->  myVarY = (REAL*) malloc(sizeof(REAL) * numX * numY * outer);
    cudaMalloc((void**)&this->dmyVarX, outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**)&this->dmyVarY, outer * numX * numY * sizeof(REAL));

    this->myResult = (REAL*) malloc(sizeof(REAL) * numX * numY * outer);
    cudaMalloc((void**)&this->dmyResult, outer * numX * numY * sizeof(REAL));

    //x and y abc
    cudaMalloc((void**)&this->dax, outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**)&this->dbx, outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**)&this->dcx, outer * numX * numY * sizeof(REAL));
    cudaMalloc((void**)&this->du, outer * numX * numY * sizeof(REAL));
  } 
} __attribute__ ((aligned (128)));


void initGrid(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
                const unsigned numX, const unsigned numY, const unsigned numT, PrivGlobs& globs   
            );

void initOperator(  REAL* x,
                    const unsigned int x_size,
                    REAL* Dxx
                 );
void transpose(REAL*,REAL*,int,int,int);

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs);

void setPayoff(const REAL strike, PrivGlobs& globs );

void tridag(
    const vector<REAL>&   a,   // size [n]
    const vector<REAL>&   b,   // size [n]
    const vector<REAL>&   c,   // size [n]
    const vector<REAL>&   r,   // size [n]
    const int             n,
          vector<REAL>&   u,   // size [n]
          vector<REAL>&   uu   // size [n] temporary
);

void rollback( const unsigned g, PrivGlobs& globs );

REAL   value(   PrivGlobs    globs,
                const REAL s0,
                const REAL strike,
                const REAL t,
                const REAL alpha,
                const REAL nu,
                const REAL beta,
                const unsigned int numX,
                const unsigned int numY,
                const unsigned int numT
            );

void run_OrigCPU(
                const unsigned int&   outer,
                const unsigned int&   numX,
                const unsigned int&   numY,
                const unsigned int&   numT,
                const REAL&           s0,
                const REAL&           t,
                const REAL&           alpha,
                const REAL&           nu,
                const REAL&           beta,
                      REAL*           res   // [outer] RESULT
            );

#endif // PROJ_HELPER_FUNS
