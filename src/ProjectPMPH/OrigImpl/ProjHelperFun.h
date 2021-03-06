#ifndef PROJ_HELPER_FUNS
#define PROJ_HELPER_FUNS

#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Constants.h"

using namespace std;


struct PrivGlobs {
  // Array dimensions
  unsigned int numX;
  unsigned int numY;
  unsigned int numT;
  unsigned int numM; // size of the expanded matrices

  //	grid
  REAL*        myX;        // [numX]
  REAL*        myY;        // [numY]
  REAL*        myTimeline; // [numT]
  unsigned            myXindex;
  unsigned            myYindex;

  //	variable
  REAL*   myResult; // [outer][numX][numY]

  //	coeffs
  REAL*   myVarX; // [outer][numX][numY]
  REAL*   myVarY; // [outer][numX][numY]

  //	operators
  REAL*  myDxx;  // [numX][4]
  REAL*  myDyy;  // [numY][4]

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
    this->myDxx = (REAL*) malloc(sizeof(REAL) * numX * 4);
    this->myY  = (REAL*) malloc(sizeof(REAL) * numY);

    this->myDyy = (REAL*) malloc(sizeof(REAL) * numY * 4);

    this->myTimeline = (REAL*) malloc(sizeof(REAL) * numT);

    this->  myVarX = (REAL*) malloc(sizeof(REAL) * numX * numY * outer);
    this->  myVarY = (REAL*) malloc(sizeof(REAL) * numX * numY * outer);
    this->myResult = (REAL*) malloc(sizeof(REAL) * numX * numY * outer);

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
