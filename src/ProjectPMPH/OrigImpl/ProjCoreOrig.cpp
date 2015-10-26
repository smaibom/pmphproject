#include "ProjHelperFun.h"
#include "Constants.h"


void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs, const int outer)
{

#pragma omp parallel for default(shared) schedule(static) if(outer>8)
  for( unsigned o = 0; o < outer; ++ o )
        {
          for(unsigned i=0;i<globs.numX;++i)
            {
              for(unsigned j=0;j<globs.numY;++j)
                {
                  globs.myVarX[o * globs.numM + i * globs.numY + j] = exp(2.0*(  beta*log(globs.myX[i])
                                                                      + globs.myY[j]
                                                                      - 0.5*nu*nu*globs.myTimeline[g] )
                                              );
                  globs.myVarY[o * globs.numM + i * globs.numY + j] = exp(2.0*(  alpha*log(globs.myX[i])
                                                                      + globs.myY[j]
                                                                      - 0.5*nu*nu*globs.myTimeline[g] )
                                              ); // nu*nu
                }
            }
        }

}

void setPayoff(PrivGlobs& globs, unsigned int outer)
{
  unsigned int myR_size = globs.numY * globs.numX;
  for(unsigned int o = 0; o < outer; o++) {
    for(unsigned i=0;i<globs.numX;++i)
      {
        for(unsigned j=0;j<globs.numY;++j)
          {
            globs.myResult[o * myR_size + i * globs.numY + j] = max(globs.myX[i] - o * 0.001, (REAL)0.0);
          }
      }
  }
}

inline void tridag(
          REAL*   a,   // size [n]
          REAL*   b,   // size [n]
          REAL*   c,   // size [n]
    const REAL*   r,   // size [n]
    const int             n,
          REAL*   u,   // size [n]
          REAL*   uu   // size [n] temporary
) {
    int    i, offset;
    REAL   beta;

    u[0]  = r[0];
    uu[0] = b[0];

    for(i=1; i<n; i++) {
        beta  = a[i] / uu[i-1];

        uu[i] = b[i] - beta*c[i-1];
        u[i]  = r[i] - beta*u[i-1];
    }

#if 1
    // X) this is a backward recurrence
    u[n-1] = u[n-1] / uu[n-1];
    for(i=n-2; i>=0; i--) {
        u[i] = (u[i] - c[i]*u[i+1]) / uu[i];
    }
#else
    // Hint: X) can be written smth like (once you make a non-constant)
    for(i=0; i<n; i++) a[i] =  u[n-1-i];
    a[0] = a[0] / uu[n-1];
    for(i=1; i<n; i++) a[i] = (a[i] - c[n-1-i]*a[i-1]) / uu[n-1-i];
    for(i=0; i<n; i++) u[i] = a[n-1-i];
#endif
}


void
rollback( const unsigned g, PrivGlobs& globs, int outer, const int& numX,  const int& numY) {

  unsigned numZ = max(numX,numY);
  unsigned numM = numX * numY;

  REAL* u = (REAL*) malloc(sizeof(REAL) * outer * numY * numX);   // [outer][numY][numX]
  //vector<vector<vector<REAL> > > v(outer, vector<vector<REAL> > (numX, vector<REAL>(numY)));   // [outer][numX][numY]
  REAL* v = (REAL*) malloc(sizeof(REAL) * outer * numX * numY);
  //vector<vector<vector<REAL> > > ax(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ)));   // [outer][max(numX,numY)][max(numX, numY)]
  REAL* ax = (REAL*) malloc(sizeof(REAL) * outer * numZ * numZ);
  //vector<vector<vector<REAL> > > bx(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ)));   // [outer][max(numX,numY)][max(numX, numY)]
  REAL* bx = (REAL*) malloc(sizeof(REAL) * outer * numZ * numZ); 
  //vector<vector<vector<REAL> > > cx(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ)));   // [outer][max(numX,numY)][max(numX, numY)]
  REAL* cx = (REAL*) malloc(sizeof(REAL) * outer * numZ * numZ); 
  //vector<vector<vector<REAL> > > ay(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ)));   // [outer][max(numX,numY)][max(numX, numY)]
  REAL* ay = (REAL*) malloc(sizeof(REAL) * outer * numZ * numZ);
  //vector<vector<vector<REAL> > > by(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ)));   // [outer][max(numX,numY)][max(numX, numY)]
  REAL* by = (REAL*) malloc(sizeof(REAL) * outer * numZ * numZ);
  //vector<vector<vector<REAL> > > cy(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ)));   // [outer][max(numX,numY)][max(numX, numY)]
  REAL* cy = (REAL*) malloc(sizeof(REAL) * outer * numZ * numZ);  

  //vector<vector<vector<REAL> > > y(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ)));   // [outer][max(numX,numY)][max(numX, numY)]
  REAL* y = (REAL*) malloc(sizeof(REAL) * outer * numZ * numZ);
  //vector<vector<REAL> > yy(outer, vector<REAL>(numZ));  // temporary used in tridag  // [outer][max(numX,numY)]
  REAL* yy = (REAL*) malloc(sizeof(REAL)*outer*numZ);



  // X-loop
#pragma omp parallel for default(shared) schedule(static) if(outer>8)
  for (int o = 0; o < outer; o++) {
    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);
    for(int j=0;j<numY;j++) {
      for(int i=0;i<numX;i++) {
        // implicit x
        ax[o * numZ * numZ + j * numZ + i] = -0.5*(0.5*globs.myVarX[o * numM + i * numY + j]*globs.myDxx[i * 4 + 0]);
        //ax[o][j][i] =		 - 0.5*(0.5*globs.myVarX[o * numM + i * numY + j]*globs.myDxx[i * 4 + 0]);
        //bx[o][j][i] = dtInv - 0.5*(0.5*globs.myVarX[o * numM + i * numY + j]*globs.myDxx[i * 4 + 1]);
        bx[o * numZ * numZ + j * numZ + i] = dtInv - 0.5*(0.5*globs.myVarX[o * numM + i * numY + j]*globs.myDxx[i * 4 + 1]);
        cx[o * numZ * numZ + j * numZ + i] =		 - 0.5*(0.5*globs.myVarX[o * numM + i * numY + j]*globs.myDxx[i * 4 + 2]);


        //	explicit x
        u[o * numX * numY + j * numX + i] = dtInv*globs.myResult[o * numM + i * numY + j];

        if(i > 0) {
          u[o * numX * numY + j * numX + i] += 0.5*( 0.5*globs.myVarX[o * numM + i * numY + j]*globs.myDxx[i * 4 + 0] )
            * globs.myResult[o * numM + (i-1) * numY + j];
        }
        u[o * numX * numY + j * numX + i]  +=  0.5*( 0.5*globs.myVarX[o * numM + i * numY + j]*globs.myDxx[i * 4 + 1] )
          * globs.myResult[o * numM + i * numY + j];
        if(i < numX-1) {
          u[o * numX * numY + j * numX + i] += 0.5*( 0.5*globs.myVarX[o * numM + i * numY + j]*globs.myDxx[i * 4 + 2] )
            * globs.myResult[o * numM + (i+1) * numY + j];
        }
      }
    }
  }
#pragma omp parallel for default(shared) schedule(static) if(outer>8)
  for (int o = 0; o < outer; o++) {
    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);
    unsigned i, j;
    // Y-Loop
    for(i=0;i<numX;i++) {
      for(j=0;j<numY;j++)
        {
          // Explicit y
          v[o * numX * numY + i * numY + j] = 0.0;

          if(j > 0) {
            v[o * numX * numY + i * numY + j] +=  ( 0.5*globs.myVarY[o * numM + i * numY + j]*globs.myDyy[j * 4 + 0] )
              *  globs.myResult[o * numM + i * numY + j-1];
          }
          v[o * numX * numY + i * numY + j]  +=   ( 0.5*globs.myVarY[o * numM + i * numY + j]*globs.myDyy[j * 4 + 1] )
            *  globs.myResult[o * numM + i  * numY + j];
          if(j < numY-1) {
            v[o * numX * numY + i * numY + j] +=  ( 0.5*globs.myVarY[o * numM + i * numY + j]*globs.myDyy[j * 4 + 2] )
              *  globs.myResult[o * numM + i * numY + j+1];
          }
          u[o * numX * numY + j * numX + i] += v[o * numX * numY + i * numY + j];

          // Implicit y
          ay[o * numZ * numZ + i * numZ + j] =		 - 0.5*(0.5*globs.myVarY[o * numM + i * numY + j]*globs.myDyy[j * 4 + 0]);
          by[o * numZ * numZ + i * numZ + j] = dtInv - 0.5*(0.5*globs.myVarY[o * numM + i * numY + j]*globs.myDyy[j * 4 + 1]);
          cy[o * numZ * numZ + i * numZ + j] =		 - 0.5*(0.5*globs.myVarY[o * numM + i * numY + j]*globs.myDyy[j * 4 + 2]);

        }
    }
  }
#pragma omp parallel for default(shared) schedule(static) if(outer>8)
  for(int o = 0; o < outer; o++) {
    for(int j=0;j<numY;j++) {
      // here yy should have size [numX]
      tridag(&ax[o * numZ * numZ + j * numZ],&bx[o * numZ * numZ + j * numZ],
             &cx[o * numZ * numZ + j * numZ], &u[o * numX * numY + j * numX], 
             numX, &u[o * numX * numY + numX * j], &yy[o*numZ]);
    }
  }

  //	implicit y
#pragma omp parallel for default(shared) schedule(static) if(outer>8)
  for(int o = 0; o < outer; o++) {
    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);
    for(int i=0;i<numX;i++) {
      for(int j=0;j<numY;j++) {  // here a, b, c should have size [numY]
        y[o * numZ * numZ + i * numZ +j] = dtInv*u[o * numX * numY + j * numX + i] - 0.5*v[o * numX * numY + i * numY + j];
      }
    }
  }
#pragma omp parallel for default(shared) schedule(static) if(outer>8)
  for(int o = 0; o < outer; o++) {
    for(int i=0;i<numX;i++) {
      // here yy should have size [numY]
      tridag(&ay[o * numZ * numZ + i * numZ],&by[o * numZ * numZ + i * numZ],
             &cy[o * numZ * numZ + i * numZ],&y[o * numZ * numZ + i * numZ],numY, 
             &globs.myResult[o * numM + i * numY],&yy[o*numZ]);
    }
  }
  free(u);
  free(ax);
  free(ay);
  free(bx);
  free(by);
  free(cx);
  free(cy);
  free(yy);
  free(y);
}



void   run_OrigCPU(
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
                   ) {
  PrivGlobs globals(numX, numY, numT, outer);
  initGrid(s0,alpha,nu,t, numX, numY, numT, globals);
  initOperator(globals.myX, globals.numX, globals.myDxx);
  initOperator(globals.myY, globals.numY, globals.myDyy);


  setPayoff(globals, outer);


  for(int g = numT-2;g>=0;--g)
    {
      updateParams(g,alpha,beta,nu,globals, outer);
      rollback(g, globals, outer, numX, numY);
    }
  for (unsigned int i = 0; i < outer; i++) {
    res[i] = globals.myResult[i * globals.numM + globals.myXindex * numY + globals.myYindex];
  }
}

//#endif // PROJ_CORE_ORIG
