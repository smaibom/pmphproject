#include "ProjHelperFun.h"
#include "Constants.h"


void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs* globs, const int outer)
{

#pragma omp parallel for default(shared) schedule(static) if(outer>8)
  for( unsigned o = 0; o < outer; ++ o )
        {
          for(unsigned i=0;i<globs[o].myX.size();++i)
            {
              for(unsigned j=0;j<globs[o].myY.size();++j)
                {
                  globs[o].myVarX[i][j] = exp(2.0*(  beta*log(globs[o].myX[i])
                                                     + globs[o].myY[j]
                                                     - 0.5*nu*nu*globs[o].myTimeline[g] )
                                              );
                  globs[o].myVarY[i][j] = exp(2.0*(  alpha*log(globs[o].myX[i])
                                                     + globs[o].myY[j]
                                                     - 0.5*nu*nu*globs[o].myTimeline[g] )
                                              ); // nu*nu
                }
            }
        }

}

void setPayoff(const REAL strike, PrivGlobs& globs )
{
  REAL payoff[globs.myX.size()];
  for (unsigned i=0;i<globs.myX.size();++i) {
    payoff[i] = max(globs.myX[i]-strike, (REAL)0.0);
  }

  for(unsigned i=0;i<globs.myX.size();++i)
	{
      for(unsigned j=0;j<globs.myY.size();++j)
        {
          globs.myResult[i][j] = payoff[i];
        }
    }
}

inline void tridag(
    const vector<REAL>&   a,   // size [n]
    const vector<REAL>&   b,   // size [n]
    const vector<REAL>&   c,   // size [n]
    const vector<REAL>&   r,   // size [n]
    const int             n,
          vector<REAL>&   u,   // size [n]
          vector<REAL>&   uu   // size [n] temporary
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
rollback( const unsigned g, PrivGlobs* globs, int outer, const int& numX,  const int& numY) {


  unsigned numZ = max(numX,numY);

  vector<vector<vector<REAL> > > u(outer, vector<vector<REAL> > (numY, vector<REAL>(numX)));   // [numY][numX]
  vector<vector<vector<REAL> > > v(outer, vector<vector<REAL> > (numX, vector<REAL>(numY)));   // [numX][numY]

  vector<vector<vector<REAL> > > ax(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ)));   // [max(numX,numY)][max(numX, numY)]
  vector<vector<vector<REAL> > > bx(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ)));   // [max(numX,numY)][max(numX, numY)]
  vector<vector<vector<REAL> > > cx(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ)));   // [max(numX,numY)][max(numX, numY)]
  vector<vector<vector<REAL> > > ay(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ)));   // [max(numX,numY)][max(numX, numY)]
  vector<vector<vector<REAL> > > by(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ)));   // [max(numX,numY)][max(numX, numY)]
  vector<vector<vector<REAL> > > cy(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ)));   // [max(numX,numY)][max(numX, numY)]


  vector<vector<vector<REAL> > > y(outer, vector<vector<REAL> > (numZ, vector<REAL>(numZ)));   // [max(numX,numY)][max(numX, numY)]
  vector<vector<REAL> > yy(outer, vector<REAL>(numZ));  // temporary used in tridag  // [max(numX,numY)]



  // X-loop
#pragma omp parallel for default(shared) schedule(static) if(outer>8)
  for (int o = 0; o < outer; o++) {
    REAL dtInv = 1.0/(globs[o].myTimeline[g+1]-globs[o].myTimeline[g]);
    for(int j=0;j<numY;j++) {
      for(int i=0;i<numX;i++) {
        // implicit x
        ax[o][j][i] =		 - 0.5*(0.5*globs[o].myVarX[i][j]*globs[o].myDxx[i][0]);
        bx[o][j][i] = dtInv - 0.5*(0.5*globs[o].myVarX[i][j]*globs[o].myDxx[i][1]);
        cx[o][j][i] =		 - 0.5*(0.5*globs[o].myVarX[i][j]*globs[o].myDxx[i][2]);


        //	explicit x
        u[o][j][i] = dtInv*globs[o].myResult[i][j];

        if(i > 0) {
          u[o][j][i] += 0.5*( 0.5*globs[o].myVarX[i][j]*globs[o].myDxx[i][0] )
            * globs[o].myResult[i-1][j];
        }
        u[o][j][i]  +=  0.5*( 0.5*globs[o].myVarX[i][j]*globs[o].myDxx[i][1] )
          * globs[o].myResult[i][j];
        if(i < numX-1) {
          u[o][j][i] += 0.5*( 0.5*globs[o].myVarX[i][j]*globs[o].myDxx[i][2] )
            * globs[o].myResult[i+1][j];
        }
      }
    }
  }
#pragma omp parallel for default(shared) schedule(static) if(outer>8)
  for (int o = 0; o < outer; o++) {
    REAL dtInv = 1.0/(globs[o].myTimeline[g+1]-globs[o].myTimeline[g]);
    unsigned i, j;
    // Y-Loop
    for(i=0;i<numX;i++) {
      for(j=0;j<numY;j++)
        {
          // Explicit y
          v[o][i][j] = 0.0;

          if(j > 0) {
            v[o][i][j] +=  ( 0.5*globs[o].myVarY[i][j]*globs[o].myDyy[j][0] )
              *  globs[o].myResult[i][j-1];
          }
          v[o][i][j]  +=   ( 0.5*globs[o].myVarY[i][j]*globs[o].myDyy[j][1] )
            *  globs[o].myResult[i][j];
          if(j < numY-1) {
            v[o][i][j] +=  ( 0.5*globs[o].myVarY[i][j]*globs[o].myDyy[j][2] )
              *  globs[o].myResult[i][j+1];
          }
          u[o][j][i] += v[o][i][j];

          // Implicit y
          ay[o][i][j] =		 - 0.5*(0.5*globs[o].myVarY[i][j]*globs[o].myDyy[j][0]);
          by[o][i][j] = dtInv - 0.5*(0.5*globs[o].myVarY[i][j]*globs[o].myDyy[j][1]);
          cy[o][i][j] =		 - 0.5*(0.5*globs[o].myVarY[i][j]*globs[o].myDyy[j][2]);

        }
    }
  }
#pragma omp parallel for default(shared) schedule(static) if(outer>8)
  for(int o = 0; o < outer; o++) {
    for(int j=0;j<numY;j++) {
      // here yy should have size [numX]
      tridag(ax[o][j],bx[o][j],cx[o][j],u[o][j],numX,u[o][j], yy[o]);
    }
  }

  //	implicit y
#pragma omp parallel for default(shared) schedule(static) if(outer>8)
  for(int o = 0; o < outer; o++) {
    REAL dtInv = 1.0/(globs[o].myTimeline[g+1]-globs[o].myTimeline[g]);
    for(int i=0;i<numX;i++) {
      for(int j=0;j<numY;j++) {  // here a, b, c should have size [numY]
        y[o][i][j] = dtInv*u[o][j][i] - 0.5*v[o][i][j];
      }
    }
  }
#pragma omp parallel for default(shared) schedule(static) if(outer>8)
  for(int o = 0; o < outer; o++) {
    for(int i=0;i<numX;i++) {
      // here yy should have size [numY]
      tridag(ay[o][i],by[o][i],cy[o][i],y[o][i],numY,globs[o].myResult[i],yy[o]);
    }
  }
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
  PrivGlobs* globals = (PrivGlobs*) malloc(outer * sizeof(PrivGlobs));
  for (unsigned int i = 0; i < outer; i++) {
    globals[i] = PrivGlobs(numX, numY, numT);
    initGrid(s0,alpha,nu,t, numX, numY, numT, globals[i]);
    initOperator(globals[i].myX,globals[i].myDxx);
    initOperator(globals[i].myY,globals[i].myDyy);

    setPayoff(0.001 * i, globals[i]);
  }


  for(int g = numT-2;g>=0;--g)
    {
      updateParams(g,alpha,beta,nu,globals, outer);
      rollback(g, globals, outer, numX, numY);
    }
  for (unsigned int i = 0; i < outer; i++) {
    res[i] = globals[i].myResult[globals[i].myXindex][globals[i].myYindex];
  }
}

//#endif // PROJ_CORE_ORIG
