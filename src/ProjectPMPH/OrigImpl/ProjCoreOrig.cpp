#include "ProjHelperFun.h"
#include "Constants.h"


void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs, const int outer)
{
    for(unsigned i=0;i<globs.myX.size();++i)
        for(unsigned j=0;j<globs.myY.size();++j) {
            globs.myVarX[i][j] = exp(2.0*(  beta*log(globs.myX[i])
                                          + globs.myY[j]
                                          - 0.5*nu*nu*globs.myTimeline[g] )
                                    );
            globs.myVarY[i][j] = exp(2.0*(  alpha*log(globs.myX[i])
                                          + globs.myY[j]
                                          - 0.5*nu*nu*globs.myTimeline[g] )
                                    ); // nu*nu
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
rollback( const unsigned g, PrivGlobs& globs ) {
    unsigned numX = globs.myX.size(),
             numY = globs.myY.size();

    unsigned numZ = max(numX,numY);

    unsigned i, j;

    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);

    vector<vector<REAL> > u(numY, vector<REAL>(numX));   // [numY][numX]
    vector<vector<REAL> > v(numX, vector<REAL>(numY));   // [numX][numY]
    vector<vector<REAL> > ax(numZ, vector<REAL>(numZ));   // [max(numX,numY)][max(numX, numY)]
    vector<vector<REAL> > bx(numZ, vector<REAL>(numZ));   // [max(numX,numY)][max(numX, numY)]
    vector<vector<REAL> > cx(numZ, vector<REAL>(numZ));   // [max(numX,numY)][max(numX, numY)]

    vector<vector<REAL> > ay(numZ, vector<REAL>(numZ));   // [max(numX,numY)][max(numX, numY)]
    vector<vector<REAL> > by(numZ, vector<REAL>(numZ));   // [max(numX,numY)][max(numX, numY)]
    vector<vector<REAL> > cy(numZ, vector<REAL>(numZ));   // [max(numX,numY)][max(numX, numY)]

    vector<vector<REAL> > y(numZ, vector<REAL>(numZ));   // [max(numX,numY)][max(numX, numY)]
    vector<REAL> yy(numZ);  // temporary used in tridag  // [max(numX,numY)]

    // X-loop
    for(j=0;j<numY;j++) {
      for(i=0;i<numX;i++) {
            // implicit x
            ax[j][i] =		 - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][0]);
            bx[j][i] = dtInv - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][1]);
            cx[j][i] =		 - 0.5*(0.5*globs.myVarX[i][j]*globs.myDxx[i][2]);


            //	explicit x
            u[j][i] = dtInv*globs.myResult[i][j];

            if(i > 0) {
              u[j][i] += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][0] )
                            * globs.myResult[i-1][j];
            }
            u[j][i]  +=  0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][1] )
                            * globs.myResult[i][j];
            if(i < numX-1) {
              u[j][i] += 0.5*( 0.5*globs.myVarX[i][j]*globs.myDxx[i][2] )
                            * globs.myResult[i+1][j];
            }
        }
    }

    // Y-Loop
    for(i=0;i<numX;i++) {
          for(j=0;j<numY;j++)
            {
              // Explicit y
              v[i][j] = 0.0;

              if(j > 0) {
                v[i][j] +=  ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][0] )
                  *  globs.myResult[i][j-1];
              }
              v[i][j]  +=   ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][1] )
                *  globs.myResult[i][j];
              if(j < numY-1) {
                v[i][j] +=  ( 0.5*globs.myVarY[i][j]*globs.myDyy[j][2] )
                  *  globs.myResult[i][j+1];
              }
              u[j][i] += v[i][j];

              // Implicit y
              ay[i][j] =		 - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][0]);
              by[i][j] = dtInv - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][1]);
              cy[i][j] =		 - 0.5*(0.5*globs.myVarY[i][j]*globs.myDyy[j][2]);

        }
    }

    for(j=0;j<numY;j++) {
        // here yy should have size [numX]
        tridag(ax[j],bx[j],cx[j],u[j],numX,u[j],yy);
    }

    //	implicit y
    for(i=0;i<numX;i++) {
        for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]
          y[i][j] = dtInv*u[j][i] - 0.5*v[i][j];
        }
    }

    for(i=0;i<numX;i++) {
        // here yy should have size [numY]
        tridag(ay[i],by[i],cy[i],y[i],numY,globs.myResult[i],yy);
    }
}

// REAL   value(   PrivGlobs    globs,
//                 const REAL s0,
//                 const REAL strike,
//                 const REAL t,
//                 const REAL alpha,
//                 const REAL nu,
//                 const REAL beta,
//                 const unsigned int numX,
//                 const unsigned int numY,
//                 const unsigned int numT
// ) {
//     initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
//     initOperator(globs.myX,globs.myDxx);
//     initOperator(globs.myY,globs.myDyy);

//     setPayoff(strike, globs);

//   for(int i = numT-2;i>=0;--i)
//    {
//      updateParams(i,alpha,beta,nu,globs);
//      rollback(i, globs);
//    }


//     return globs.myResult[globs.myXindex][globs.myYindex];
// }

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
#pragma omp parallel for default(shared) schedule(static) if(outer>8)
      for( unsigned i = 0; i < outer; ++ i )
        {
          updateParams(g,alpha,beta,nu,globals[i], outer);
          rollback(g, globals[i]);
        }
    }
  for (unsigned int i = 0; i < outer; i++) {
    res[i] = globals[i].myResult[globals[i].myXindex][globals[i].myYindex];
  }
}

//#endif // PROJ_CORE_ORIG
