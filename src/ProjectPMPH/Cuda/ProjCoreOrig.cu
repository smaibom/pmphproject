#include "ProjHelperFun.h"
#include "Constants.h"
#include "TridagPar.h"

#define BLOCK_SIZE 16

void updateParams(const unsigned g, const REAL alpha, const REAL beta, 
                  const REAL nu, PrivGlobs& globs, const int outer)
{
#pragma omp parallel for default(shared) schedule(static) if(outer>8)
  for( unsigned o = 0; o < outer; ++ o )
        {
          for(unsigned i=0;i<globs.numX;++i)
            {
              for(unsigned j=0;j<globs.numY;++j)
                {
                  globs.myVarX[o * globs.numM + i * globs.numY + j] = 
                    exp(2.0*(  beta*log(globs.myX[i])+ globs.myY[j]
                        - 0.5*nu*nu*globs.myTimeline[g]));
                  globs.myVarY[o * globs.numM + i * globs.numY + j] = 
                    exp(2.0*(  alpha*log(globs.myX[i])+ globs.myY[j]
                        - 0.5*nu*nu*globs.myTimeline[g] )); // nu*nu
                }
            }
        }
  REAL* myVarXNew = (REAL*) malloc(sizeof(REAL) * globs.numX * globs.numY * outer);
  transpose(globs.myVarX,myVarXNew,globs.numX,globs.numY,outer);
  free(globs.myVarX);
  globs.myVarX = myVarXNew;
}

void setPayoff(PrivGlobs& globs, unsigned int outer)
{
  unsigned int myR_size = globs.numY * globs.numX;
  for(unsigned int o = 0; o < outer; o++) {
    for(unsigned i=0;i<globs.numX;++i)
      {
        for(unsigned j=0;j<globs.numY;++j)
          {
            globs.myResult[o * myR_size + i * globs.numY + j] = 
              max(globs.myX[i] - o * 0.001, (REAL)0.0);
          }
      }
  }
}

//All arrays are size [n]
inline void tridag(REAL* a,REAL* b,REAL* c,const REAL* r,const int n, 
  REAL* u, REAL* uu)
{
    int    i, offset;
    REAL   beta;

    u[0]  = r[0];
    uu[0] = b[0];

    for(i=1; i<n; i++) 
    {
        beta  = a[i] / uu[i-1];

        uu[i] = b[i] - beta*c[i-1];
        u[i]  = r[i] - beta*u[i-1];
    }

#if 1
    // X) this is a backward recurrence
    u[n-1] = u[n-1] / uu[n-1];
    for(i=n-2; i>=0; i--) 
    {
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
rollback( const unsigned g, PrivGlobs& globs, int outer, const int& numX, 
          const int& numY) 
{
  unsigned numZ = max(numX,numY);
  unsigned numM = numX * numY;

  REAL* u = (REAL*) malloc(sizeof(REAL) * outer * numY * numX);   // [outer][numY][numX]
  REAL* v = (REAL*) malloc(sizeof(REAL) * outer * numX * numY);   // [outer][numX][numY]
  REAL* ax = (REAL*) malloc(sizeof(REAL) * outer * numX * numY); // [outer][numY][numX]
  REAL* bx = (REAL*) malloc(sizeof(REAL) * outer * numX * numY); // [outer][numY][numX]
  REAL* cx = (REAL*) malloc(sizeof(REAL) * outer * numX * numY); // [outer][numY][numX]
  REAL* ay = (REAL*) malloc(sizeof(REAL) * outer * numX * numY); // [outer][numX][numY]
  REAL* by = (REAL*) malloc(sizeof(REAL) * outer * numX * numY); // [outer][numX][numY]
  REAL* cy = (REAL*) malloc(sizeof(REAL) * outer * numX * numY); // [outer][numX][numY]
  REAL* y = (REAL*) malloc(sizeof(REAL) * outer * numX * numY); // [outer][numZ][numZ]
  REAL* yy = (REAL*) malloc(sizeof(REAL)*outer*numZ); // [outer][numZ]

// X-loop
#pragma omp parallel for default(shared) schedule(static) if(outer>8)
  for (int o = 0; o < outer; o++) 
  {
    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);
    for(int j=0;j<numY;j++) 
    {
      for(int i=0;i<numX;i++) 
      {
        // implicit x
        ax[o * numX * numY + j * numX + i] = 
          -0.5*(0.5*globs.myVarX[o * numM + j * numX + i]
                   *globs.myDxx[i * 4 + 0]);
        bx[o * numX * numY + j * numX + i] = 
          dtInv - 0.5*(0.5*globs.myVarX[o * numM + j * numX + i]
                          *globs.myDxx[i * 4 + 1]);
        cx[o * numX * numY + j * numX + i] =
          -0.5*(0.5*globs.myVarX[o * numM + j * numX + i]
                   *globs.myDxx[i * 4 + 2]);
        //  explicit x
        u[o * numX * numY + j * numX + i] = 
          dtInv*globs.myResult[o * numM + i * numY + j];
        if(i > 0) 
        {
          u[o * numX * numY + j * numX + i] += 
            0.5*(0.5*globs.myVarX[o * numM + + j * numX + i]
                    *globs.myDxx[i * 4 + 0])
                    *globs.myResult[o * numM + (i-1) * numY + j];
        }
        u[o * numX * numY + j * numX + i]  +=  
          0.5*(0.5*globs.myVarX[o * numM + + j * numX + i]*globs.myDxx[i * 4 + 1])
                  *globs.myResult[o * numM + i * numY + j];
        if(i < numX-1) 
        {
          u[o * numX * numY + j * numX + i] += 
            0.5*(0.5*globs.myVarX[o * numM + + j * numX + i]
                    *globs.myDxx[i * 4 + 2])
                    *globs.myResult[o * numM + (i+1) * numY + j];
        }
      }
    }
  }
#pragma omp parallel for default(shared) schedule(static) if(outer>8)
  for (int o = 0; o < outer; o++) 
  {
    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);
    unsigned i, j;
    // Y-Loop
    for(i=0;i<numX;i++) 
    {
      for(j=0;j<numY;j++)
        {
          // Explicit y
          v[o * numX * numY + i * numY + j] = 0.0;
          if(j > 0) 
          {
            v[o * numX * numY + i * numY + j] +=  
              (0.5*globs.myVarY[o * numM + i * numY + j]*globs.myDyy[j * 4 + 0])
                  *globs.myResult[o * numM + i * numY + j-1];
          }

          v[o * numX * numY + i * numY + j] += 
            (0.5*globs.myVarY[o * numM + i * numY + j]
                *globs.myDyy[j * 4 + 1])
                *globs.myResult[o * numM + i  * numY + j];

          if(j < numY-1) 
          {
            v[o * numX * numY + i * numY + j] +=  
              (0.5*globs.myVarY[o * numM + i * numY + j]
                  *globs.myDyy[j * 4 + 2])
                  *globs.myResult[o * numM + i * numY + j+1];
          }
          u[o * numX * numY + j * numX + i] += v[o * numX * numY + i * numY + j];
          // Implicit y
          ay[o * numX * numY + i * numY + j] =
            -0.5*(0.5*globs.myVarY[o * numM + i * numY + j]
                     *globs.myDyy[j * 4 + 0]);
          by[o * numX * numY + i * numY + j] = 
            dtInv - 0.5*(0.5*globs.myVarY[o * numM + i * numY + j]
                            *globs.myDyy[j * 4 + 1]);
          cy[o * numX * numY + i * numY + j] =
            -0.5*(0.5*globs.myVarY[o * numM + i * numY + j]
                     *globs.myDyy[j * 4 + 2]);
        }
    }
  }
#pragma omp parallel for default(shared) schedule(static) if(outer>8)
  for(int o = 0; o < outer; o++) 
  {
    for(int j=0;j<numY;j++) 
      {
      // here yy should have size [numX]
      tridagPar(&ax[o * numX * numY + j * numX],&bx[o * numX * numY + j * numX],
                &cx[o * numX * numY + j * numX], &u[o * numX * numY + j * numX], 
                numX, &u[o * numX * numY + numX * j], &yy[o*numZ]);
    }
  }

  //  implicit y
#pragma omp parallel for default(shared) schedule(static) if(outer>8)
  for(int o = 0; o < outer; o++)
  {
    REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);
    for(int i=0;i<numX;i++)
    {
      for(int j=0;j<numY;j++)
      {  // here a, b, c should have size [numY]
        y[o * numX * numY + i * numY + j] =
          dtInv*u[o * numX * numY + j * numX + i]
           -0.5*v[o * numX * numY + i * numY + j];
      }
    }
  }
#pragma omp parallel for default(shared) schedule(static) if(outer>8)
  for(int o = 0; o < outer; o++)
  {
    for(int i=0;i<numX;i++)
    {
      // here yy should have size [numY]
      tridagPar(&ay[o * numX * numY + i * numY],&by[o * numX * numY + i * numY],
                &cy[o * numX * numY + i * numY],&y[o * numX * numY + i * numY],
                numY,&globs.myResult[o * numM + i * numY],&yy[o*numZ]);
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



void   run_OrigCPU(const unsigned int& outer,const unsigned int& numX,
                   const unsigned int& numY,const unsigned int& numT,
                   const REAL& s0,const REAL& t,const REAL& alpha,
                   const REAL& nu,const REAL& beta,REAL* res) // [outer] RESULT
{
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
  for (unsigned int i = 0; i < outer; i++) 
  {
    res[i] = globals.myResult[i * globals.numM + globals.myXindex * numY + globals.myYindex];
  }
}

//Kernels

__global__ void rollback_x(REAL* ax, REAL* bx, REAL* cx, REAL* u, REAL* myVarX, REAL* myDxx, REAL* myResult,
                           REAL dtInv, int numX, int numY) {

  int i = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  int j = BLOCK_SIZE * blockIdx.y + threadIdx.y;
  int o = blockIdx.z;

  int numM = numY * numX;

  ax[o * numX * numY + j * numX + i] = -0.5*(0.5*myVarX[o * numM + j * numX + i]*myDxx[i * 4 + 0]);

  bx[o * numX * numY + j * numX + i] =
    dtInv - 0.5*(0.5*myVarX[o * numM + j * numX + i]
                 *myDxx[i * 4 + 1]);
  cx[o * numX * numY + j * numX + i] =
    -0.5*(0.5*myVarX[o * numM + j * numX + i]
          *myDxx[i * 4 + 2]);
  //  explicit x
  u[o * numX * numY + j * numX + i] =
    dtInv*myResult[o * numM + i * numY + j];

  if(i > 0) {
      u[o * numX * numY + j * numX + i] +=
        0.5*(0.5*myVarX[o * numM + + j * numX + i]
             *myDxx[i * 4 + 0])
        *myResult[o * numM + (i-1) * numY + j];
    }

  u[o * numX * numY + j * numX + i]  +=
    0.5*(0.5*myVarX[o * numM + + j * numX + i]*myDxx[i * 4 + 1])
    *myResult[o * numM + i * numY + j];

  if(i < numX-1) {
      u[o * numX * numY + j * numX + i] +=
        0.5*(0.5*myVarX[o * numM + + j * numX + i]
             *myDxx[i * 4 + 2])
        *myResult[o * numM + (i+1) * numY + j];
    }


}



//#endif // PROJ_CORE_ORIG