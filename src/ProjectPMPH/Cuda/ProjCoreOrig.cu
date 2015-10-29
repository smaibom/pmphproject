#include "ProjHelperFun.cu.h"
#include "Constants.h"
#include "TridagPar.h"

#define BLOCK_SIZE 32



__global__ void updateParamsKernel(const unsigned g, const REAL alpha, 
                                   const REAL beta, const REAL nu, REAL* myVarX,
                                   REAL* myVarY, REAL* myY, REAL* myX, 
                                   REAL* myTimeline,const int numY, const int numM){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    myVarX[z*numM+i*numY+j] = exp(2.0*(beta*log(myX[i])+myY[j]-0.5*nu*nu*myTimeline[g]));
    myVarY[z* numM + i * numY + j] =exp(2.0*(  alpha*log(myX[i])+myY[j]- 0.5*nu*nu*myTimeline[g] ));
}

void updateParams(const unsigned g, const REAL alpha, const REAL beta, 
                  const REAL nu, PrivGlobs& globs, const int outer)
{
  int numX = globs.numX;
  int numY = globs.numY;
  int numT = globs.numT;
  int numM = numX*numY;

  //Device memory
  cudaMemcpy(globs.dmyY, globs.myY, numY * sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(globs.dmyTimeline, globs.myTimeline, numT * sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(globs.dmyX, globs.myX, numX * sizeof(REAL), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks(numX / BLOCK_SIZE, numY / BLOCK_SIZE, outer);
  updateParamsKernel<<< numBlocks, threadsPerBlock >>> (g, alpha, beta, nu, globs.dmyVarX, 
                                                        globs.dmyVarY, globs.dmyY,globs.dmyX,globs.dmyTimeline,numY,numX*numY);

  cudaMemcpy(globs.myVarX, globs.dmyVarX, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(globs.myVarY, globs.dmyVarY, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
  REAL* myVarXNew = (REAL*) malloc(sizeof(REAL) * globs.numX * globs.numY * outer);
  transpose(globs.myVarX,myVarXNew,globs.numX,globs.numY,outer);
  free(globs.myVarX);
  globs.myVarX = myVarXNew;
}

__global__ void setPayoffKernel(REAL* myX, REAL*   myResult, unsigned int numX, unsigned int numY) {
    int i = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    int j = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    int o = blockIdx.z;
  
    myResult[o * (numY * numX) + i * numY + j] = 
      max(myX[i] - o * 0.001, (REAL)0.0);
}

void setPayoff_cuda(PrivGlobs& globs, unsigned int outer)
{ 
    REAL* myResult_d;
  cudaMemcpy(globs.dmyX, globs.myX, globs.numX*sizeof(REAL ), cudaMemcpyHostToDevice);
  
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(globs.numX / BLOCK_SIZE, globs.numY / BLOCK_SIZE, outer);
  
  //kernel
    setPayoffKernel<<<numBlocks, threadsPerBlock>>>(globs.dmyX, globs.dmyResult, globs.numX, globs.numY);

  cudaMemcpy(globs.myResult, globs.dmyResult, outer*globs.numX*globs.numY*sizeof(REAL), cudaMemcpyDeviceToHost);
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


__global__ void rollback_implicit_y (REAL* y, REAL*  u, REAL* v, REAL dtInv, int numX, int numY) {
  int j = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  int i = BLOCK_SIZE * blockIdx.y + threadIdx.y;
  int o = blockIdx.z;

  y[o * numX * numY + i * numY + j] = dtInv*u[o * numX * numY + j * numX + i]
    - 0.5*v[o * numX * numY + i * numY + j];

}


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


__global__ void rollback_y(REAL* ay, REAL* by, REAL* cy, REAL* u, REAL* v, REAL* myVarY, REAL* myDyy, REAL* myResult,
                           REAL dtInv, int numX, int numY) {
  int j = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  int i = BLOCK_SIZE * blockIdx.y + threadIdx.y;
  int o = blockIdx.z;
  int numM = numX * numY;

  v[o * numX * numY + i * numY + j] = 0.0;

  if(j > 0) 
    {
      v[o * numX * numY + i * numY + j] +=  (0.5*myVarY[o * numM + i * numY + j]*myDyy[j * 4 + 0])
  *myResult[o * numM + i * numY + j-1];
    }

  v[o * numX * numY + i * numY + j] += (0.5*myVarY[o * numM + i * numY + j]
          *myDyy[j * 4 + 1])
    * myResult[o * numM + i  * numY + j];

  if(j < numY-1) 
    {
      v[o * numX * numY + i * numY + j] +=  
  (0.5*myVarY[o * numM + i * numY + j]
   *myDyy[j * 4 + 2])
  *myResult[o * numM + i * numY + j+1];
    }

  u[o * numX * numY + j * numX + i] += v[o * numX * numY + i * numY + j];

  // Implicit y

  ay[o * numX * numY + i * numY + j] =
    -0.5*(0.5*myVarY[o * numM + i * numY + j]
    *myDyy[j * 4 + 0]);

  by[o * numX * numY + i * numY + j] = 
    dtInv - 0.5*(0.5*myVarY[o * numM + i * numY + j]
     *myDyy[j * 4 + 1]);

  cy[o * numX * numY + i * numY + j] =
    -0.5*(0.5*myVarY[o * numM + i * numY + j]
    *myDyy[j * 4 + 2]);
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

  //Device memory

  cudaMemcpy(globs.du, u, outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(globs.dmyResult, globs.myResult, outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(globs.dmyVarX, globs.myVarX, outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(globs.dmyDxx, globs.myDxx, outer * numX * 4 * sizeof(REAL), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks(numX / BLOCK_SIZE, numY / BLOCK_SIZE, outer);

  REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);
  rollback_x<<< numBlocks, threadsPerBlock >>> (globs.dax, globs.dbx, globs.dcx, globs.du, globs.dmyVarX, globs.dmyDxx, globs.dmyResult,
            dtInv, numX, numY);

  cudaMemcpy(ax, globs.dax, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(bx, globs.dbx, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(cx, globs.dcx, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);


  REAL* dv,* dmyVarY,* dmyDyy,* day,* dby,* dcy;

  cudaMalloc((void**)&day, outer * numX * numY * sizeof(REAL));
  cudaMalloc((void**)&dby, outer * numX * numY * sizeof(REAL));
  cudaMalloc((void**)&dcy, outer * numX * numY * sizeof(REAL));
  cudaMalloc((void**)&dv, outer * numX * numY * sizeof(REAL));

  cudaMalloc((void**)&dmyVarY, outer * numX * numY * sizeof(REAL));
  cudaMalloc((void**)&dmyDyy, outer * numY * 4 * sizeof(REAL));


  cudaMemcpy(dmyVarY, globs.myVarY, outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(dmyDyy, globs.myDyy, outer * numX * 4 * sizeof(REAL), cudaMemcpyHostToDevice);


  numBlocks.x = numY / BLOCK_SIZE;
  numBlocks.y = numX / BLOCK_SIZE;

  rollback_y<<< numBlocks, threadsPerBlock >>> (day, dby, dcy, globs.du, dv, dmyVarY, dmyDyy, globs.dmyResult,
            dtInv, numX, numY);

  cudaMemcpy(ay, day, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(by, dby, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(cy, dcy, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(v, dv, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
  cudaMemcpy(u, globs.du, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);



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


  REAL* dy;

  cudaMalloc((void**)&dy, outer * numY * numX * sizeof(REAL));


  cudaMemcpy(globs.du, u, outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);
  rollback_implicit_y<<< numBlocks, threadsPerBlock >>> (dy, globs.du, dv,
            dtInv, numX, numY);
  cudaMemcpy(y, dy, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);

  // for(int o = 0; o < outer; o++)
  // {
  //   REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);
  //   for(int i=0;i<numX;i++)
  //   {
  //     for(int j=0;j<numY;j++)
  //     {  // here a, b, c should have size [numY]
  //       y[o * numX * numY + i * numY + j] =
  //         dtInv*u[o * numX * numY + j * numX + i]
  //          -0.5*v[o * numX * numY + i * numY + j];
  //     }
  //   }
  // }

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


  /* Free Memory */
  cudaFree(dv);
  cudaFree(dmyVarY);
  cudaFree(dmyDyy);
  cudaFree(day);
  cudaFree(dby);
  cudaFree(dcy);
  
  
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


  setPayoff_cuda(globals, outer);



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

//#endif // PROJ_CORE_ORIG
