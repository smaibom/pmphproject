#include "ProjHelperFun.cu.h"
#include "Constants.h"
#include "TridagKernel.cu.h"

#define BLOCK_SIZE 32

template <int T>
__global__ void tilling_transpose_kernel(REAL *m_in, REAL *m_out, int rows ,int collums) {
    __shared__ float tile[T][T+1];
    int i = threadIdx.y;
    int j = threadIdx.x;
    int z = blockIdx.z;
    int ii = blockIdx.y*T+i;
    int jj = blockIdx.x*T+j;
    if(ii < rows && jj < collums ) {
        tile[i][j] = m_in[z*rows*collums+ii*collums+jj];
    } 
    __syncthreads();
    ii = blockIdx.y*T+j;
    jj = blockIdx.x*T+i;
    if( jj < collums && ii < rows){
        m_out[z*rows*collums+jj*rows+ii] = tile[j][i];
    }

}


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
  
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks(numX / BLOCK_SIZE, numY / BLOCK_SIZE, outer);
  updateParamsKernel<<< numBlocks, threadsPerBlock >>> (g, alpha, beta, nu, globs.dmyVarX, 
                                                        globs.dmyVarY, globs.dmyY,globs.dmyX,globs.dmyTimeline,numY,numX*numY);

  cudaMemcpy(globs.myVarX, globs.dmyVarX, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
  //REAL* myVarXNew = (REAL*) malloc(sizeof(REAL) * globs.numX * globs.numY * outer);
  const int T = BLOCK_SIZE;

  tilling_transpose_kernel<T><<<numBlocks,threadsPerBlock>>>(globs.dmyVarX,globs.tmp,globs.numX,globs.numY);
  //free(globs.myVarX);
  //globs.myVarX = myVarXNew;
  REAL* tmp = globs.dmyVarX;
  globs.dmyVarX = globs.tmp;
  globs.tmp = tmp;

  //cudaMemcpy(globs.dmyVarX, globs.myVarX, outer * numX * numY * sizeof(REAL), cudaMemcpyHostToDevice);
  
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
  
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(globs.numX / BLOCK_SIZE, globs.numY / BLOCK_SIZE, outer);
  
  //kernel
    setPayoffKernel<<<numBlocks, threadsPerBlock>>>(globs.dmyX, globs.dmyResult, globs.numX, globs.numY);

  cudaMemcpy(globs.myResult, globs.dmyResult, outer*globs.numX*globs.numY*sizeof(REAL), cudaMemcpyDeviceToHost);
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

  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks(numX / BLOCK_SIZE, numY / BLOCK_SIZE, outer);

  REAL dtInv = 1.0/(globs.myTimeline[g+1]-globs.myTimeline[g]);
  rollback_x<<< numBlocks, threadsPerBlock >>> (globs.dax, globs.dbx, globs.dcx, globs.du, globs.dmyVarX, globs.dmyDxx, globs.dmyResult,
            dtInv, numX, numY);

  numBlocks.x = numY / BLOCK_SIZE;
  numBlocks.y = numX / BLOCK_SIZE;
  cudaThreadSynchronize();
  rollback_y<<< numBlocks, threadsPerBlock >>> (globs.day, globs.dby, globs.dcy, globs.du, globs.dv, globs.dmyVarY, globs.dmyDyy, globs.dmyResult,
            dtInv, numX, numY);

  cudaThreadSynchronize();
  tridagCUDAWrapper( numY,
         globs.dax,
         globs.dbx,
         globs.dcx,
         globs.du,
         numX * numY * outer,
         numX,
         globs.du,
         globs.duu );
  rollback_implicit_y<<< numBlocks, threadsPerBlock >>> (globs.dy, globs.du, globs.dv,
            dtInv, numX, numY);
  cudaThreadSynchronize();
  tridagCUDAWrapper( numY,
         globs.day,
         globs.dby,
         globs.dcy,
         globs.dy,
         numX * numY * outer,
         numY,
         globs.dmyResult,
         globs.duu );
  
}



void   run_OrigCPU(const unsigned int& outer,const unsigned int& numX,
                   const unsigned int& numY,const unsigned int& numT,
                   const REAL& s0,const REAL& t,const REAL& alpha,
                   const REAL& nu,const REAL& beta,REAL* res) // [outer] RESULT
{
  PrivGlobs globs(numX, numY, numT, outer);
  initGrid(s0,alpha,nu,t, numX, numY, numT, globs);
  initOperator(globs.myX, globs.numX, globs.myDxx);
  initOperator(globs.myY, globs.numY, globs.myDyy);

  cudaMemcpy(globs.dmyX, globs.myX, globs.numX*sizeof(REAL ), cudaMemcpyHostToDevice);

  setPayoff_cuda(globs, outer);

  cudaMemcpy(globs.dmyDxx, globs.myDxx, outer * numX * 4 * sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(globs.dmyDyy, globs.myDyy, outer * numX * 4 * sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(globs.dmyY, globs.myY, numY * sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(globs.dmyTimeline, globs.myTimeline, numT * sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(globs.dmyX, globs.myX, numX * sizeof(REAL), cudaMemcpyHostToDevice);


  for(int g = numT-2;g>=0;--g)
    {
      updateParams(g,alpha,beta,nu,globs, outer);
      cudaThreadSynchronize();
      rollback(g, globs, outer, numX, numY);
    }

  cudaMemcpy(globs.myResult, globs.dmyResult, outer * numX * numY * sizeof(REAL), cudaMemcpyDeviceToHost);
  
  for (unsigned int i = 0; i < outer; i++) 
  {
    res[i] = globs.myResult[i * globs.numM + globs.myXindex * numY + globs.myYindex];
  }
}

//#endif // PROJ_CORE_ORIG
