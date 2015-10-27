#include <iostream>
__global__ void updateParamsKernel(const unsigned g, const REAL alpha, 
                                   const REAL beta, const REAL nu, REAL* myVarX
                                   REAL* myVarY, REAL* myY, REAL* myX, 
                                   REAL* myTimeline,const int numY, const int numM){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    myVarX[z*numM+i*numY+j] = exp(2.0*(beta*log(myX[i])+myY[j]-0.5*nu*nu*myTimeline[g]));

}


void test(){
    std::cout << "nej\n";
    std::cout << "nej\n";
    std::cout << "nej\n";
    std::cout << "nej\n";
    std::cout << "nej\n";
    std::cout << "nej\n";
    std::cout << "nej\n";
}