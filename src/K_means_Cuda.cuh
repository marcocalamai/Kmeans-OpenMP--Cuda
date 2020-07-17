//
// Created by marco on 22/05/20.
//

#ifndef KMEANS_OPENMP_K_MEANS_CUDA_CUH
#define KMEANS_OPENMP_K_MEANS_CUDA_CUH

#include "cuda_runtime.h"
#include "Point.h"
#include <tuple>


#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err);



std::tuple<double *, short *> cuda_KMeans(double * deviceDataset, double * deviceCentroids, const int numPoint, const short k, const short dimPoint);
#endif //KMEANS_OPENMP_K_MEANS_CUDA_CUH

