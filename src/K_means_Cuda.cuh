//
// Created by marco on 22/05/20.
//

#ifndef KMEANS_OPENMP_K_MEANS_CUDA_CUH
#define KMEANS_OPENMP_K_MEANS_CUDA_CUH

#include "Point.h"
#include <tuple>

std::tuple<double *, short *> cuda_KMeans(double * deviceDataset, double * deviceCentroids, const int numPoint, const short k, const short dimPoint);
#endif //KMEANS_OPENMP_K_MEANS_CUDA_CUH

