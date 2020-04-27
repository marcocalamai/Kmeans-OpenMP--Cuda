//
// Created by marco on 18/04/20.
//

#ifndef PC_PROJECT_KMEANS_OPENMP_H
#define PC_PROJECT_KMEANS_OPENMP_H

#include "Point.h"
#include <tuple>

bool checkEqualClusters2(std::vector<Point> dataset, std::vector<Point> oldDataset, int numPoint);
std::tuple<std::vector<Point>,std::vector<Point>> openMP_kmeans(std::vector<Point> dataset, std::vector<Point> centroids, int k);

#endif //PC_PROJECT_KMEANS_OPENMP_H
