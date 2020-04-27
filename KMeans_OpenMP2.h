//
// Created by marco on 18/04/20.
//

#ifndef PC_PROJECT_KMEANS2_OPENMP_H
#define PC_PROJECT_KMEANS2_OPENMP_H

#include "Point.h"
#include <tuple>

bool checkEqualClusters3(std::vector<Point> dataset, std::vector<Point> oldDataset, int numPoint);
std::tuple<std::vector<Point>,std::vector<Point>> openMP_kmeans2(std::vector<Point> dataset, std::vector<Point> centroids, int k);

#endif //PC_PROJECT_KMEANS2_OPENMP_H
