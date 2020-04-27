//
// Created by marco on 18/04/20.
//

#ifndef PC_PROJECT_K_MEANS_SEQUENTIAL_H
#define PC_PROJECT_K_MEANS_SEQUENTIAL_H

#include "Point.h"
#include <tuple>

bool checkEqualClusters(std::vector<Point> dataset, std::vector<Point> oldDataset, int numPoint);
std::tuple<std::vector<Point>,std::vector<Point>> sequential_kmeans(std::vector<Point> dataset, std::vector<Point> centroids, int k);


#endif //PC_PROJECT_K_MEANS_SEQUENTIAL_H