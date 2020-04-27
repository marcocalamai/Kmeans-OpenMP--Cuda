#include <cmath>
#include "K_Means_Sequential.h"



bool checkEqualClusters(std::vector<Point> dataset, std::vector<Point> oldDataset, int numPoint){
    for (int i = 0; i < numPoint; i++){
        if(dataset[i].clusterId != oldDataset[i].clusterId){
            return false;
        }
    }
    return true;
}
void
Assignement(std::vector<Point> &dataset, std::vector<Point> &centroids, int k, const unsigned long numPoint,
            const unsigned long dimPoint, int clusterLabel, double distance, double minDistance) {
    for (int i = 0; i < numPoint; i++) {
        minDistance = std::numeric_limits<double>::max();
        for (int j = 0; j < k; j++) {
            distance = 0;
            for (int h = 0; h < dimPoint; h++) {
                distance += pow(dataset[i].dimensions[h] - centroids[j].dimensions[h], 2);
            }
            distance = sqrt(distance);
            if (distance < minDistance) {
                minDistance = distance;
                clusterLabel = j;
            }
        }
        dataset[i].clusterId = clusterLabel;
    }
}


void UpdateCentroids(std::vector<Point> &dataset, std::vector<Point> &centroids, int k,
                     const unsigned long numPoint, const unsigned long dimPoint, std::vector<int> &count) {
    std::fill(count.begin(), count.end(), 0);
    for (int j = 0; j < k; j++) {
        //centroids[j].dimensions = zeros;
        std::fill(centroids[j].dimensions.begin(), centroids[j].dimensions.end(), 0);
    }


    for (int i = 0; i < numPoint; i++) {
        for (int h = 0; h < dimPoint; h++) {
            centroids[dataset[i].clusterId].dimensions[h] += dataset[i].dimensions[h];
        }
        count[dataset[i].clusterId]++;
    }


    for (int j = 0; j < k; j++) {
        for (int h = 0; h < dimPoint; h++) {
            centroids[j].dimensions[h] /= count[j];
        }
    }
}

std::tuple<std::vector<Point>, std::vector<Point>> sequential_kmeans(std::vector<Point> dataset, std::vector<Point> centroids, int k){
    bool convergence = false, firstIteration = true;
    std::vector<Point> oldDataset;
    //Number of dataset points
    const auto numPoint = dataset.size();
    //Points dimension
    const auto dimPoint = dataset[0].dimensions.size();


    int clusterLabel;
    double distance, minDistance;
    std::vector<int> count(k);
    //std::vector<double> zeros(dimPoint);
    while (!convergence){


        //Find the nearest centroid and assign the point to that cluster
        Assignement(dataset, centroids, k, numPoint, dimPoint, clusterLabel, distance, minDistance);

        //Centroids update
        //First, initialize centroids to zero before update their values
        UpdateCentroids(dataset, centroids, k, numPoint, dimPoint, count);


        if (!firstIteration && checkEqualClusters(dataset, oldDataset, numPoint)){
            convergence = true;
        }
        else{
            oldDataset = dataset;
            firstIteration = false;
        }
    }
    return{dataset, centroids};
}


