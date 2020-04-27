#ifdef _OPENMP
#include <omp.h>
#endif
//#include "Point.h"
//#include <tuple>

#include <cmath>
#include "KMeans_OpenMP2.h"





bool checkEqualClusters3(std::vector<Point> dataset, std::vector<Point> oldDataset, int numPoint){
    for (int i = 0; i < numPoint; i++){
        if(dataset[i].clusterId != oldDataset[i].clusterId){
            return false;
        }
    }
    return true;
}


std::tuple<std::vector<Point>,std::vector<Point>> openMP_kmeans2(std::vector<Point> dataset, std::vector<Point> centroids, int k) {
    bool convergence = false, firstIteration = true;
    std::vector<Point> oldDataset;
    //Number of dataset points
    auto numPoint = dataset.size();
    //Points dimension
    auto dimPoint = dataset[0].dimensions.size();

    int clusterLabel;
    double privateMinDistance, privateDistance;
    std::vector<int> count(k);



    while (!convergence) {

        //Find the nearest centroid and assign the point to that cluster
#pragma omp parallel for num_threads(4) default(none) \
    private(privateDistance, privateMinDistance, clusterLabel)\
    shared(numPoint, k, dimPoint, dataset, centroids)
            for (int i = 0; i < numPoint; i++) {
                privateMinDistance = std::numeric_limits<double>::max();
                for (int j = 0; j < k; j++) {
                    //distance = 0;
                    privateDistance = 0;
                    for (int h = 0; h < dimPoint; h++) {
                        privateDistance += pow(dataset[i].dimensions[h] - centroids[j].dimensions[h], 2);
                    }
                    privateDistance = sqrt(privateDistance);
                    if (privateDistance < privateMinDistance) {
                        privateMinDistance = privateDistance;
                        clusterLabel = j;
                    }
                }
                dataset[i].clusterId = clusterLabel;
            }

            //Centroids update
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


        if (!firstIteration && checkEqualClusters3(dataset, oldDataset, numPoint)) {
            convergence = true;
        } else {
            oldDataset = dataset;
            firstIteration = false;
        }
    }
    return {dataset, centroids};

}


