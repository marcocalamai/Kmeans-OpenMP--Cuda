#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include <fstream>
#include <chrono>
#include <numeric>
#include "K_Means_Sequential.h"
#include "KMeans_OpenMP.h"
#include "KMeans_OpenMP2.h"




int main(int argc, char* argv[]) {
    //Txt line and Dataset
    std::string line;
    double value;
    std::vector<Point> dataset;

    if (argc != 3) {
        std::cerr << "usage: k_means <data-file> <k>" << std::endl;
        std::exit(EXIT_FAILURE);

    }

    std::ifstream dataset_file(argv[1]);
    if (!dataset_file) {
        std::cerr << "Could not open file: " << argv[1] << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (dataset_file.is_open()) {
        while (getline(dataset_file, line)) {
            std::istringstream iss(line);
            Point my_point;
            while (iss >> value) {
                my_point.dimensions.push_back(value);
            }
            dataset.push_back(my_point);
        }
        dataset_file.close();
    }

    //Number of dataset points
    auto numPoint = dataset.size();
    //Points dimension
    auto dimPoint = dataset[0].dimensions.size();
    //Get cluster number from input
    const auto k = std::strtol(argv[2], nullptr, 0);
    if (k == 0) {
        std::cerr << "Could not obtain the number of clusters, you have inserted: " << argv[2] << std::endl;
        exit(EXIT_FAILURE);
    }

    //Generate K random centroids
    std::vector<Point> centroids(k);
    std::vector<int> v(numPoint);
    std::iota(v.begin(), v.end(), 0);


    std::shuffle(v.begin(), v.end(), std::mt19937(std::random_device()()));

    for (int i = 0; i < k; i++) {
        centroids[i] = dataset[v[i]];
        centroids[i].clusterId = i;
    }

    std::vector<Point> output_Centroids;
    std::vector<Point> output_Dataset;








    //EXECUTION OF SEQUENTIAL Kmeans
    //CHRONO START
    auto start = std::chrono::high_resolution_clock::now();

    std::tie(output_Dataset, output_Centroids) = sequential_kmeans(dataset, centroids, k);
    //std::tie(output_Dataset, output_Centroids) = openMP_kmeans2(dataset, centroids, k);


    //CHRONO END
    auto finish = std::chrono::high_resolution_clock::now();
    //CHRONO TIME CALCULATION AND PRINT
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "SEQUENTIAL Elapsed time : " << elapsed.count() << " s\n \n";

    //Print centroids
    for (int i=0; i < k; i++){
        for(int j=0; j < dimPoint; j++){
            std::cout << output_Centroids[i].dimensions[j] << " ";
        }
        std::cout << std::endl;
    }


    //Execution of OpenMP Kmeans
    //CHRONO START
    start = std::chrono::high_resolution_clock::now();

    std::tie(output_Dataset, output_Centroids) = openMP_kmeans(dataset, centroids, k);


    //CHRONO END
    finish = std::chrono::high_resolution_clock::now();
    //CHRONO TIME CALCULATION AND PRINT
    elapsed = finish - start;
    std::cout << "OPEN_MP Elapsed time: " << elapsed.count() << " s\n \n";



    //Print centroids
    for (int i=0; i < k; i++){
        for(int j=0; j < dimPoint; j++){
            std::cout << output_Centroids[i].dimensions[j] << " ";
        }
        std::cout << std::endl;
    }

    //Execution of OpenMP Kmeans 2
    //CHRONO START
    start = std::chrono::high_resolution_clock::now();

    std::tie(output_Dataset, output_Centroids) = openMP_kmeans2(dataset, centroids, k);


    //CHRONO END
    finish = std::chrono::high_resolution_clock::now();
    //CHRONO TIME CALCULATION AND PRINT
    elapsed = finish - start;
    std::cout << "OPEN_MP2 Elapsed time: " << elapsed.count() << " s\n \n";



    //Print centroids
    for (int i=0; i < k; i++){
        for(int j=0; j < dimPoint; j++){
            std::cout << output_Centroids[i].dimensions[j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

