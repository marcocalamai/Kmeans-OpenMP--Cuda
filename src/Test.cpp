#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include <fstream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include "K_Means_Sequential.h"
#include "KMeans_OpenMP.h"
#include "KMeans_OpenMP2.h"
#include "K_means_Cuda.cuh"
//#include "cuda_runtime.h"


int main(int argc, char* argv[]) {
// /home/marco/Programming/CLionProjects/Kmeans-OpenMP-&-Cuda/Datasets/KDDCUP04Bio.txt 2000

    //Txt line and Dataset
    std::string line;
    double value;
    std::vector<Point> dataset;

    if (argc == 1) {
        //std::cout << "arg ==1 \n";
        std::random_device rd; //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(0.0, 1.0);
        std::string nameDataset;
        int pointsDataset[] = {1000, 10000, 100000, 1000000};
        int dimensionsDataset[] = {10, 100, 1000, 10000};
        for (auto i : pointsDataset){
            for(auto j : dimensionsDataset) {
                nameDataset = "/home/marco/Programming/CLionProjects/Kmeans-OpenMP-&-Cuda/Datasets/" + std::to_string(i) +"x" + std::to_string(j) + ".txt";
                std::ifstream dataset_file(nameDataset);
                if (!dataset_file) {
                    std::cout << "dataset "+ std::to_string(i) +"x" + std::to_string(j) + ".txt NON trovato \n";

                    std::ofstream outfile(nameDataset);
                    for (auto numP = 0; numP < i; numP++) {
                        for (auto dimP = 0; dimP < j; dimP++) {
                            outfile << dis(gen) << " ";
                        }
                        outfile << "\n";
                    }
                    outfile.close();
                    std::cout << "GENERATO file " + std::to_string(i) +"x" + std::to_string(j) + "\n";
                }
                std::cout << "dataset "+ std::to_string(i) +"x" + std::to_string(j) + ".txt TROVATO \n";
            }
        }
        std::exit(0);

    };


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


/*
    //PRINT INITIAL CENTROIDS
    //Print centroids
    std::cout << "PRINT INITIAL CENTROIDS after then dataset is load \n";
    for (int i=0; i < k; i++){
        for(int j=0; j < dimPoint; j++){
            std::cout << centroids[i].dimensions[j] << " ";
        }
        std::cout << std::endl;
    }
*/




    //************************EXECUTION OF SEQUENTIAL Kmeans************************
    //CHRONO START
    auto start = std::chrono::high_resolution_clock::now();

    std::tie(output_Dataset, output_Centroids) = sequential_kmeans(dataset, centroids, k);



    //CHRONO END
    auto finish = std::chrono::high_resolution_clock::now();
    //CHRONO TIME CALCULATION AND PRINT
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "SEQUENTIAL Elapsed time : " << elapsed.count() << " s\n \n";


/*
    //Print centroids

    for (int i=0; i < k; i++){
        for(int j=0; j < dimPoint; j++){
            std::cout << output_Centroids[i].dimensions[j] << " ";
        }
        std::cout << std::endl;
    }
*/


    //************************Execution of OpenMP Kmeans************************
    //CHRONO START
    start = std::chrono::high_resolution_clock::now();

    std::tie(output_Dataset, output_Centroids) = openMP_kmeans(dataset, centroids, k);

    //CHRONO END
    finish = std::chrono::high_resolution_clock::now();
    //CHRONO TIME CALCULATION AND PRINT
    elapsed = finish - start;
    std::cout << "OPEN_MP Elapsed time: " << elapsed.count() << " s\n \n";


/*
    //Print centroids
    for (int i=0; i < k; i++){
        for(int j=0; j < dimPoint; j++){
            std::cout << output_Centroids[i].dimensions[j] << " ";
        }
        std::cout << std::endl;
    }
*/

    //************************Execution of OpenMP Kmeans 2************************
    //CHRONO START
    start = std::chrono::high_resolution_clock::now();

    std::tie(output_Dataset, output_Centroids) = openMP_kmeans2(dataset, centroids, k);


    //CHRONO END
    finish = std::chrono::high_resolution_clock::now();
    //CHRONO TIME CALCULATION AND PRINT
    elapsed = finish - start;
    std::cout << "OPEN_MP2 Elapsed time: " << elapsed.count() << " s\n \n";


/*
    //Print centroids
    for (int i=0; i < k; i++){
        for(int j=0; j < dimPoint; j++){
            std::cout << output_Centroids[i].dimensions[j] << " ";
        }
        std::cout << std::endl;
    }
*/




    //************************CUDA************************


    auto size_dataset = numPoint*dimPoint*sizeof(double);
    auto size_centroids = k*dimPoint*sizeof(double);

    double *hostDataset;
    hostDataset = (double *) malloc(size_dataset);
    double *hostCentroids;
    hostCentroids = (double *) malloc(size_centroids);

    double *deviceDataset, *deviceCentroids;

    //Moove dataset and centroids from vector to simple matrix

    for(auto i = 0; i<numPoint; i++){
        for(auto j = 0; j<dimPoint; j++){
            hostDataset[i*dimPoint+j] = dataset[i].dimensions[j];
        }
    }

    for(auto i = 0; i<k; i++){
        for(auto j = 0; j<dimPoint; j++){
            hostCentroids[i*dimPoint+j] = centroids[i].dimensions[j];
        }
    }



    //ALLOCATE AND COPY DATASET AND CENTROIDS TO DEVICE
    CUDA_CHECK_RETURN(cudaMalloc((void **) &deviceDataset, size_dataset));
    CUDA_CHECK_RETURN(cudaMemcpy(deviceDataset, hostDataset, size_dataset, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &deviceCentroids, size_centroids));
    CUDA_CHECK_RETURN(cudaMemcpy(deviceCentroids, hostCentroids, size_centroids, cudaMemcpyHostToDevice));

    short * hostAssignment;
    hostAssignment = (short *) malloc(numPoint * sizeof(short));



    //CHRONO START
    start = std::chrono::high_resolution_clock::now();

    //KERNEL LAUNCH
    std::tie(deviceCentroids, hostAssignment) = cuda_KMeans(deviceDataset, deviceCentroids, numPoint, k, dimPoint);

    //CHRONO END
    finish = std::chrono::high_resolution_clock::now();
    //CHRONO TIME CALCULATION AND PRINT
    elapsed = finish - start;
    std::cout << "CUDA Elapsed time: " << elapsed.count() << " s\n \n";


    CUDA_CHECK_RETURN(cudaMemcpy(hostCentroids, deviceCentroids, size_centroids, cudaMemcpyDeviceToHost));

/*
    //PRINT FINAL CENTROIDS
    std::cout << "PRINT FINAL CUDA CENTROIDS \n";
    for(auto i = 0; i<k; i++){
        for(auto j = 0; j<dimPoint; j++){
            std::cout << hostCentroids[i*dimPoint+j] << " ";
        }
        std::cout << "\n";
    }
*/

    CUDA_CHECK_RETURN(cudaFree(deviceDataset));
    CUDA_CHECK_RETURN(cudaFree(deviceCentroids));


    free(hostDataset);
    free(hostCentroids);
    free(hostAssignment);


    return 0;
}






