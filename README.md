# Parallel Computing Project
Project for Parallel Computing course. Sequential C++ and parallel implementation with Open-MP and CUDA of K-Means algorithm. Time comparison and speed-up can be found [Report](https://github.com/marcocalamai/Kmeans-OpenMP--Cuda/blob/master/Report/Report.pdf)

Note: KMeans_OpenMP2.cpp is a partial parallelization of K-Means algorithm (only assignment phase parallelized). 

## Installation

1. Clone the repo.
```sh
git clone https://github.com/marcocalamai/Kmeans-OpenMP--Cuda
```
2. Build with CMake.

## Usage

- For run K-Means algorithm with all implementations to search clusters on a dataset, pass two arguments (dataset path and numer of cluster to find):
```bash 
parallel_kmeans <dataset file path> <number of clusters>
```
- For generating random datasets according to global variables, pass no arguments:
```bash 
parallel_kmeans <>
```

## Authors
* **Marco Calamai** - GitHub: [marcocalamai](https://github.com/marcocalamai)
* **Elia Mercatanti** - GitHub: [elia-mercatanti](https://github.com/elia-mercatanti)

