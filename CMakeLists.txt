cmake_minimum_required(VERSION 3.15)
project(Kmeans-OpenMP)

set(CMAKE_CXX_STANDARD 20)
# added -fopenmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

add_executable(Kmeans-OpenMP src/KMeans_Sequential.cpp src/KMeans_OpenMP.cpp src/K_Means_Sequential.h src/KMeans_OpenMP.h src/Test.cpp src/Point.h src/KMeans_OpenMP2.cpp src/KMeans_OpenMP2.h)
