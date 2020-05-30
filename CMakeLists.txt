cmake_minimum_required(VERSION 3.15)
project(Kmeans-OpenMP LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 14)


# added -fopenmp
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

add_executable(Kmeans-OpenMP src/KMeans_Sequential.cpp src/KMeans_OpenMP.cpp src/K_Means_Sequential.h src/KMeans_OpenMP.h src/Test.cpp src/Point.h src/KMeans_OpenMP2.cpp src/KMeans_OpenMP2.h src/K_means_Cuda.cu src/K_means_Cuda.cuh src)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Kmeans-OpenMP PUBLIC OpenMP::OpenMP_CXX)
endif()

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")