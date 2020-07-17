//
// Created by marco on 22/05/20.
//


#include "K_means_Cuda.cuh"
#include <cmath>
#include <iostream>



__constant__ short constK;
__constant__ int constNumPoint;
__constant__ short constDimPoint;

void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess) {
        return;
    }
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}



void print_device(double *device, int row, int col){
    double *host;
    host = (double *) malloc(row * col * sizeof(double));
    cudaMemcpy(host, device, row * col * sizeof(double),cudaMemcpyDeviceToHost);

    for (auto i = 0; i < row; i++) {
        for (auto j = 0; j < col; j++) {
            std::cout <<"- "<< host[i * col + j] << " ";
        }
        std::cout << "-" << std::endl;
    }
    std::cout << std::endl;
}

void print_device(short *device, int row, int col){
    short *host;
    host = (short *) malloc(row * col * sizeof(short));
    cudaMemcpy(host, device, row * col * sizeof(short),cudaMemcpyDeviceToHost);

    for (auto i = 0; i < row; i++) {
        for (auto j = 0; j < col; j++) {
            std::cout <<"- "<< host[i * col + j] << " ";
        }
        std::cout << "-" << std::endl;
    }
    std::cout << std::endl;
}

void print_device(int *device, int row, int col){
    int *host;
    host = (int *) malloc(row * col * sizeof(int));
    cudaMemcpy(host, device, row * col * sizeof(int),cudaMemcpyDeviceToHost);

    for (auto i = 0; i < row; i++) {
        for (auto j = 0; j < col; j++) {
            std::cout <<"- "<< host[i * col + j] << " ";
        }
        std::cout << "-" << std::endl;
    }
    std::cout << std::endl;
}



/*
//INIZIALIZE CENDROID ASSIGNEMENT TO ZERO FOR ALL POINT'S DATASETS
//Assegno ogni punto al cluster -1
__global__
void initialize_assignment(short * deviceAssignment){
    unsigned int threadId = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (threadId < constNumPoint){
        //printf("STAMPA DEL DEVICEASSIGNEMENT [%d] \n",deviceAssignment[threadId]);
        deviceAssignment[threadId] = -1;

    }
}
*/


__device__ double doubleAtomicAdd(double*address, double val){
    auto *address_as_ull = (unsigned long long int *) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val + __longlong_as_double((long long int)assumed)));
    } while (assumed != old);
    return __longlong_as_double((long long int)old);
}


__host__
bool checkEqualAssignment(const short * hostOldAssignment, const short * hostAssignment, const int numPoint){
    for (auto i = 0; i < numPoint; i++){
        if(hostOldAssignment[i] != hostAssignment[i]){
            return false;
        }
    }
    return true;
}


__global__
void compute_distances(const double * deviceDataset, const double * deviceCentroids, double * deviceDistances){
    double distance = 0;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < constNumPoint && col < constK){
        for (int i = 0; i < constDimPoint; i++) {
            distance += pow(deviceDataset[row*constDimPoint+i] - deviceCentroids[col*constDimPoint+i], 2);
            //printf("centroide %f ",deviceCentroids[col*constDimPoint+i]);
        }
        deviceDistances[row*constK+col] = sqrt(distance);
        //if (deviceDistances[row*constK+col] == 0){
        //    printf("distanza %f ",deviceDistances[row*constK+col]);
       //}
    }
}


__global__
void point_assignment(const double *deviceDistances, short *deviceAssignment){
    unsigned int threadId = (blockDim.x * blockIdx.x) + threadIdx.x;
    double min = INFINITY;
    short clusterLabel;
    double distance;
    if (threadId < constNumPoint){
        for (auto i = 0; i < constK; i++){
            distance = deviceDistances[threadId*constK + i];
            //printf("distanza %f ",distance);
            if(distance < min){
                min = distance;
                clusterLabel = i;
            }
        }
        deviceAssignment[threadId] = clusterLabel;
        //printf(" clusterID: %d",deviceAssignment[threadId]);
    }
}

__global__
void initialize_centroids(double * deviceCentroids){
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < constDimPoint && row < constK){
        deviceCentroids[row*constDimPoint + col] = 0;
    }
}


//Original compute sum with 2D grid (better with dataset with too much dimensions)
__global__
void compute_sum(const double *deviceDataset, double * deviceCentroids, const short *deviceAssignment, int * deviceCount){
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < constDimPoint && row < constNumPoint){
        short clusterId = deviceAssignment[row];
        //printf(" clusterID: %d",clusterId);
        doubleAtomicAdd(&deviceCentroids[clusterId*constDimPoint +col], deviceDataset[row*constDimPoint +col]);
        atomicAdd(&deviceCount[clusterId], 1);
        //printf(" c %f ",clusterId);
    }
}

//compute sum with 1D grid and iterate on point's dimensions
__global__
void compute_sum2(const double *deviceDataset, double * deviceCentroids, const short *deviceAssignment, int * deviceCount){
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < constNumPoint){
        short clusterId = deviceAssignment[row];
        for (auto i = 0; i< constDimPoint; i++){
            doubleAtomicAdd(&deviceCentroids[clusterId*constDimPoint+i], deviceDataset[row*constDimPoint+i]);
        }
        atomicAdd(&deviceCount[clusterId], 1);
    }
}

//Update centroids with 2D grid (better with dataset with too much dimensions)
__global__
void update_centroids(double * deviceCentroids, const int * deviceCount){
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < constDimPoint && row < constK) {
        //printf(" count  %f", (deviceCount[row]));
        deviceCentroids[row * constDimPoint + col] = deviceCentroids[row * constDimPoint + col] / (double(deviceCount[row])/constDimPoint);
        //printf(" centroide %f ",deviceCount[row]);
    }
}

//Update centroids with 1D grid (no need to divide count for point's dimensions)
__global__
void update_centroids2(double * deviceCentroids, const int * deviceCount){
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < constDimPoint && row < constK) {
        deviceCentroids[row * constDimPoint + col] /= deviceCount[row];

    }
}




__host__
std::tuple<double *, short *>cuda_KMeans(double * deviceDataset, double * deviceCentroids, const int numPoint, const short k, const short dimPoint){
    //int c = 0;
    dim3 dimBlockDistance(2, 512, 1);
    dim3 dimGridDistance(ceil(k/2.0), ceil(numPoint/512.0), 1);

    dim3 dimBlockInitialize(16, 16, 1);
    dim3 dimGridInitialize(ceil(dimPoint / 16.0), ceil(k / 16.0), 1);

    dim3 dimBlockComputeSum(2, 512, 1);
    dim3 dimGridComputeSum(ceil(dimPoint / 2.0), ceil(numPoint / 512.0), 1);

    dim3 dimBlockUpdateCentroids(16, 16, 1);
    dim3 dimGridUpdateCentroids(ceil(dimPoint / 16.0), ceil(k / 16.0), 1);

    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(constK, &k, sizeof(short)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(constNumPoint, &numPoint, sizeof(int)));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(constDimPoint, &dimPoint, sizeof(short)));
    //constant_dimPoint = dimPoint;
    bool convergence = false;

    short * hostOldAssignment;
    hostOldAssignment = (short *) malloc(numPoint * sizeof(short));

    short * hostAssignment;
    hostAssignment = (short *) malloc(numPoint * sizeof(short));


    short * deviceAssignment;
    CUDA_CHECK_RETURN(cudaMalloc((void **) &deviceAssignment, numPoint*sizeof(short)));
    double * deviceDistances;
    CUDA_CHECK_RETURN(cudaMalloc((void**) &deviceDistances, numPoint*k*sizeof(double)));
    int * deviceCount;
    CUDA_CHECK_RETURN(cudaMalloc((void**) &deviceCount, k*sizeof(int)));



    while (!convergence){
        //ASSIGNMENT
        //Find the nearest centroid and assign the point to that cluster
        compute_distances<<<dimGridDistance, dimBlockDistance>>>(deviceDataset, deviceCentroids, deviceDistances);
        cudaDeviceSynchronize();
        point_assignment<<<ceil(numPoint/1024.0), 1024>>>(deviceDistances, deviceAssignment);
        cudaDeviceSynchronize();

        //CENTROIDS UPDATE
        //Initialize centroids to 0 and set count to 0 (for compute means)
        initialize_centroids<<<dimGridInitialize, dimBlockInitialize>>>(deviceCentroids);

        //print_device(deviceCentroids, k,  dimPoint);
        //return{deviceCentroids, hostAssignment};
        CUDA_CHECK_RETURN(cudaMemset(deviceCount, 0, k*sizeof(int)));
        //print_device(deviceCount, k,  1);
        cudaDeviceSynchronize();
        //Compute all sum for centroids

        compute_sum<<<dimGridComputeSum,dimBlockComputeSum>>>(deviceDataset, deviceCentroids, deviceAssignment, deviceCount);
        //compute_sum2<<<ceil(numPoint/1024.0), 1024>>>(deviceDataset, deviceCentroids, deviceAssignment, deviceCount);

        cudaDeviceSynchronize();
        //printf("\n STAMPA DI TEST \n");
        //print_device(deviceCentroids, k,  dimPoint);
        //printf("\n");
        //return{deviceCentroids, hostAssignment};
        //Compute mean: division for count


        update_centroids<<<dimGridUpdateCentroids,dimBlockUpdateCentroids>>>(deviceCentroids,deviceCount);
        //update_centroids2<<<dimGridUpdateCentroids,dimBlockUpdateCentroids>>>(deviceCentroids,deviceCount);

        cudaDeviceSynchronize();

        CUDA_CHECK_RETURN(cudaMemcpy(hostAssignment, deviceAssignment, numPoint*sizeof(short), cudaMemcpyDeviceToHost));


/*
        std::cout << "PRINT HOST ASSIGNMENT     ";
        for(auto i = 0; i<numPoint; i++){
            std::cout << hostAssignment[i] << " ";
        }
        std::cout << "\n" ;

        std::cout << "PRINT HOST OLD ASSIGNMENT ";
        for(auto i = 0; i<numPoint; i++){
            std::cout << hostOldAssignment[i] << " " ;
        }
        std::cout << "\n" ;
*/
        //c ++;


        if (checkEqualAssignment(hostOldAssignment, hostAssignment, numPoint)){
            convergence = true;
        }
        else{
            CUDA_CHECK_RETURN(cudaMemcpy(hostOldAssignment, deviceAssignment, numPoint*sizeof(short), cudaMemcpyDeviceToHost));
        }
        //printf("\n");

    }
    //std::cout << "Numero di iterazioni: " << c << " \n";

    return{deviceCentroids, hostAssignment};
}
