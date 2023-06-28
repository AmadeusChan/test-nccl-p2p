#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <unistd.h>

#include "nccl.h"
#include "mpi.h"
#include "cuda_runtime.h"
#include "cuda.h"

#include <chrono>
#include <set>

#define MPICHECK(cmd) do {                          \
    int e = cmd;                                      \
    if( e != MPI_SUCCESS ) {                          \
        printf("Failed: MPI error %s:%d '%d'\n",        \
                __FILE__,__LINE__, e);   \
        exit(EXIT_FAILURE);                             \
    }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
        printf("Failed: Cuda error %s:%d '%s'\n",             \
                __FILE__,__LINE__,cudaGetErrorString(e));   \
        exit(EXIT_FAILURE);                             \
    }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
    ncclResult_t r = cmd;                             \
    if (r!= ncclSuccess) {                            \
        printf("Failed, NCCL error %s:%d '%s'\n",             \
                __FILE__,__LINE__,ncclGetErrorString(r));   \
        exit(EXIT_FAILURE);                             \
    }                                                 \
} while(0)

static uint64_t getHostHash(const char* string) {
    // Based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++){
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}

static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i=0; i< maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

static int check_process_placement_policy(
        int mpi_rank, int mpi_size
        ) {
    uint64_t hostname_hash[mpi_size];
    char hostname[1024];
    getHostName(hostname, 1024);
    uint64_t hash = getHostHash(hostname);
    MPICHECK(
            MPI_Allgather(
                &hash, 1, MPI_INT64_T,
                hostname_hash, 1, MPI_INT64_T,
                MPI_COMM_WORLD
                )
            );
    // verify that the process placement policy is round-robin
    std::set<uint64_t> hostname_set;
    for (int i = 0; i < mpi_size; ++ i) {
        hostname_set.insert(hostname_hash[i]);
    }
    int num_hosts = hostname_set.size();
    if (mpi_size % num_hosts != 0) {
        fprintf(stderr, "Please make sure that each node has the same number of processes (i.e., used GPUs)\n");
        exit(-1);
    }
    int num_gpu_per_host = mpi_size / num_hosts;
    bool assertion = true;
    for (int host = 0; host < num_hosts; ++ host) {
        int idx = host * num_gpu_per_host;
        for (int gpu_id = 1; gpu_id < num_gpu_per_host; ++ gpu_id) {
            assertion = assertion && (hostname_hash[idx + gpu_id] == hostname_hash[idx + gpu_id - 1]);
        }
    }
    if (! assertion) {
        fprintf(stderr, "Please make sure that the MPI uses the round-robin process placement policy. For example, if there are 8 processes and 2 nodes, the fist node should handle processes 0-3 while the second node handling processes 4-7.\n");
        exit(-1);
    }
    return mpi_rank % num_gpu_per_host;
}

int main(int argc, char ** argv) {

    int rank, world_size, provided;
    MPICHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));
    assert(provided == MPI_THREAD_MULTIPLE);
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    int gpu_id = check_process_placement_policy(rank, world_size);

    ncclUniqueId id;
    ncclComm_t comm;
    cudaStream_t s_0;
    cudaStream_t s_1;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    CUDACHECK(cudaSetDevice(gpu_id));
    NCCLCHECK(ncclCommInitRank(&comm, world_size, id, rank));
    CUDACHECK(cudaStreamCreateWithFlags(&s_0, cudaStreamNonBlocking));
    CUDACHECK(cudaStreamCreateWithFlags(&s_1, cudaStreamNonBlocking));

    const int msg_size = 32 * 1024 * 1024;
    uint8_t * send_buff = NULL;
    uint8_t * recv_buff = NULL;
    CUDACHECK(cudaMalloc(&send_buff, msg_size));
    CUDACHECK(cudaMalloc(&recv_buff, msg_size));
    CUDACHECK(cudaMemset(send_buff, 0, msg_size));
    CUDACHECK(cudaMemset(recv_buff, 0, msg_size));

    const int count = 128;
    if (rank == 0) {
        printf("Evaluating the Uni-Directional NCCL P2P Bandwidth (Gbps)\n");
        printf("   D\\D");
        for (int i = 0; i < world_size; ++ i) {
            printf("%6d ", i);
        }
        printf("\n");
    }
    for (int src = 0; src < world_size; ++ src) {
        if (rank == 0) {
            printf("%6d ",src);
        }
        for (int dst = 0; dst < world_size; ++ dst) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (src == dst) {
                if (rank == 0) {
                    printf("%6.02f ", 0.);
                }
                continue;
            }
            auto start = std::chrono::system_clock::now();
            for (int i = 0; i < count; ++ i) {
                if (rank == src) {
                    NCCLCHECK(
                            ncclSend(
                                send_buff, msg_size, ncclInt8,
                                dst, comm, s_0
                                )
                            );
                    CUDACHECK(cudaStreamSynchronize(s_0));
                } else if (rank == dst) {
                    NCCLCHECK(
                            ncclRecv(
                                recv_buff, msg_size, ncclInt8,
                                src, comm, s_0
                                )
                            );
                    CUDACHECK(cudaStreamSynchronize(s_0));
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            double time = elapsed_seconds.count() / count;
            double throughput = msg_size * 8. / time / 1e9; // Gbps
            if (rank == 0) {
                printf("%6.02f ", throughput);
                fflush(stdout);
            }
        }
        if (rank == 0) {
            printf("\n");
        }
    }

    if (rank == 0) {
        printf("\nEvaluating the Bi-Directional NCCL P2P Bandwidth (Gbps)\n");
        printf("   D\\D");
        for (int i = 0; i < world_size; ++ i) {
            printf("%6d ", i);
        }
        printf("\n");
    }
    for (int src = 0; src < world_size; ++ src) {
        if (rank == 0) {
            printf("%6d ",src);
        }
        for (int dst = 0; dst < world_size; ++ dst) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (src == dst) {
                if (rank == 0) {
                    printf("%6.02f ", 0.);
                }
                continue;
            }
            auto start = std::chrono::system_clock::now();
            for (int i = 0; i < count; ++ i) {
                if (rank == src) {
                    NCCLCHECK(
                            ncclGroupStart()
                            );
                    NCCLCHECK(
                            ncclSend(
                                send_buff, msg_size, ncclInt8,
                                dst, comm, s_0
                                )
                            );
                    NCCLCHECK(
                            ncclRecv(
                                recv_buff, msg_size, ncclInt8,
                                dst, comm, s_1
                                )
                            );
                    NCCLCHECK(
                            ncclGroupEnd()
                            );
                    CUDACHECK(cudaStreamSynchronize(s_0));
                    CUDACHECK(cudaStreamSynchronize(s_1));
                } else if (rank == dst) {
                    NCCLCHECK(
                            ncclGroupStart()
                            );
                    NCCLCHECK(
                            ncclSend(
                                send_buff, msg_size, ncclInt8,
                                src, comm, s_0
                                )
                            );
                    NCCLCHECK(
                            ncclRecv(
                                recv_buff, msg_size, ncclInt8,
                                src, comm, s_1
                                )
                            );
                    NCCLCHECK(
                            ncclGroupEnd()
                            );
                    CUDACHECK(cudaStreamSynchronize(s_0));
                    CUDACHECK(cudaStreamSynchronize(s_1));
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            double time = elapsed_seconds.count() / count;
            double throughput = msg_size * 8. / time / 1e9 * 2; // Gbps
            if (rank == 0) {
                printf("%6.02f ", throughput);
                fflush(stdout);
            }
        }
        if (rank == 0) {
            printf("\n");
        }
    }


    ncclCommDestroy(comm);
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    //MPICHECK(MPI_Finalize());

    return 0;
}


