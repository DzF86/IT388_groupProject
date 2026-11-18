/* OpenMP Floyd-Warshall with row decomposition (pure OpenMP).
 *
 * Compile:  gcc -O2 -Wall -fopenmp -o fw_omp FW_OMP.c
 * Run:      ./fw_omp 4 [n=512] [density=0.3] [seed=42] [max_w=100] [--print]
 *           (first argument is the number of threads; on Expanse run it inside the job script)
 *
 * How it works:
 *   - Builds the graph and runs serial Floyd-Warshall first for the correct answer and the speedup baseline.
 *   - For each k (this loop must stay sequential): every thread updates its own
 *     block of rows using row k, then all threads wait at a barrier before k+1.
 *   - Checks OpenMP against serial and prints time, sync, speedup, efficiency + checksum.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <omp.h>

#define INF (INT_MAX / 2)                 
#define IDX(i, j, n) ((size_t)(i) * (size_t)(n) + (size_t)(j))
#define MAX_N 46340                                              

void Usage(char* prog_name);
void generateGraph(int *D, int N, double density, int maxWeight, unsigned int seed);
void serial(int *D, int N);
double parallel(int *D, int N, int nThreads);
void printMatrix(int *D, int N);

int main(int argc, char* argv[]) {
    int N = 512, maxWeight = 100, nThreads, print = 0;    
    double density = 0.3;
    unsigned int seed = 42;
    double start, end;                                      
    double serialTime, parallelTime, syncTime, speedup, efficiency;

    if (argc < 2) Usage(argv[0]);

    nThreads = atoi(argv[1]);
    if (nThreads < 1) Usage(argv[0]);

    int pos = 0;
    for (int a = 2; a < argc; a++) {
        if (strcmp(argv[a], "--print") == 0) print = 1;
        else if (pos == 0) { N = atoi(argv[a]); pos++; }
        else if (pos == 1) { density = atof(argv[a]); pos++; }
        else if (pos == 2) { seed = (unsigned int) strtoul(argv[a], NULL, 10); pos++; }
        else if (pos == 3) { maxWeight = atoi(argv[a]); pos++; }
        else {
            fprintf(stderr, "Too many arguments (extra argument \"%s\")\n", argv[a]);
            Usage(argv[0]);
        }
    }

    if (N < 1 || N > MAX_N) {
        fprintf(stderr, "n must be an integer from 1 to %d\n", MAX_N);
        Usage(argv[0]);
    }
    if (density < 0.0 || density > 1.0) {
        fprintf(stderr, "density must be a number from 0 to 1\n");
        Usage(argv[0]);
    }
    if (maxWeight < 1) {
        fprintf(stderr, "max_weight must be an integer >= 1\n");
        Usage(argv[0]);
    }
    //  Longest possible path must stay below INF or sums could be mistaken for "no path"
    if ((long long) maxWeight * (N - 1) >= (long long) INF) {
        fprintf(stderr, "n=%d with max_weight=%d is too large for the INF limit (%d)\n", N, maxWeight, INF);
        Usage(argv[0]);
    }

    //  Allocate the starting graph and one copy for each version
    size_t bytes = (size_t) N * N * sizeof(int);
    int *G = malloc(bytes);
    int *serialD = malloc(bytes);
    int *parallelD = malloc(bytes);
    if (G == NULL || serialD == NULL || parallelD == NULL) {
        printf("Memory allocation failed.\n");
        free(G); free(serialD); free(parallelD);
        return 1;
    }

    //  Generate the starting graph
    generateGraph(G, N, density, maxWeight, seed);

    if (print) {
        printf("Input:\n");
        printMatrix(G, N);
    }

    memcpy(serialD, G, bytes);
    memcpy(parallelD, G, bytes);

    start = omp_get_wtime();
    serial(serialD, N);
    end = omp_get_wtime();
    serialTime = end - start;

    //  Time the OpenMP version (syncTime = barrier wait)
    start = omp_get_wtime();
    syncTime = parallel(parallelD, N, nThreads);
    end = omp_get_wtime();
    parallelTime = end - start;

    speedup = serialTime / parallelTime;
    efficiency = speedup / nThreads;

    if (print) {
        printf("Shortest paths:\n");
        printMatrix(parallelD, N);
    }

    //  Check that OpenMP got the same answer as serial
    if (memcmp(serialD, parallelD, bytes) != 0) {
        printf("WARNING: OpenMP result does not match serial result\n");
    }

    //  Checksum: sum of reachable distances and count of INF
    long long sum = 0, infs = 0;
    for (size_t k = 0; k < (size_t) N * N; k++) {
        if (parallelD[k] >= INF) infs++;
        else sum += parallelD[k];
    }

    printf("OpenMP n=%d threads=%d density=%.2f seed=%u time=%.6f s sync=%.6f s serial=%.6f s speedup=%.3f efficiency=%.3f checksum=%lld inf_count=%lld\n",
           N, nThreads, density, seed, parallelTime, syncTime, serialTime, speedup, efficiency, sum, infs);

    free(G);
    free(serialD);
    free(parallelD);

    return 0;
}  /* main */

//  Serial Floyd-Warshall, used as the baseline for correctness and speedup
void serial(int *D, int N) {
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < N; i++) {
            int distIK = D[IDX(i, k, N)];
            if (distIK >= INF) continue;

            for (int j = 0; j < N; j++) {
                int distKJ = D[IDX(k, j, N)];
                if (distKJ >= INF) continue;

                int throughK = distIK + distKJ;
                if (throughK < D[IDX(i, j, N)]) D[IDX(i, j, N)] = throughK;
            }
        }
    }
}

//    OpenMP Floyd-Warshall with row decomposition, returns barrier wait time
//      - k loop stays in order (iteration k uses the results of k-1)
//      - rows i are split across threads, each thread does every j for its rows
//      - same split as MPI: the first (N % nThreads) threads (rank 0 included) take one extra row
//      - one barrier per k, so N synchronization points
double parallel(int *D, int N, int nThreads) {
    double maxSync = 0.0;

    #pragma omp parallel num_threads(nThreads) reduction(max:maxSync)
    {
        int rank = omp_get_thread_num();
        int numThreads = omp_get_num_threads();
        int baseRows = N / numThreads;
        int extraRows = N % numThreads;
        int myRows = baseRows + (rank < extraRows ? 1 : 0);
        int myStart = rank * baseRows + (rank < extraRows ? rank : extraRows);
        int myEnd = myStart + myRows;
        double mySync = 0.0;

        for (int k = 0; k < N; k++) {
            //  Row k and column k don't change during iteration k (D[k][k] = 0),
            //  so every thread can read them safely (no broadcast needed like MPI)
            for (int i = myStart; i < myEnd; i++) {
                int distIK = D[IDX(i, k, N)];
                if (distIK >= INF) continue;

                for (int j = 0; j < N; j++) {
                    int distKJ = D[IDX(k, j, N)];
                    if (distKJ >= INF) continue;

                    int throughK = distIK + distKJ;
                    if (throughK < D[IDX(i, j, N)]) D[IDX(i, j, N)] = throughK;
                }
            }

            //  Every thread must finish iteration k before anyone starts k+1
            double syncStart = omp_get_wtime();
            #pragma omp barrier
            mySync += omp_get_wtime() - syncStart;
        }

        maxSync = mySync;
    }

    return maxSync;
}

//  Random graph generator
void generateGraph(int *D, int N, double density, int maxWeight, unsigned int seed) {
    unsigned int state = seed;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                D[IDX(i, j, N)] = 0;
            } else {
                double r = (double) rand_r(&state) / (double) RAND_MAX;
                if (r < density) D[IDX(i, j, N)] = 1 + (rand_r(&state) % maxWeight);
                else D[IDX(i, j, N)] = INF;
            }
        }
    }
}

//  Print matrix function
void printMatrix(int *D, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (D[IDX(i, j, N)] >= INF) printf("  INF");
            else printf("%5d", D[IDX(i, j, N)]);
        }
        printf("\n");
    }
}

void Usage(char* prog_name) {
   fprintf(stderr, "Usage: %s <number of threads> [n] [density] [seed] [max_weight] [--print]\n", prog_name);
   exit(0);
}  /* Usage */
