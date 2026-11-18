/* MPI Floyd-Warshall with row decomposition (pure MPI).
 *
 * Compile:  mpicc -O2 -o fw_mpi fw_mpi.c
 * Run:      mpirun -np 4 ./fw_mpi [n=512] [density=0.3] [seed=42] [max_w=100] [--print]
 *           (on Expanse use srun inside the job script)
 *
 * How it works:
 *   - Rank 0 builds the graph and scatters contiguous blocks of rows.
 *   - For each k (this loop must stay sequential): the rank that owns row k
 *     broadcasts it, then every rank updates its own rows using row k.
 *   - Rank 0 gathers the result and prints time + checksum.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <errno.h>
#include <mpi.h>

#define INF (INT_MAX / 2)
#define IDX(i, j, n) ((size_t)(i) * (size_t)(n) + (size_t)(j))

/* ---------- Graph generation ---------- */

void generate_graph(int *D, int n, double density, int max_w, unsigned seed)
{
    unsigned state = seed;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(i == j){
                D[IDX(i, j, n)] = 0;
            }else{
                double r = (double)rand_r(&state) / (double)RAND_MAX;
                if(r < density){
                    D[IDX(i, j, n)] = 1 + (rand_r(&state) % max_w);
                }else{
                    D[IDX(i, j, n)] = INF;
                }
            }
        }
    }
}

void matrix_checksum(const int *D, int n, long long *sum, long long *inf_count)
{
    long long s = 0, c = 0;
    for(size_t k = 0; k < (size_t)n * (size_t)n; k++){
        if (D[k] >= INF) c++;
        else s += D[k];
    }
    *sum = s;
    *inf_count = c;
}

void print_matrix(const int *D, int n)
{
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(D[IDX(i, j, n)] >= INF){
                printf("  INF");
            }else{
                printf("%5d", D[IDX(i, j, n)]);
            }
        }
        printf("\n");
    }
}

/* ---------- Command-line argument checking ---------- */
 
#define MAX_N 46340   /* largest n where n*n still fits in an int (MPI counts) */
 
static void print_usage(FILE *out, const char *prog)
{
    fprintf(out, "Usage: %s [n] [density] [seed] [max_weight] [--print]\n", prog);
}
 
#define FAIL(...) do { \
    if (verbose) print_usage(stderr, argv[0]); \
    return 1; } while (0)
 
/* Returns 0 if OK, 1 if an argument is invalid, 2 if help was requested.
 * Only prints messages when verbose is nonzero (so only one MPI rank talks). */
static int parse_args(int argc, char **argv, int *n, double *density, unsigned *seed, int *max_w, int *print, int verbose)
{
    int pos = 0;
    for(int a = 1; a < argc; a++){
        const char *arg = argv[a];
        char *end;
 
        if(strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0){
            if(verbose){
                print_usage(stdout, argv[0]);
            }
            return 2;
        }
        if(strcmp(arg, "--print") == 0){
            *print = 1; continue;
        }
        if(arg[0] == '-' && arg[1] == '-'){
            FAIL("unknown option \"%s\"", arg);
        }
 
        errno = 0;
        switch(pos){
        case 0: {
            long v = strtol(arg, &end, 10);
            if(*arg == '\0' || *end != '\0' || errno || v < 1 || v > MAX_N){
                FAIL("n must be an integer from 1 to %d, but got \"%s\"", MAX_N, arg);
            }
            *n = (int)v;
            break; }
        case 1: {
            double v = strtod(arg, &end);
            if(*arg == '\0' || *end != '\0' || errno || !(v >= 0.0 && v <= 1.0)){
                FAIL("density must be a number from 0 to 1, but got \"%s\"", arg);
            }
            *density = v;
            break; }
        case 2: {
            unsigned long long v = strtoull(arg, &end, 10);
            if(*arg == '\0' || *arg == '-' || *end != '\0' || errno || v > UINT_MAX){
                FAIL("seed must be a non-negative integer, but got \"%s\"", arg);
            }
            *seed = (unsigned)v;
            break; }
        case 3: {
            long v = strtol(arg, &end, 10);
            if(*arg == '\0' || *end != '\0' || errno || v < 1 || v > INT_MAX){
                FAIL("max_weight must be an integer >= 1, but got \"%s\"", arg);
            }
            *max_w = (int)v;
            break; }
        default:
            FAIL("too many arguments (extra argument \"%s\")", arg);
        }
        pos++;
    }
 
    /* Longest possible path must stay below INF or sums could be mistaken for "no path" */
    if((long long)(*max_w) * (*n - 1) >= (long long)INF){
        FAIL("n=%d with max_weight=%d is too large: path lengths could reach %lld, "
             "which overflows the INF limit (%d). Use a smaller n or max_weight.",
             *n, *max_w, (long long)(*max_w) * (*n - 1), INF);
    }
    return 0;
}

/* ---------- Main ---------- */

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 512, max_w = 100, print = 0, pos = 0;
    double density = 0.3;
    unsigned seed = 42;
    int status = parse_args(argc, argv, &n, &density, &seed, &max_w, &print, rank == 0);
    if(status != 0){
        MPI_Finalize(); return status == 1 ? 1 : 0;
    }

    /* Row decomposition: rank r gets rows_per[r] rows starting at first_row[r].
     * The first (n % size) ranks (rank 0 included) take one extra row. */
    int *rows_per = malloc(size * sizeof(int));
    int *first_row = malloc(size * sizeof(int));
    int *sendcnt = malloc(size * sizeof(int));/* in ints, for Scatterv/Gatherv */
    int *displs = malloc(size * sizeof(int));
    int *owner = malloc(n * sizeof(int));/* owner[k] = rank holding row k */

    int base = n / size, rem = n % size, start = 0;
    for(int r = 0; r < size; r++){
        rows_per[r]  = base + (r < rem ? 1 : 0);
        first_row[r] = start;
        sendcnt[r]   = rows_per[r] * n;
        displs[r]    = start * n;
        for(int i = start; i < start + rows_per[r]; i++){
            owner[i] = r;
        }
        start += rows_per[r];
    }
    int my_rows  = rows_per[rank];
    int my_first = first_row[rank];

    int *D = NULL;/* full matrix, rank 0 only */
    if(rank == 0){
        D = malloc((size_t)n * n * sizeof(int));
        generate_graph(D, n, density, max_w, seed);
        if(print){
            printf("Input:\n"); print_matrix(D, n);
        }
    }
    int *local = malloc((size_t)(my_rows > 0 ? my_rows : 1) * n * sizeof(int));
    int *row_k = malloc((size_t)n * sizeof(int));

    MPI_Scatterv(D, sendcnt, displs, MPI_INT, local, my_rows * n, MPI_INT, 0, MPI_COMM_WORLD);

    /* ----- timed section ----- */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();
    double comm_time = 0.0;

    for (int k = 0; k < n; k++) {
        /* Owner copies row k into the broadcast buffer */
        if(rank == owner[k]){
            memcpy(row_k, &local[IDX(k - my_first, 0, n)], n * sizeof(int));
        }

        double c0 = MPI_Wtime();
        MPI_Bcast(row_k, n, MPI_INT, owner[k], MPI_COMM_WORLD);
        comm_time += MPI_Wtime() - c0;

        /* Update my rows using row k */
        for(int i = 0; i < my_rows; i++){
            int dik = local[IDX(i, k, n)];
            if(dik >= INF){
                continue;
            }
            for(int j = 0; j < n; j++){
                int dkj = row_k[j];
                if(dkj >= INF){
                    continue;
                }
                int through_k = dik + dkj;
                if(through_k < local[IDX(i, j, n)]){
                    local[IDX(i, j, n)] = through_k;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_total = MPI_Wtime() - t_start;

    MPI_Gatherv(local, my_rows * n, MPI_INT, D, sendcnt, displs, MPI_INT, 0, MPI_COMM_WORLD);

    double max_total, max_comm;
    MPI_Reduce(&t_total,   &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &max_comm,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0){
        if(print){
            printf("Shortest paths:\n"); print_matrix(D, n);
        }
        long long sum, infs;
        matrix_checksum(D, n, &sum, &infs);
        printf("MPI n=%d procs=%d density=%.2f seed=%u time=%.6f s comm=%.6f s checksum=%lld inf_count=%lld\n",
               n, size, density, seed, max_total, max_comm, sum, infs);
        free(D);
    }

    free(local); free(row_k);
    free(rows_per); free(first_row); free(sendcnt); free(displs); free(owner);
    MPI_Finalize();
    return 0;
}