#include <stdlib.h>
#include <stdio.h>
//#include <offload.h>

#include <unistd.h>
#include <sys/mman.h>


#include <time.h>
#include <sys/time.h>
#define MICRO_IN_SEC 1000000.00


#include <iostream>
#include <cmath>

#include "anonymouslib_avx512.h"

#include "../utils/mmio.h"
#include "../utils/microtime.h"
#include "../utils/basicMatrix.h"

using namespace std;

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#ifndef NUM_RUN
#define NUM_RUN 1000
#endif

/*
double microtime(){
        int tv_sec,tv_usec;
        double time;
        struct timeval tv;
        struct timezone tz;
        gettimeofday(&tv,&tz);

        return tv.tv_sec+tv.tv_usec/MICRO_IN_SEC;
}
*/

double runKernel(anonymouslibHandle<int, unsigned int, VALUE_TYPE> A, VALUE_TYPE* y_bench, VALUE_TYPE alpha, int numIterations)
{
    
//       anonymouslib_timer CSR5Spmv_timer;

//        printf(" Iterations : %d\n", NUM_RUN);

        int err = 0;
/*
        for (int i = 0; i < 50; i++)
            err = A.spmv(alpha, y_bench);
*/
//        CSR5Spmv_timer.start();

        // time spmv by running NUM_RUN times
        for (int i = 0; i < numIterations; i++)
        {
            err = A.spmv(alpha, y_bench);
//            std::cout<<" y-"<<i<<" = "<< y_bench[3633]<<endl;
        }
/*
        double CSR5Spmv_time = CSR5Spmv_timer.stop() / (double)NUM_RUN;

        return CSR5Spmv_time;
*/
}

int conduct_csr5(int m, int n, int nnzA,
                  int *csrRowPtrA, int *csrColIdxA, VALUE_TYPE *csrValA,
                  VALUE_TYPE *x, VALUE_TYPE *y, VALUE_TYPE alpha, char *matrixName, int numIterations)
{
    int err = 0;

    VALUE_TYPE *y_bench = (VALUE_TYPE *)_mm_malloc(sizeof(VALUE_TYPE) * m, ANONYMOUSLIB_X86_CACHELINE);
//    double gb = getB<int, VALUE_TYPE>(m, nnzA);
//    double gflop = getFLOP<int>(nnzA);

//    anonymouslib_timer asCSR5_timer;
//    anonymouslib_timer ref_timer;

//    printf("omp_get_max_threads = %i\n", omp_get_max_threads());

    /*
    ref_timer.start();

    int ref_iter = 1000;
    #pragma omp parallel for
    for (int iter = 0; iter < ref_iter; iter++)
    {
        for (int i = 0; i < m; i++)
        {
            VALUE_TYPE sum = 0;
            for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
                sum += x[csrColIdxA[j]] * csrValA[j];
            y_bench[i] = sum;
        }
    }

    double ref_time = ref_timer.stop() / (double)ref_iter;

    printf("CSR-based SpMV OMP time = %f ms. Bandwidth = %f GB/s. GFlops = %f GFlops.\n\n",
               ref_time, gb/(1.0e+6 * ref_time), gflop/(1.0e+6 * ref_time));
    */

    // y for back to check the correctness, y_iter for the iterative spmv
    VALUE_TYPE *y_iter = (VALUE_TYPE *)_mm_malloc(sizeof(VALUE_TYPE) * m, ANONYMOUSLIB_X86_CACHELINE);

    anonymouslibHandle<int, unsigned int, VALUE_TYPE> A(m, n);
    err = A.inputCSR(nnzA, csrRowPtrA, csrColIdxA, csrValA);
    //cout << "inputCSR err = " << err << endl;

    err = A.setX(x); // you only need to do it once!
    //cout << "setX err = " << err << endl;

    int sigma = ANONYMOUSLIB_CSR5_SIGMA; //nnzA/(8*ANONYMOUSLIB_CSR5_OMEGA);
    A.setSigma(sigma);
/*
    A.asCSR5();
    A.asCSR();
*/
    // record a correct CSR->CSR5 time without PCIe overhead
//    asCSR5_timer.start();

    std::cout<<" Converting CSR->CSR5"<<endl;
    double t1 = microtime();
    err = A.asCSR5();
    double t2 = microtime();

//    printf("CSR->CSR5 time = %f ms.\n", asCSR5_timer.stop());
//    printf("CSR5 PRE Time %s  %f s. \n", filename, t2 - t1);
//    printf("The Pre-processing(CSR->CSR5) Time of CSR5 is %f seconds. \n", t2 - t1);
    //cout << "asCSR5 err = " << err << endl;
   
    // check correctness by running 1 time
//    err = A.spmv(alpha, y);
    //cout << "spmv err = " << err << endl;

//    double CSR5Spmv_gflops = 0;
    // warm up by running 50 times
   
//    double CSR5Spmv_time = 0; 

    std::cout<<" Executing CSR5"<<endl;
    // return to check correctness
    runKernel(A, y, alpha, 1);

    double t3 = microtime();    
    runKernel(A, y_iter, alpha, numIterations);
    double t4 = microtime();

    A.asCSR();

    int omp_threads = omp_get_max_threads();
    printPerformance(matrixName, "CSR5", omp_threads, t2 - t1, (t4 - t3)/numIterations);


//    printf("CSR5-based SpMV AVX512 time = %f ms. Bandwidth = %f GB/s. GFlops = %f GFlops.\n",
//    CSR5Spmv_time, gb/(1.0e+6 * CSR5Spmv_time), gflop/(1.0e+6 * CSR5Spmv_time));

//    printf("CSR5 EXE Time %s  %f s. \n", filename , CSR5Spmv_time/1000) ;
//    printf("The SpMV Execution time of CSR5 is %f seconds. \n", CSR5Spmv_time/1000);
//    printf(" Throughput of CSR5 is %f GFlops\n", nnzA/CSR5Spmv_time/1000000);
//    CSR5Spmv_gflops = gflop/(1.0e+6 * CSR5Spmv_time);
/*
    // write results to text (scv) file
    FILE *fout = fopen("results.csv", "a");
    if (fout == NULL)
        printf("Writing results fails.\n");
    fprintf(fout, "%s,%f\n", filename, CSR5Spmv_gflops);
    fclose(fout);
*/
//    _mm_free(y_bench);

    return err;
}
