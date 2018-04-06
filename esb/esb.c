/*******************************************************************************
*   Copyright(C) 2013 Intel Corporation. All Rights Reserved.
*
*   The source code, information  and  material ("Material") contained herein is
*   owned  by Intel Corporation or its suppliers or licensors, and title to such
*   Material remains  with Intel Corporation  or its suppliers or licensors. The
*   Material  contains proprietary information  of  Intel or  its  suppliers and
*   licensors. The  Material is protected by worldwide copyright laws and treaty
*   provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
*   modified, published, uploaded, posted, transmitted, distributed or disclosed
*   in any way  without Intel's  prior  express written  permission. No  license
*   under  any patent, copyright  or  other intellectual property rights  in the
*   Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
*   implication, inducement,  estoppel or  otherwise.  Any  license  under  such
*   intellectual  property  rights must  be express  and  approved  by  Intel in
*   writing.
*
*   *Third Party trademarks are the property of their respective owners.
*
*   Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
*   this  notice or  any other notice embedded  in Materials by Intel or Intel's
*   suppliers or licensors in any way.
*
********************************************************************************
*
*   Content :  Double-precision performance benchmark for
*              Intel(R) MKL SpMV Format Prototype Package, version 0.2
*
*   usage:  ./exec.file sparse_matrix_in_matrix_market_format
*
********************************************************************************/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/mman.h>

#include <omp.h>
#include <mkl.h>

#ifdef __KNC__
    #include "./spmv_interface.h"
#endif

#include "./esb.h"

#include "../utils/basicMatrix.h"

#include <time.h>
#include <sys/time.h>
#define MICRO_IN_SEC 1000000.00

double fetchtime(){
        int tv_sec,tv_usec;
        double time;
        struct timeval tv;
        struct timezone tz;
        gettimeofday(&tv,&tz);

        return tv.tv_sec+tv.tv_usec/MICRO_IN_SEC;
}

// alignment for memory allocation in mkl_malloc
#define ALIGN 64
// threshold for validation of SpMV results 
#define EPS 1.0e-15

/*
// This structure is used for storing sparse matrices in COO and CSR formats
struct SparseMatrix
{
    int num_rows;   // number of rows in the matrix
    int num_cols;   // number of columns in the matrix
    int nnz;        // number of non-zero elements in the matrix
    int *rows;      // array with row indices
    int *cols;      // array with column indices
    double *vals;   // array with non-zero elements
};
*/


double calcFrobeniusNorm ( int vectorLength, double *vectorValues );

/***********************************************************************************************/

/*
This function provides reference implementation of SpMV
functionality:
y = alpha * A * x + beta * y.
alpha, beta - scalars,
x - input vector,
y - resulting vector,
A - sparse matrix in 0-based CSR representation:
rows - number of rows in the matrix A
csrRows - integer array of length rows+1
csrCols - integer array of length nnz
csrVals - double  array of length nnz
*/

void referenceSpMV2 ( const struct SparseMatrix *csrMatrix,
                            double       alpha,
                            const double *x,
                            double       beta,
                            double       *y )
{
    int i;
    int rows = csrMatrix->num_rows;
#pragma omp parallel for
    for ( i = 0; i < rows; i++ )
    {
        double yi = 0.0;
        int start = csrMatrix->rows[i];
        int end   = csrMatrix->rows[i + 1];
        int j;
        for ( j = start; j < end; j++ )
            yi += csrMatrix->vals[j] * x[csrMatrix->cols[j]];
        y[i] = yi * alpha + beta * y[i];
    }
}   // referenceSpMV

// Prints measured performance results for all SpMV benchmarks: Intel MKL, CSR, and ESB.
// parameter schedule = -1 indicates that no schedule-related message will be printed 
void printPerformanceResults( const char* testName, double mflop, double bench_time, int niters, int schedule )
{
    double time = bench_time / niters;      //  time for a single SpMV call
    double gflops = mflop / time / 1000;
    if ( bench_time < 0.5 )
    {
        printf( "Warning: measured time is less than 0.5 sec: %f\n", time );
        printf( "Measured results could be unstable\n" );
    }

    printf( "%s performance results:\n", testName );
    printf( "   SpMV GFlop/s: %f\n", gflops );
    printf( "   SpMV time:    %f\n", time );
    switch ( schedule )
    {
        case INTEL_SPARSE_SCHEDULE_STATIC:  printf( "   schedule:     static\n" );    break;
        case INTEL_SPARSE_SCHEDULE_DYNAMIC: printf( "   schedule:     dynamic\n" );   break;
        case INTEL_SPARSE_SCHEDULE_BLOCK:   printf( "   schedule:     block\n" );     break;
    }
}   // printPerformanceResults

// Initialize input vectors
void initVectors2( int    rows,
                         int    cols,
                         double *x,
                         double *y,
                         double *y_ref )
{
    int i;
    for ( i = 0; i < rows; i++ )
    {
//        y[i] = M_PI;
//        y_ref[i] = M_PI;
        y[i] = 0;
        y_ref[i] = 0;
    }
    for ( i = 0; i < cols; i++ )
    {
//        x[i] = M_PI;
        x[i] = 1;
    }
}   // initVectors


// Calculate Frobenius norm of the matrix and vectors: 
static double calcFrobeniusNorm ( int vectorLength, double *vectorValues )
{
    int i;
    double norm = 0.0;
    for ( i = 0; i < vectorLength; i++ )
    {
        norm += vectorValues[i] * vectorValues[i];
    }
    return sqrt (norm) ;
}   // calcFrobeniusNorm

// Calculate Frobenius norm of vectors y and y_ref residual
static double calculateResidual ( int          size,
                                  const double *y,
                                  const double *y_ref )
{
    double res = 0.0;
    int i;

    for ( i = 0; i < size; i++ )
    {
        res += ( y[i] - y_ref[i] ) * ( y[i] - y_ref[i] );
    }

    res = sqrt ( res );

    return res;
}   // calculateResidual







// Validate and measure performance of experimental ESB SpMV implementation
int benchmark_ESB_SpMV ( const struct _PAVA_CSRMatrix *csrMatrix,
                                double       alpha,
                                double       *x,
                                double       beta,
                                double       *y,
                                double       *y_ref,
                                double       mflop,
                                double       matrixFrobeniusNorm,
                                sparseSchedule_t schedule,
                                char *       matrixName,
                                int          numIterations)
{
    sparseESBMatrix_t esbA;     // Structure with ESB matrix
    sparseMatDescr_t descrA;    // ESB matrix descriptor
//    double time_start, time_end, time;
    double normX, normY, residual;
    double estimatedAccuracy;

    double t1, t2, t3, t4;
    int omp_threads, mkl_threads;

    omp_threads = omp_get_max_threads();
    mkl_set_num_threads_local(omp_threads);
    mkl_threads = mkl_get_max_threads();

    // y = y_ref
//    initVectors2( csrMatrix->numRows, csrMatrix->numCols, x, y, y_ref );
    // y_ref = alpha * A * x + beta * y_ref


    normX = calcFrobeniusNorm ( csrMatrix->numCols, x );
    normY = calcFrobeniusNorm ( csrMatrix->numRows, y );


    // estimate accuracy of SpMV oparation: y = alpha * A * x + beta * y
    // || y1 - y2 || < eps * ( |alpha| * ||A|| * ||x|| + |beta| * ||y||)
    estimatedAccuracy = fabs( alpha ) * matrixFrobeniusNorm * normX + fabs( beta ) * normY;
    estimatedAccuracy *= EPS;

//    referenceSpMV2 ( csrMatrix, alpha, x, beta, y_ref );

    /* Functions below could return an error in the following situations:
   INTEL_SPARSE_STATUS_ALLOC_FAILED - not enough memory to allocate working arrays;
   INTEL_SPARSE_STATUS_EXECUTION_FAILED - implemented algorithm reported wrong result. */

    if(schedule == INTEL_SPARSE_SCHEDULE_DYNAMIC)
        std::cout<<" Converting CSR->ESB-dynamic"<<std::endl;
    else if(schedule == INTEL_SPARSE_SCHEDULE_STATIC)
        std::cout<<" Converting CSR->ESB-static"<<std::endl;

//    double time_pre = fetchtime();

    t1 = fetchtime();

    if ( sparseCreateESBMatrix ( &esbA, schedule ) != INTEL_SPARSE_STATUS_SUCCESS )
    {
        fprintf( stderr, "Error after creation of ESB matrix\n" );
        return -1;
    }

    if ( sparseCreateMatDescr ( &descrA ) != INTEL_SPARSE_STATUS_SUCCESS )
    {
        fprintf( stderr, "Error after creation of matrix descriptor\n" );
        sparseDestroyESBMatrix ( esbA );
        return -2;
    }

    if ( sparseDcsr2esb ( csrMatrix->numRows, csrMatrix->numCols, descrA, csrMatrix->vals, csrMatrix->rowOffsets, csrMatrix->cols, esbA ) != INTEL_SPARSE_STATUS_SUCCESS )
    {
        fprintf( stderr, "Error after conversion to ESB matrix\n" );
        sparseDestroyESBMatrix ( esbA );
        sparseDestroyMatDescr ( descrA );
        return -3;
    }

//    printf(" **********************************************\n");
//    printf(" Pre-processing Time of ESB is %f\n", fetchtime()-time_pre);
//    printf(" **********************************************\n");
     
//    printf("ESB PRE Time %s %f \n", filename, fetchtime() - time_pre);
    t2 = fetchtime();

/*
    // y = alpha * A * x + beta * y
    if ( sparseDesbmv ( INTEL_SPARSE_OPERATION_NON_TRANSPOSE, &alpha, esbA, x, &beta, y ) != INTEL_SPARSE_STATUS_SUCCESS )
    {
        fprintf( stderr, "Error after SpMV in ESB format\n" );
        sparseDestroyESBMatrix ( esbA );
        sparseDestroyMatDescr ( descrA );
        return -4;
    }

    // check for equality of y_ref and y
//    residual = calculateResidual ( csrMatrix->num_rows, y, y_ref );
    if ( residual > estimatedAccuracy )
    {
        // The library implementation of SpMV probably computed a wrong result
        fprintf( stderr, "ERROR: the difference is too high. Residual %e is above threshold %f\n",
                 residual, estimatedAccuracy );
        sparseDestroyESBMatrix ( esbA );
        sparseDestroyMatDescr ( descrA );
        return -5;
    }
    else
    {
        printf( "Validation PASSED\n" );
    }
*/    
    // estimate number of iterations: measured time should be long enough for stable measurement
//    niters = (int) ( 2.0e3 / mflop );
/*
#ifdef NITERS
    niters = NITERS;
#else   
//    niters = 1000;
    niters = numIterations;
#endif
*/

    if(schedule == INTEL_SPARSE_SCHEDULE_DYNAMIC)
        std::cout<<" Executing ESB-dynamic"<<std::endl;
    else if(schedule == INTEL_SPARSE_SCHEDULE_STATIC)
        std::cout<<" Executing ESB-static"<<std::endl;

    t3 = fetchtime();
    for (int iter = 0; iter < numIterations; iter++ )
        sparseDesbmv ( INTEL_SPARSE_OPERATION_NON_TRANSPOSE, &alpha, esbA, x, &beta, y );
    t4 = fetchtime();

    if(schedule == INTEL_SPARSE_SCHEDULE_DYNAMIC)
        printPerformance(matrixName, "ESB-d", mkl_threads, t2 - t1, (t4-t3)/numIterations);
    else if(schedule == INTEL_SPARSE_SCHEDULE_STATIC)
        printPerformance(matrixName, "ESB-s", mkl_threads, t2 - t1, (t4-t3)/numIterations);
    
    initVectors( csrMatrix->numRows, csrMatrix->numCols, NULL, y, NULL );
    if ( sparseDesbmv ( INTEL_SPARSE_OPERATION_NON_TRANSPOSE, &alpha, esbA, x, &beta, y ) != INTEL_SPARSE_STATUS_SUCCESS )
    {
        fprintf( stderr, "Error after SpMV in ESB format\n" );
        sparseDestroyESBMatrix ( esbA );
        sparseDestroyMatDescr ( descrA );
        return -4;
    }

    // print performance results in GFlops and time per single SpMV call
//    printPerformanceResults( "ESB SpMV", mflop, time, niters, schedule );
//    printf("ESB EXE Time %s %f \n", filename, time/niters);

    sparseDestroyESBMatrix ( esbA );
    sparseDestroyMatDescr ( descrA );
    return 0;
}   // benchmark_ESB_SpMV
/*
void main(int argc, char** argv)
{
 std::cout<<" xx"<<std::endl;
}
*/
