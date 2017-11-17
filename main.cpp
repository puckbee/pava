//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
#include <omp.h>
#include <unistd.h>
#include <sys/mman.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <list>
#include <vector>
#include <set>

#include <immintrin.h>
#include <mkl.h>

#include "utils/microtime.h"
#include "utils/basicMatrix.h"
#include "utils/pava_formats.h"

using namespace std;

#ifndef TYPE_PRECISION
#define TYPE_PRECISION

//#define TYPE_SINGLE
#define TYPE_DOUBLE

#define FLOAT_TYPE double

#ifdef TYPE_SINGLE
#define SIMD_LEN 16
#else
#define SIMD_LEN 8
#endif

#endif


int main(int argc, char** argv)
{
    cout<<" =================================       now we begins PAVA     ==================================="<<endl;

    if ( argc < 2 )
    {
        fprintf( stderr, "Insufficient number of input parameters:\n");
        fprintf( stderr, "File name with sparse matrix for benchmarking is missing\n" );
        fprintf( stderr, "Usage: %s [martix market filename]\n", argv[0] );
        exit(1);
    }

    char* matrixName = argv[1];
    double *y, *y_ref;
    double *x;


    struct _PAVA_COOMatrix* cooMatrix = (struct _PAVA_COOMatrix*) malloc(sizeof(struct _PAVA_COOMatrix));
    struct _PAVA_CSRMatrix* csrMatrix = (struct _PAVA_CSRMatrix*) malloc(sizeof(struct _PAVA_CSRMatrix));

    if ( 0 != readCOOMatrix( matrixName, cooMatrix ) )
    {
        fprintf(stderr, "Reading COO matrix in matrix market format failed\n" );
        return -2;
    }

    std::cout<<"Basic informations"<<endl;
    std::cout<<"        numRows = "<<cooMatrix->numRows<<endl;
    std::cout<<"        numCols = "<<cooMatrix->numCols<<endl;
    std::cout<<"            nnz = "<<cooMatrix->nnz<<endl;



    convertCOO2CSR(cooMatrix, csrMatrix);

    // Align allocated memory to boost performance 
#ifdef MMAP
    x     = (double*)mmap(0, csrMatrix->numCols* sizeof(double), PROT_READ|PROT_WRITE,MAP_ANONYMOUS|MAP_PRIVATE|MAP_HUGETLB,-1,0);
#else
    x     = ( double* ) MKL_malloc ( csrMatrix->numCols * sizeof( double ), ALIGN );
#endif
    y     = ( double* ) MKL_malloc ( csrMatrix->numRows * sizeof( double ), ALIGN );
    y_ref = ( double* ) MKL_malloc ( csrMatrix->numRows * sizeof( double ), ALIGN );

    if ( NULL == x || NULL == y || NULL == y_ref )
    {
        fprintf( stderr, "Could not allocate memory for vectors!\n" );
        MKL_free( x );
        MKL_free( y );
        MKL_free( y_ref );

        deleteCSRMatrix( csrMatrix );
        return -1;
    }

    initVectors( csrMatrix->numRows, csrMatrix->numCols, x, y, y_ref );

    referenceSpMV(csrMatrix, x, y_ref);
    double  alpha = 1;
    double beta = 1;

    benchmark_MKL_SpMV( csrMatrix, alpha, x, beta, y, y_ref, 0, matrixName, 1000);


    return 0;
}































