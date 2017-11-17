
#include <iostream>
#include <mkl.h>
#include "basicMatrix.h"

#include "pava_formats.h"

using namespace std;

int  benchmark_MKL_SpMV ( struct _PAVA_CSRMatrix *csrMatrix, double alpha, double *x, double beta, double *y, double *y_ref, double matrixFrobeniusNorm, char* filename, int numIterations )
{


    char transa = 'n';
    char matdescra[6] = {'G', 'x', 'x', 'C', 'x', 'x'};


    mkl_dcsrmv( &transa, (int*)&csrMatrix->numRows, (int*)&csrMatrix->numCols, &alpha, matdescra, csrMatrix->vals, csrMatrix->cols, csrMatrix->rowOffsets, &csrMatrix->rowOffsets[1], x, &beta, y );
    // check for equality of y_ref and y

    double res = checkResults(csrMatrix->numRows, y, y_ref);

    if(res==0)
        std::cout<<" Results right"<<endl;
    else
        std::cout<<" Error: "<< res <<endl;

}
