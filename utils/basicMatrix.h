#ifndef BASICMATRIX_H
#define BASICMATRIX_H


#include <stdio.h>
#include <stdlib.h>

#define ALIGN512 64

#include <immintrin.h>

typedef __attribute__((aligned(64))) union zmmd {                     
        __m512d reg;        
        __m512i regi32;      
        double elems[8]; 
        int elemsi32[16];
} zmmd_t;







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


// This structure is used for storing sparse matrices in COO format
struct _PAVA_COOMatrix
{
    int nnz;        // number of non-zero elements in the matrix
    int numRows;   // number of rows in the matrix
    int numCols;   // number of columns in the matrix

    double *vals;   // array with non-zero elements
    int *cols;      // array with column indices
    int *rows;      // array with row indices
};


// This structure is used for storing sparse matrices in CSR formats
struct _PAVA_CSRMatrix
{
    int nnz;        // number of non-zero elements in the matrix
    int numRows;   // number of rows in the matrix
    int numCols;   // number of columns in the matrix

    double *vals;   // array with non-zero elements
    int *cols;      // array with column indices
    int *rowOffsets;      // array with row offsets
};


// This structure is used for storing sparse matrices in CSR formats
struct _PAVA_CSCMatrix
{
    int nnz;        // number of non-zero elements in the matrix
    int numRows;   // number of rows in the matrix
    int numCols;   // number of columns in the matrix

    double *vals;   // array with non-zero elements
    int *rows;      // array with column indices
    int *colOffsets;      // array with row offsets
};


// This structure is used for storing sparse matrices in DIA formats
struct _PAVA_DIAMatrix
{
    int nnz;        // number of non-zero elements in the matrix
    int numRows;   // number of rows in the matrix
    int numCols;   // number of columns in the matrix

    int idiag;      // leading deminsion
    int ndiag;      // number of diagnals

    double *vals;   // array with non-zero elements
    int *distance;      // distance array
};

// This structure is used for storing sparse matrices in BSR formats
struct _PAVA_BSRMatrix
{
    int nnz;        // number of non-zero elements in the matrix
    int numRows;   // number of rows in the matrix
    int numCols;   // number of columns in the matrix

    int nBlocks;
    int nBlockRows;
    int sizeBlock;
    int leadingBlock;


    double *vals;   // array with non-zero elements
    int *cols;      
    int *rowIdx;


    double* y_bsr;

};


double calcFrobeniusNorm ( int vectorLength, double *vectorValues );
int readCOOMatrix ( char* matrixName, struct _PAVA_COOMatrix *cooMatrix );

int convertCOO2SquareCSR ( const struct _PAVA_COOMatrix *cooMatrix, struct _PAVA_CSRMatrix *csrMatrix );
int convertCOO2CSR ( const struct _PAVA_COOMatrix *cooMatrix, struct _PAVA_CSRMatrix *csrMatrix );

void deleteCOOMatrix ( struct _PAVA_COOMatrix *matrix );
void deleteCSRMatrix ( struct _PAVA_CSRMatrix *matrix );

void printMatrixInfo ( const struct SparseMatrix *csrMatrix );

int printPerformance(char* matrixName, char* format, int numThreads, double convertTime, double executionTime);

void initVectors( int numRows, int numCols, double *x, double *y, double *y_ref );


void referenceSpMV ( struct _PAVA_CSRMatrix *csrMatrix, double *x, double *y_ref );


int checkResults ( int size, const double *y, const double *y_ref );

void deleteCSCMatrix ( struct _PAVA_CSCMatrix *matrix );

void deleteDIAMatrix ( struct _PAVA_DIAMatrix *matrix );

void deleteBSRMatrix ( struct _PAVA_BSRMatrix *matrix );
#endif
