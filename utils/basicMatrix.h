#ifndef BASICMATRIX_H
#define BASICMATRIX_H


#include <stdio.h>
#include <stdlib.h>

#define ALIGN 512

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

int readCOOMatrix ( char* matrixName, struct _PAVA_COOMatrix *cooMatrix );

int convertCOO2CSR ( const struct _PAVA_COOMatrix *cooMatrix, struct _PAVA_CSRMatrix *csrMatrix );

void deleteCOOMatrix ( struct _PAVA_COOMatrix *matrix );
void deleteCSRMatrix ( struct _PAVA_CSRMatrix *matrix );

void printMatrixInfo ( const struct SparseMatrix *csrMatrix );


void initVectors( int numRows, int numCols, double *x, double *y, double *y_ref );


void referenceSpMV ( struct _PAVA_CSRMatrix *csrMatrix, double *x, double *y_ref );


double checkResults ( int size, const double *y, const double *y_ref );

#endif
