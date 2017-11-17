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

#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "mkl.h"
#include <math.h>
#include <omp.h>

#include <unistd.h>
#include <sys/mman.h>

#include "basicMatrix.h"


/*
Sparse Matrix auxiliary routines:

    readSparseCOOMatrix - read input matrix in COO format from the file;
        Input file should be in Matrix Matrix format.

    convertCOO2CSR - create matrix in CSR format from COO representation.
        Elements in rows of CSR matrix are sorted by column numbers in
        increasing order

    deleteSparseMatrix - free memory previously allocated for the matrix

    printMatrixInfo - prints short sparse matrix statistics
*/

int readCOOMatrix ( char* matrixName, struct _PAVA_COOMatrix *cooMatrix )
{
    FILE *f;
    if ( (f = fopen(matrixName, "r")) == NULL )
    {
        fprintf( stderr, "Error opening input file: %s\n", matrixName );
        exit(1);
    }
    else
    {
        printf( "Input file: %s\n", matrixName );
    }

    MM_typecode matcode;
    int isComplex, isInteger, isReal, isSymmetric, isPattern;
    int sizeM, sizeN, sizeV, nnz, ret_code, counter, idum, i;
    double ddum;
    int *rows = NULL, *cols = NULL;
    double *vals = NULL;

    if ( mm_read_banner(f, &matcode) != 0 )
    {
        fprintf( stderr, "Could not process matrix market banner.\n");
        return -1;
    }
    if ( !mm_is_matrix( matcode ) )
    {
        fprintf( stderr, "Could not process non-matrix input.\n");
        return -2;
    }

    if ( !mm_is_sparse( matcode ) )
    {
        fprintf( stderr, "Could not process non-sparse matrix input.\n");
        return -3;
    }

    isComplex = 0;
    isReal    = 0;
    isInteger = 0;
    isSymmetric = 0;
    isPattern = 0;

    if ( mm_is_complex( matcode ) )
    {
        isComplex = 1;
    }

    if ( mm_is_real ( matcode) )
    {
        isReal = 1;
    }

    if ( mm_is_integer ( matcode ) )
    {
        isInteger = 1;
    }

    if ( mm_is_pattern ( matcode ) )
    {
        isPattern = 1;
    }

    /* find out size of sparse matrix .... */

    if ( ( ret_code = mm_read_mtx_crd_size( f, &sizeM, &sizeN, &sizeV ) ) != 0 )
    {
        fprintf( stderr, "Could not process matrix sizes\n");
        return -4;
    }

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric = 1;
        nnz = sizeV * 2; // up to two times more elements than in half of the matrix
    }
    else
    {
        nnz = sizeV;
    }

    /* allocate memory for matrices */

    rows = ( int* )    MKL_malloc( nnz * sizeof( int ), ALIGN );
    cols = ( int* )    MKL_malloc( nnz * sizeof( int ), ALIGN );
    vals = ( double* ) MKL_malloc( nnz * sizeof( double ), ALIGN );

    if ( NULL == rows || NULL == cols || NULL == vals )
    {
        MKL_free( rows );
        MKL_free( cols );
        MKL_free( vals );
        fprintf( stderr, "Could not allocate memory for input matrix arrays in COO format\n" );
        fprintf( stderr, "Rows = %d, Columns = %d, NNZ = %d\n", sizeM, sizeN, nnz );
        return -5;
    }

    counter = 0;

    for ( i = 0; i < sizeV; i++ )
    {
        if ( isComplex )
        {
            fscanf(f, "%d %d %lg %lg\n", &rows[counter], &cols[counter], &vals[counter], &ddum );
        }
        else if ( isReal )
        {
            fscanf(f, "%d %d %lg\n", &rows[counter], &cols[counter], &vals[counter] );
        }
        else if ( isInteger )
        {
            fscanf(f, "%d %d %d\n", &rows[counter], &cols[counter], &idum );
            vals[counter] = idum;
        }
        else if ( isPattern )
        {
            fscanf(f, "%d %d\n", &rows[counter], &cols[counter] );
            vals[counter] = 1;
        }
        counter++;
        if ( isSymmetric && rows[counter-1] != cols[counter-1] )
        // expand symmetric formats to "general" one
        {
            rows[counter] = cols[counter-1];
            cols[counter] = rows[counter-1];
            vals[counter] = vals[counter-1];
            counter++;
        }
    }

    if ( f !=stdin ) fclose(f);

    printf("Reading matrix completed\n" );

    cooMatrix->numRows = sizeM;
    cooMatrix->numCols = sizeN;
    cooMatrix->nnz   = counter; // Actual number of non-zeroes elements in COO matrix
    cooMatrix->rows  = rows;
    cooMatrix->cols  = cols;
    cooMatrix->vals  = vals;
    return 0;
}   // readSparseCOOMatrix

int convertCOO2CSR ( const struct _PAVA_COOMatrix *cooMatrix, struct _PAVA_CSRMatrix *csrMatrix )
{
    int info;
    int job[8];

    /************************/
    /* now convert matrix in COO 1-based format to CSR 0-based format */
    /************************/

    job[0] = 2; // COO -> sorted CSR
    job[1] = 0; // 0-based CSR
    job[2] = 0; // 1-based COO
    job[4] = cooMatrix->nnz;
    job[5] = 0; // all CSR arrays are filled

    info = 0;

    csrMatrix->numRows = cooMatrix->numRows;
    csrMatrix->numCols = cooMatrix->numCols;
    csrMatrix->nnz   = cooMatrix->nnz;

    csrMatrix->rowOffsets = ( int* )    MKL_malloc( ( cooMatrix->numRows + 1 ) * sizeof( int ), ALIGN );
    csrMatrix->cols = ( int* )    MKL_malloc( cooMatrix->nnz * sizeof( int ),           ALIGN );
    csrMatrix->vals = ( double* ) MKL_malloc( cooMatrix->nnz * sizeof( double ),        ALIGN );

    if ( NULL == csrMatrix->rowOffsets || NULL == csrMatrix->cols || NULL == csrMatrix->vals )
    {
        MKL_free( csrMatrix->rowOffsets );
        MKL_free( csrMatrix->cols );
        MKL_free( csrMatrix->vals );
        fprintf( stderr, "Could not allocate memory for converting matrix to CSR format\n" );
        return -5;
    }

    mkl_dcsrcoo ( job,
                  &csrMatrix->numRows,
                  csrMatrix->vals,
                  csrMatrix->cols,
                  csrMatrix->rowOffsets,
                  (int*)&cooMatrix->nnz,
                  cooMatrix->vals,
                  cooMatrix->rows,
                  cooMatrix->cols,
                  &info );

    if ( info != 0 )
    {
        fprintf( stderr, "Error converting COO -> CSR: %d\n", info );
        MKL_free( csrMatrix->rowOffsets );
        MKL_free( csrMatrix->cols );
        MKL_free( csrMatrix->vals );
        return -10;
    }

/*
    int kkkk=0;
    for(kkkk=0; kkkk<256; kkkk++)
       printf(" cols[%d]= %d, vals[%d]=%f\n", kkkk, csrMatrix->cols[kkkk], kkkk, csrMatrix->vals[kkkk]);
*/

    printf( "Operation COO->CSR completed\n" );
    return 0;
}   // convertCOO2CSR

void deleteCOOMatrix ( struct _PAVA_COOMatrix *matrix )
{
    MKL_free( matrix->rows );
    MKL_free( matrix->cols );
    MKL_free( matrix->vals );
}   // deleteSparseMatrix

void deleteCSRMatrix ( struct _PAVA_CSRMatrix *matrix )
{
    MKL_free( matrix->rowOffsets );
    MKL_free( matrix->cols );
    MKL_free( matrix->vals );
}   // deleteSparseMatrix

void printMatrixInfo ( const struct SparseMatrix *csrMatrix )
{
    int omp_threads;

    int env_threads = omp_get_max_threads();

    mkl_set_num_threads_local(env_threads);

    omp_threads = mkl_get_max_threads();

    printf( "Number of OMP threads: %d\n", omp_threads );
    printf( "Sparse matrix info:\n" );
    printf( "       rows: %d\n", csrMatrix->num_rows );
    printf( "       cols: %d\n", csrMatrix->num_cols );
    printf( "       nnz:  %d\n", csrMatrix->nnz );
}   // printMatrixInfo



// Initialize input vectors
void initVectors( int numRows, int numCols, double *x, double *y, double *y_ref )
{
    int i;
    for ( i = 0; i < numRows; i++ )
    {
//        y[i] = M_PI;
//        y_ref[i] = M_PI;
        y[i] = 0;
        y_ref[i] = 0;
    }
    for ( i = 0; i < numCols; i++ )
    {
//        x[i] = M_PI;
        x[i] = 1;
    }
}   // initVectors



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

void referenceSpMV ( struct _PAVA_CSRMatrix *csrMatrix, double *x, double *y_ref )
{
    int i;
    int numRows = csrMatrix->numRows;
#pragma omp parallel for
    for ( i = 0; i < numRows; i++ )
    {
        double yi = 0.0;
        int start = csrMatrix->rowOffsets[i];
        int end   = csrMatrix->rowOffsets[i + 1];
        int j;
        for ( j = start; j < end; j++ )
            yi += csrMatrix->vals[j] * x[csrMatrix->cols[j]];
//        y[i] = yi * alpha + beta * y[i];
        y_ref[i] = yi;
    }
}   // referenceSpMV


double checkResults ( int size, const double *y, const double *y_ref )
{
    double res = 0.0;
    int i;

    for ( i = 0; i < size; i++ )
    {
        res += ( y[i] - y_ref[i] ) * ( y[i] - y_ref[i] );
    }

    res = sqrt ( res );

    return res;
}   // checkResults






