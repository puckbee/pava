
#include <iostream>
#include <mkl.h>
#include <assert.h>
#include <memory.h>
#include <unistd.h>
#include <sys/mman.h>
#include <omp.h>

#include "basicMatrix.h"
#include "pava_formats.h"



#include <time.h>
#include <sys/time.h>
#define MICRO_IN_SEC 1000000.00

double catchtime(){
        int tv_sec,tv_usec;
        double time;
        struct timeval tv;
        struct timezone tz;
        gettimeofday(&tv,&tz);

        return tv.tv_sec+tv.tv_usec/MICRO_IN_SEC;
}






using namespace std;




int  benchmark_CSR_SpMV ( struct _PAVA_CSRMatrix *csrMatrix, double alpha, double *x, double beta, double *y, double *y_ref, double matrixFrobeniusNorm, char* filename, int numIterations )
{

    char transa = 'n';
    char matdescra[6] = {'G', 'x', 'x', 'C', 'x', 'x'};
    for(int i=0; i<numIterations; i++)
    mkl_dcsrmv( &transa, (int*)&csrMatrix->numRows, (int*)&csrMatrix->numCols, &alpha, matdescra, csrMatrix->vals, csrMatrix->cols, csrMatrix->rowOffsets, &csrMatrix->rowOffsets[1], x, &beta, y );

}


int  benchmark_COO_SpMV ( struct _PAVA_COOMatrix *cooMatrix, double alpha, double *x, double beta, double *y, double *y_ref, double matrixFrobeniusNorm, char* filename, int numIterations )
{

    char transa = 'n';
    char matdescra[6] = {'G', 'x', 'x', 'C', 'x', 'x'};

    for(int i=0; i<numIterations; i++)
    mkl_dcoomv( &transa, (int*)&cooMatrix->numRows, (int*)&cooMatrix->numCols, &alpha, matdescra, cooMatrix->vals, cooMatrix->rows, cooMatrix->cols, &cooMatrix->nnz, x, &beta, y );
    
//    deleteCOOMatrix(cooMatrix);
}


int  benchmark_CSC_SpMV ( struct _PAVA_CSCMatrix *cscMatrix, double alpha, double *x, double beta, double *y, double *y_ref, double matrixFrobeniusNorm, char* filename, int numIterations )
{

    char transa = 'n';
    char matdescra[6] = {'G', 'x', 'x', 'C', 'x', 'x'};

    for(int i=0; i<numIterations; i++)
    mkl_dcscmv( &transa, (int*)&cscMatrix->numRows, (int*)&cscMatrix->numCols, &alpha, matdescra, cscMatrix->vals, cscMatrix->rows, cscMatrix->colOffsets, &cscMatrix->colOffsets[1], x, &beta, y );

}


int  benchmark_DIA_SpMV ( struct _PAVA_DIAMatrix *diaMatrix, double alpha, double *x, double beta, double *y, double *y_ref, double matrixFrobeniusNorm, char* filename, int numIterations )
{

    char transa = 'n';
    char matdescra[6] = {'G', 'x', 'x', 'C', 'x', 'x'};

    for(int i=0; i<numIterations; i++)
    mkl_ddiamv( &transa, (int*)&diaMatrix->numRows, (int*)&diaMatrix->numCols, &alpha, matdescra, diaMatrix->vals, (int*)&diaMatrix->idiag, diaMatrix->distance, (int*)&diaMatrix->ndiag, x, &beta, y );

}

int  benchmark_BSR_SpMV ( struct _PAVA_BSRMatrix *bsrMatrix, double alpha, double *x, double beta, double *y, double *y_ref, double matrixFrobeniusNorm, char* filename, int numIterations )
{

    char transa = 'n';
    char matdescra[6] = {'G', 'x', 'x', 'C', 'x', 'x'};

    for(int i=0; i<numIterations; i++)
    mkl_dbsrmv( &transa, (int*)&bsrMatrix->nBlockRows, (int*)&bsrMatrix->nBlockRows, (int*)&bsrMatrix->sizeBlock, &alpha, matdescra, bsrMatrix->vals, bsrMatrix->cols, bsrMatrix->rowIdx, &bsrMatrix->rowIdx[1], x, &beta, y );

}


int  benchmark_IE_SpMV ( struct _PAVA_CSRMatrix *csrMatrix, double alpha, double *x, double beta, double *y, double *y_ref, double matrixFrobeniusNorm, char* matrixName, int numIterations )
{
    int res = 0;
    double t1, t2, t3, t4;
    sparse_matrix_t csrA = NULL;
    int omp_threads, mkl_threads;

    omp_threads = omp_get_max_threads();
    mkl_set_num_threads_local(omp_threads);
    mkl_threads = mkl_get_max_threads();

    std::cout<<" Converting CSR->IE"<<endl;

    t1 = catchtime();
    struct matrix_descr descr_type_gen;
    descr_type_gen.type = SPARSE_MATRIX_TYPE_GENERAL;

    sparse_status_t status;
    status = mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, csrMatrix->numRows, csrMatrix->numCols, csrMatrix->rowOffsets, csrMatrix->rowOffsets+1, csrMatrix->cols, csrMatrix->vals);
    if(status!= SPARSE_STATUS_SUCCESS)
    {
        printf(" Error creating csr. Failed!\n");
        printPerformance(matrixName, "IE", mkl_threads, -1, -1);
        return -1;
    }
    mkl_sparse_set_mv_hint(csrA, SPARSE_OPERATION_NON_TRANSPOSE, descr_type_gen, 1);
    mkl_sparse_optimize(csrA);
    t2 = catchtime();

    std::cout<<" Executing IE"<<endl;
    t3 = catchtime();
    for(int i=0; i<numIterations; i++)
        mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrA, descr_type_gen, x, 0.0, y);
    t4 = catchtime();

    printPerformance(matrixName, "IE", mkl_threads, t2 - t1, (t4 - t3)/numIterations);

    mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrA, descr_type_gen, x, 0.0, y);

    if (mkl_sparse_destroy(csrA) != SPARSE_STATUS_SUCCESS)
    {
        printf(" Error after MKL_SPARSE_DESTROY, csrA \n"); 
        fflush(0);
        res = 1;
    }

    return 0;
}

int convertCSR2COO ( const struct _PAVA_CSRMatrix *csrMatrix, struct _PAVA_COOMatrix *cooMatrix )
{
    int info;
    int job[8];

    /************************/
    /* now convert matrix in CSR 0-based format to COO 0-based format */
    /************************/

    job[0] = 0; //  CSR  ->   COO
    job[1] = 0; // 0-based CSR
    job[2] = 0; // 0-based COO
    job[4] = csrMatrix->nnz;
    job[5] = 3; // all CSR arrays are filled

    info = 0;

    cooMatrix->numRows = csrMatrix->numRows;
    cooMatrix->numCols = csrMatrix->numCols;
    cooMatrix->nnz   = csrMatrix->nnz;

//    std::cout<<" start coo malloc"<<std::endl;

    cooMatrix->rows = ( int* )    MKL_malloc( cooMatrix->nnz * sizeof( int ),    ALIGN512 );
    cooMatrix->cols = ( int* )    MKL_malloc( cooMatrix->nnz * sizeof( int ),    ALIGN512 );
    cooMatrix->vals = ( double* ) MKL_malloc( cooMatrix->nnz * sizeof( double ), ALIGN512 );

    if ( NULL == cooMatrix->rows || NULL == cooMatrix->cols || NULL == cooMatrix->vals )
    {
        MKL_free( cooMatrix->rows );
        MKL_free( cooMatrix->cols );
        MKL_free( cooMatrix->vals );
        fprintf( stderr, "Could not allocate memory for converting matrix to COO format\n" );
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
        fprintf( stderr, " Error converting CSR -> COO: %d\n", info );
        MKL_free( cooMatrix->rows );
        MKL_free( cooMatrix->cols );
        MKL_free( cooMatrix->vals );
        return -10;
    }

/*
    int kkkk=0;
    for(kkkk=0; kkkk<256; kkkk++)
       printf(" cols[%d]= %d, vals[%d]=%f\n", kkkk, csrMatrix->cols[kkkk], kkkk, csrMatrix->vals[kkkk]);
*/

//    printf( "Operation CSR->COO completed\n" );
    return 0;
}   // convertCOO2CSR


int convertCSR2CSC ( const struct _PAVA_CSRMatrix *csrMatrix, struct _PAVA_CSCMatrix *cscMatrix )
{
    int info;
    int job[8];

    /************************/
    /* now convert matrix in CSR 0-based format to COO 0-based format */
    /************************/

    job[0] = 0; //  CSR  ->   CSC
    job[1] = 0; // 0-based CSR
    job[2] = 0; // 0-based CSC
    job[5] = 1; // all CSC arrays are filled

    info = 0;
    

    cscMatrix->numRows = csrMatrix->numRows;
    cscMatrix->numCols = csrMatrix->numCols;
    cscMatrix->nnz   = csrMatrix->nnz;


    if(cscMatrix->numRows != cscMatrix->numCols)
    {
        std::cout<<" Error: Not a square matrix "<<std::endl;
        return 1;
    }



    cscMatrix->rows = ( int* )    MKL_malloc( cscMatrix->nnz * sizeof( int ),    ALIGN512 );
    cscMatrix->colOffsets = ( int* )    MKL_malloc( (cscMatrix->numCols + 1) * sizeof( int ),    ALIGN512 );
    cscMatrix->vals = ( double* ) MKL_malloc( cscMatrix->nnz * sizeof( double ), ALIGN512 );

    if ( NULL == cscMatrix->rows || NULL == cscMatrix->colOffsets || NULL == cscMatrix->vals )
    {
        MKL_free( cscMatrix->rows );
        MKL_free( cscMatrix->colOffsets );
        MKL_free( cscMatrix->vals );
        fprintf( stderr, "Could not allocate memory for converting matrix to CSC format\n" );
        return -5;
    }

    int dim = max(cscMatrix->numRows, cscMatrix->numCols);


    mkl_dcsrcsc ( job,
//                  &csrMatrix->numRows,
                  &dim,
                  csrMatrix->vals,
                  csrMatrix->cols,
                  csrMatrix->rowOffsets,
                  cscMatrix->vals,
                  cscMatrix->rows,
                  cscMatrix->colOffsets,
                  &info );

    if ( info != 0 )
    {
        fprintf( stderr, " Error converting CSR -> CSC: %d\n", info );
        MKL_free( cscMatrix->rows );
        MKL_free( cscMatrix->colOffsets );
        MKL_free( cscMatrix->vals );
        return -10;
    }

/*
    int kkkk=0;
    for(kkkk=0; kkkk<256; kkkk++)
       printf(" cols[%d]= %d, vals[%d]=%f\n", kkkk, csrMatrix->cols[kkkk], kkkk, csrMatrix->vals[kkkk]);
*/

//    printf( "Operation CSR->CSC completed\n" );
    return 0;
}   // convertCSR2CSC


int convertCSR2DIA ( const struct _PAVA_CSRMatrix *csrMatrix, struct _PAVA_DIAMatrix *diaMatrix )
{
    int info;
    int job[8];

    /************************/
    /* now convert matrix in CSR 0-based format to COO 0-based format */
    /************************/

    job[0] = 0; //  CSR  ->   DIA
    job[1] = 0; // 0-based CSR
    job[2] = 1; // 1-based DIA
    job[5] = 10; 

    info = 0;
    

    diaMatrix->numRows = csrMatrix->numRows;
    diaMatrix->numCols = csrMatrix->numCols;
    diaMatrix->nnz   = csrMatrix->nnz;

    if(diaMatrix->numRows != diaMatrix->numCols)
    {
        std::cout<<" Error: Not a square matrix "<<std::endl;
        return 1;
    }


    diaMatrix->idiag = csrMatrix->numRows;
    diaMatrix->ndiag = 0;


    bool* usedDiag = (bool* ) MKL_malloc(sizeof(bool) * (csrMatrix->numRows*2-1), ALIGN512);
    memset(usedDiag, false, sizeof(bool)*(csrMatrix->numRows*2-1));

    for(int idxRow = 0 ; idxRow < csrMatrix->numRows; ++idxRow){
        for(int idxVal = csrMatrix->rowOffsets[idxRow] ; idxVal < csrMatrix->rowOffsets[idxRow+1] ; ++idxVal){
            const int idxCol = csrMatrix->cols[idxVal];
            const int diag = csrMatrix->numRows-idxRow+idxCol-1;
            if(!(0<=diag && diag < (csrMatrix->numRows*2-1)))
            {
                std::cout<<" Error: out of range"<<std::endl;
                return 1;
            }
//            assert(0 <= diag && diag < csrMatrix->numRows*2-1);
            if(usedDiag[diag] == false){
                usedDiag[diag] = true;
                diaMatrix->ndiag += 1;
            }
        }
    }
    MKL_free(usedDiag);                                    
//    delete[] usedDiag;

//    std::cout<<" NNN = "<<diaMatrix->ndiag <<"; iii = "<<diaMatrix->idiag<<endl;

    long long int max_size = 1 * 1024;
    max_size *= 1024;
    max_size *= 1024;
    if((diaMatrix->ndiag * diaMatrix->idiag) > max_size)
    {
        std::cout<<" Error: too large for memory allocation "<<std::endl;
        return 1;
    }

    diaMatrix->vals = ( double* )    MKL_malloc( diaMatrix->ndiag * diaMatrix->idiag * sizeof( double ),    ALIGN512 );
    diaMatrix->distance = ( int* )   MKL_malloc( (diaMatrix->ndiag) * sizeof( int ),    ALIGN512 );
//    cscMatrix->vals = ( double* ) MKL_malloc( cscMatrix->nnz * sizeof( double ), ALIGN );

    if ( NULL == diaMatrix->vals || NULL == diaMatrix->distance)
    {
        MKL_free( diaMatrix->vals );
        MKL_free( diaMatrix->distance );
        fprintf( stderr, "Could not allocate memory for converting matrix to DIA format\n" );
        return -5;
    }

    int dim = max(diaMatrix->numRows, diaMatrix->numCols);
//    int dim = 2;

//    std::cout<<" before dcsrdia"<<endl;


    mkl_dcsrdia ( job,
//                  &csrMatrix->numRows,
                  &dim,
                  csrMatrix->vals,
                  csrMatrix->cols,
                  csrMatrix->rowOffsets,
                  diaMatrix->vals,
                  &diaMatrix->idiag,
                  diaMatrix->distance,
                  &diaMatrix->ndiag,
                  NULL,
                  NULL,
                  NULL,
                  &info );

    if ( info != 0 )
    {
        fprintf( stderr, " Error converting: %d\n", info );
        MKL_free( diaMatrix->vals );
        MKL_free( diaMatrix->distance );
        return -10;
    }

/*
    int kkkk=0;
    for(kkkk=0; kkkk<256; kkkk++)
       printf(" cols[%d]= %d, vals[%d]=%f\n", kkkk, csrMatrix->cols[kkkk], kkkk, csrMatrix->vals[kkkk]);
*/

//    printf( "Operation CSR->DIA completed\n" );
    return 0;
}   // convertCOO2DIA



int convertCSR2BSR ( const struct _PAVA_CSRMatrix *csrMatrix, struct _PAVA_BSRMatrix *bsrMatrix, int blockSize )
{
    int info;
    int job[8];

    /************************/
    /* now convert matrix in CSR 0-based format to COO 0-based format */
    /************************/

    job[0] = 0; //  CSR  ->   BSR
    job[1] = 0; // 0-based CSR
    job[2] = 0; // 0-based BSR
    job[5] = 1; // all BSR arrays are filled

    info = 0;
    

    bsrMatrix->numRows = csrMatrix->numRows;
    bsrMatrix->numCols = csrMatrix->numCols;
    bsrMatrix->nnz   = csrMatrix->nnz;


    if(bsrMatrix->numRows != bsrMatrix->numCols)
    {
        std::cout<<" Error: Not a square matrix "<<std::endl;
        return 1;
    }



    bsrMatrix->nBlocks = 0;

    const MKL_INT maxBlockPerRow = (bsrMatrix->numRows + blockSize - 1)/blockSize;

    long long int max_size = 2 * 1024;
    max_size *= 1024;
    max_size *= 1024;
    if((maxBlockPerRow * maxBlockPerRow) > max_size)
    {
        std::cout<<" Error: too large for memory allocation "<<std::endl;
//        std::cout<<" block*block "<< maxBlockPerRow * maxBlockPerRow<<"  "<<max_size<<std::endl;
        return 1;
    }

//    unsigned* usedBlocks = new unsigned[maxBlockPerRow * maxBlockPerRow];
//    bool* usedBlocks = new bool[maxBlockPerRow * maxBlockPerRow];
    bool* usedBlocks = (bool*)MKL_malloc(sizeof(bool) * maxBlockPerRow * maxBlockPerRow,ALIGN512);

    memset(usedBlocks, false, sizeof(bool)*maxBlockPerRow*maxBlockPerRow);

    int _max = 0;
    for(int idxRow = 0 ; idxRow < csrMatrix->numRows ; ++idxRow){
//        if(idxRow%blockSize == 0){
//            memset(usedBlocks, 0, sizeof(unsigned)*maxBlockPerRow);
            for(int idxVal = csrMatrix->rowOffsets[idxRow] ; idxVal < csrMatrix->rowOffsets[idxRow+1] ; ++idxVal){
                const int idxCol = csrMatrix->cols[idxVal];
//                if(usedBlocks[idxCol/blockSize] == 0){
                if(!usedBlocks[idxRow/blockSize * maxBlockPerRow + idxCol/blockSize])
                    bsrMatrix->nBlocks += 1;
                usedBlocks[idxRow/blockSize * maxBlockPerRow + idxCol/blockSize] = true;
//                if(_max < usedBlocks[idxRow/blockSize * maxBlockPerRow + idxCol/blockSize])
//                     _max = usedBlocks[idxRow/blockSize * maxBlockPerRow + idxCol/blockSize];
//                std::cout<<" row = "<< idxRow<<"; col = "<< idxCol<<"; num="<<usedBlocks[idxRow/blockSize*maxBlockPerRow + idxCol/blockSize]<<endl;
//                }
//            }
        }
                             
//        delete[] usedBlocks;
    }
    MKL_free(usedBlocks);
/*
    for (int i=0; i<8; i++)
        for (int j=0; j<8; j++)
        {
            std::cout<<" num["<<i<<"]["<<j<<"] = "<< usedBlocks[i*maxBlockPerRow + j]<<endl;
        
        }
*/
    bsrMatrix->nBlockRows = (bsrMatrix->numRows+blockSize-1)/blockSize;
    bsrMatrix->sizeBlock = blockSize;
    bsrMatrix->leadingBlock = blockSize * blockSize;
//    bsrMatrix->leadingBlock = _max*8;

    bsrMatrix->vals = ( double* ) MKL_malloc( bsrMatrix->nBlocks * bsrMatrix->leadingBlock * sizeof( double ), ALIGN512 );
//    bsrMatrix->vals = ( double* )mmap( 0, bsrMatrix->nBlocks * bsrMatrix->leadingBlock * sizeof( double ), PROT_READ|PROT_WRITE,MAP_ANONYMOUS|MAP_PRIVATE|MAP_HUGETLB, -1,0);
//    bsrMatrix->rowIdx = ( int* )    MKL_malloc((bsrMatrix->nBlockRows+1) * sizeof( int ), ALIGN512 );
    bsrMatrix->rowIdx = ( int* )    MKL_malloc((bsrMatrix->numRows+1) * sizeof( int ), ALIGN512 );
//    bsrMatrix->rowIdx = ( int* )    _mm_malloc((bsrMatrix->nBlockRows+1) * sizeof( int ), ALIGN512 );
    bsrMatrix->cols = ( int* )    MKL_malloc(bsrMatrix->nBlocks * sizeof( int ), ALIGN512 );
//    bsrMatrix->cols = ( int* )    _mm_malloc(bsrMatrix->nBlocks * sizeof( int ), ALIGN );

    bsrMatrix->y_bsr = (double*)MKL_malloc(sizeof(double) * bsrMatrix->nBlockRows * bsrMatrix->sizeBlock,ALIGN512);

//    std::cout<<"nblocks ="<<bsrMatrix->nBlocks<<" leadingBlock="<<bsrMatrix->leadingBlock<<" size = "<< bsrMatrix->nBlocks * bsrMatrix->leadingBlock<<endl;
//    std::cout<< ! bsrMatrix->vals <<"  "<< !bsrMatrix->rowIdx<<" "<<!bsrMatrix->cols<<endl;

    if ( NULL == bsrMatrix->cols || NULL == bsrMatrix->rowIdx || NULL == bsrMatrix->vals || NULL == bsrMatrix->y_bsr )
    {
        MKL_free( bsrMatrix->cols );
        MKL_free( bsrMatrix->rowIdx );
        MKL_free( bsrMatrix->vals );
        MKL_free( bsrMatrix->y_bsr);
        fprintf( stderr, "Could not allocate memory for converting matrix to BSR format\n" );
        return -5;
    }

    int dim = max(bsrMatrix->numRows, bsrMatrix->numCols);

    mkl_dcsrbsr ( job,
//                  &csrMatrix->numRows,
                  &dim,
                  &bsrMatrix->sizeBlock,
                  &bsrMatrix->leadingBlock,
                  csrMatrix->vals,
                  csrMatrix->cols,
                  csrMatrix->rowOffsets,
                  bsrMatrix->vals,
                  bsrMatrix->cols,
                  bsrMatrix->rowIdx,
                  &info );

    if ( info != 0 )
    {
        fprintf( stderr, " Error converting CSR -> BSR: %d\n", info );
        MKL_free( bsrMatrix->cols );
        MKL_free( bsrMatrix->rowIdx );
        MKL_free( bsrMatrix->vals );
        MKL_free( bsrMatrix->y_bsr);
        return -10;
    }

/*
    int kkkk=0;
    for(kkkk=0; kkkk<256; kkkk++)
       printf(" cols[%d]= %d, vals[%d]=%f\n", kkkk, csrMatrix->cols[kkkk], kkkk, csrMatrix->vals[kkkk]);
*/

//    printf( "Operation CSR->BSR completed\n" );
    return 0;
}   // convertCSR2BSR




