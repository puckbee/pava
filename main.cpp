#include <stdio.h>
#include <stdlib.h>
//#include <string.h>
#include <omp.h>
#include <unistd.h>
#include <sys/mman.h>
#include <dirent.h>
#include <sys/stat.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <list>
#include <vector>
#include <set>
#include <cmath>

#include <immintrin.h>
#include <mkl.h>



#include "csr5/csr5.h"
#include "cvr/cvr.h"
#include "vhcc/vhcc.h"
#include "esb/esb.h"

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

#define EPS 1.0e-15


#define LARGE_ITERS 1000
#define SMALL_ITERS 10

int conduct_benchmark(char* fileName, int numThreads, _PAVA_COOMatrix* fileCOOMatrix);


int benchFile(char* fullpath, int thread_idx)
{

    int minSize = 4096;
    int maxSize = 1024+128;   // 1 billion data
    maxSize *= 1024;
    maxSize *= 1024;

   
    struct _PAVA_COOMatrix* cooMatrix = (struct _PAVA_COOMatrix*) malloc(sizeof(struct _PAVA_COOMatrix));

    if ( 0 != readCOOMatrix( fullpath, cooMatrix ) )
    {
        fprintf(stderr, "Reading COO matrix in matrix market format failed\n" );
        return -2;
    }

    std::cout<<"Basic informations"<<endl;
    std::cout<<"        numRows = "<<cooMatrix->numRows<<endl;
    std::cout<<"        numCols = "<<cooMatrix->numCols<<endl;
    std::cout<<"            nnz = "<<cooMatrix->nnz<<endl;
    if(cooMatrix->nnz < minSize)
    {
        std::cout<<" FileWarning! "<<fullpath <<" is too small (4k)"<<endl;
        std::cout<<endl<<"Congratulations! This File comes to an Normal End! Flag[NormalEnding]"<<endl<<endl;
        return -1;
    }
    else if (cooMatrix->nnz > maxSize)
    {
        std::cout<<" FileWarning! "<<fullpath <<" is too large (1g))"<<endl;
        std::cout<<endl<<"Congratulations! This File comes to an Normal End! Flag[NormalEnding]"<<endl<<endl;
        return -1;
    }
    else if (cooMatrix->numRows < 2048 || cooMatrix->numCols < 2048)
    {
        std::cout<<" FileWarning! "<<fullpath <<" has too few rows or cols"<<endl;
        std::cout<<endl<<"Congratulations! This File comes to an Normal End! Flag[NormalEnding]"<<endl<<endl;
        return -1;
    }
        
    std::cout<<endl<<"Print out result in [groupName matrixName][format][thread][convertTime][executionTime]"<<endl;

    if(thread_idx ==0)
        for(int iterThreads=68; iterThreads<=272; iterThreads+=68)
        {
            conduct_benchmark(fullpath, iterThreads, cooMatrix);
        }
    else
        conduct_benchmark(fullpath, thread_idx, cooMatrix);

    std::cout<<endl<<"Congratulations! This File comes to an Normal End! Flag[NormalEnding]"<<endl<<endl;
    return 0;

}





void benchDir(char *path)  
{  
    DIR              *pDir ;  
    struct dirent    *ent  ;  
    int               i=0  ;  
    char              childpath[512];  
    char              fullpath[512];
  
    pDir=opendir(path);  
    memset(childpath,0,sizeof(childpath)); 

  
    while((ent=readdir(pDir))!=NULL)  
    {  
  
        if(ent->d_type & DT_DIR)
        {  
            if(strcmp(ent->d_name,".")==0 || strcmp(ent->d_name,"..")==0)  
                continue;  
  
            sprintf(childpath,"%s/%s",path,ent->d_name);  
//            printf("path:%s\n",childpath);  
 
            cout<<" "<<ent->d_name;

            benchDir(childpath);  
        }  
        else
        {
            strcpy(fullpath, path);
            strcat(fullpath, "/");
            strcat(fullpath, ent->d_name);

            benchFile(fullpath, 0);

//            cout<<" -- "<<fullpath<<endl;
//            cout<<ent->d_name<<endl;
        }
    }  
  
}  


int main(int argc, char** argv)
{

//    conduct_benchmark(argv[1], 136);
//    return 0;


    cout<<" =================================       now we begin PAVA     ==================================="<<endl;

    if ( argc < 2 )
    {
        fprintf( stderr, "Insufficient number of input parameters:\n");
        fprintf( stderr, "File name with sparse matrix for benchmarking is missing\n" );
        fprintf( stderr, "Usage: %s [martix market filename]\n", argv[0] );
        exit(1);
    }

    char* input_path = argv[1];


    struct stat s_buf;
    stat(input_path, &s_buf);

    if(S_ISDIR(s_buf.st_mode))
    {
//        printf(" %s is a dir\n", input_path);
        benchDir(input_path);        
    }

    if(S_ISREG(s_buf.st_mode))
    {
//        printf(" %s is a regular file \n", input_path);
        benchFile(input_path, atoi(argv[2]));
    }


    return 0;

}

int conduct_benchmark(char* fileName, int numThreads, _PAVA_COOMatrix* fileCOOMatrix)
{
//    char* fileName = argv[1];

    omp_set_num_threads(numThreads);
    int fileLen = strlen(fileName);
//    char* tmpMatrixName = (char*)malloc(sizeof(char) * (fileLen+1));
    char* tmpMatrixName = (char*)malloc(sizeof(char) * 512);
    strcpy(tmpMatrixName, fileName);

    char ch = '/';
    char ch2 = '.';

    if( strrchr(tmpMatrixName,ch2) )
        strrchr(tmpMatrixName,ch2)[0] = '\0';

    if( strrchr(tmpMatrixName,ch) )
        strrchr(tmpMatrixName,ch)[0] = ' ';
        
    char* matrixName;
    if( strrchr(tmpMatrixName,ch) )
        matrixName = strrchr(tmpMatrixName,ch) + 1;
    else
        matrixName = tmpMatrixName;

    
    double *y, *y_ref;
    double *x;

/*
    struct _PAVA_COOMatrix* cooMatrix = (struct _PAVA_COOMatrix*) malloc(sizeof(struct _PAVA_COOMatrix));
    if ( 0 != readCOOMatrix( fileName, cooMatrix ) )
    {
        fprintf(stderr, "Reading COO matrix in matrix market format failed\n" );
        return -2;
    }
*/


    // Align allocated memory to boost performance 
#ifdef MMAP
    x     = (double*)mmap(0, fileCOOMatrix->numCols* sizeof(double), PROT_READ|PROT_WRITE,MAP_ANONYMOUS|MAP_PRIVATE|MAP_HUGETLB,-1,0);
#else
    x     = ( double* ) MKL_malloc ( fileCOOMatrix->numCols * sizeof( double ), ALIGN );
#endif
    y     = ( double* ) MKL_malloc ( fileCOOMatrix->numRows * sizeof( double ), ALIGN );
    y_ref = ( double* ) MKL_malloc ( fileCOOMatrix->numRows * sizeof( double ), ALIGN );

    if ( NULL == x || NULL == y || NULL == y_ref )
    {
        fprintf( stderr, "Could not allocate memory for vectors!\n" );
        MKL_free( x );
        MKL_free( y );
        MKL_free( y_ref );

        deleteCOOMatrix( fileCOOMatrix );
        return -1;
    }




    // res is the number of items with error
    int res = 0;
    int numIterations;
    double t1, t2, t3, t4;
    int omp_threads;                              // get the numThreads from env variable of openMP
    int mkl_threads;                              // get the numThreads from env variable of MKL

    double  alpha = 1;
    double beta = 1;

    double normX, normY, residual;
    double estimatedAccuracy;
    double matrixFrobeniusNorm = calcFrobeniusNorm ( fileCOOMatrix->nnz, fileCOOMatrix->vals );
    normX = calcFrobeniusNorm ( fileCOOMatrix->numCols, x );
    normY = calcFrobeniusNorm ( fileCOOMatrix->numRows, y );
    // estimate accuracy of SpMV oparation: y = alpha * A * x + beta * y
    // || y1 - y2 || < eps * ( |alpha| * ||A|| * ||x|| + |beta| * ||y||)
    estimatedAccuracy = fabs( alpha ) * matrixFrobeniusNorm * normX + fabs( beta ) * normY;
    estimatedAccuracy *= EPS;


///////////////////////
//      CSR      
///////////////////////

    std::cout<<"**************"<<endl;
    std::cout<<"      CSR     "<<endl;
    std::cout<<"**************"<<endl;

    struct _PAVA_CSRMatrix* csrMatrix = (struct _PAVA_CSRMatrix*) malloc(sizeof(struct _PAVA_CSRMatrix));
    numIterations = LARGE_ITERS;

    omp_threads = omp_get_max_threads();
    mkl_set_num_threads_local(omp_threads);
    mkl_threads = mkl_get_max_threads();

    std::cout<<" Converting COO->CSR"<<endl;
    t1 = microtime();
    res=convertCOO2CSR(fileCOOMatrix, csrMatrix);
    t2 = microtime();
    if(res!=0)
    {
        std::cout<<" CSR Converting Failed"<<std::endl;

        printPerformance(matrixName, "CSR", mkl_threads, -1, -1);
        free(csrMatrix);
    }
    else
    {
    //    std::cout<<"Converting(COO->CSR)   Time of CSR    is "<< t2 - t1 <<endl;

        // first time to init x, y, y_ref
        initVectors( csrMatrix->numRows, csrMatrix->numCols, x, y, y_ref );
        referenceSpMV(csrMatrix, x, y_ref);

        std::cout<<" Executing CSR"<<endl;
        benchmark_CSR_SpMV( csrMatrix, alpha, x, beta, y, y_ref, 0, matrixName, 1);
        res = checkResults(csrMatrix->numRows, y, y_ref);

        t3 = microtime();
        benchmark_CSR_SpMV( csrMatrix, alpha, x, beta, y, y_ref, 0, matrixName, numIterations);
        t4 = microtime();
    //    std::cout<<" The SpMV Execution Time of CSR    is "<< (t4 - t3)/nuIterations<<endl;
       
        printPerformance(matrixName, "CSR", mkl_threads, t2 - t1, (t4 - t3)/numIterations);

        // we leave fileCOOMarix for the next iteration
        deleteCOOMatrix( fileCOOMatrix );    
    }






///////////////////////
//      COO      
///////////////////////

    std::cout<<"**************"<<endl;
    std::cout<<"      COO     "<<endl;
    std::cout<<"**************"<<endl;
    struct _PAVA_COOMatrix* cooMatrix = (struct _PAVA_COOMatrix*) malloc(sizeof(struct _PAVA_COOMatrix));
    numIterations = SMALL_ITERS;

    omp_threads = omp_get_max_threads();
    mkl_set_num_threads_local(omp_threads);
    mkl_threads = mkl_get_max_threads();

    std::cout<<" Converting CSR->COO"<<endl;
    t1 = microtime();
    res = convertCSR2COO(csrMatrix, cooMatrix);
    t2 = microtime();
    if(res!=0)
    {
        std::cout<<" COO Converting Failed"<<std::endl;

        printPerformance(matrixName, "COO", mkl_threads, -1, -1);
        free(cooMatrix);
    }
    else
    {
        initVectors( cooMatrix->numRows, cooMatrix->numCols, NULL, y, NULL );

        std::cout<<" Executing COO"<<endl;
        benchmark_COO_SpMV( cooMatrix, alpha, x, beta, y, y_ref, 0, matrixName, 1);
        res = checkResults(cooMatrix->numRows, y, y_ref);

        t3 = microtime();
        benchmark_COO_SpMV( cooMatrix, alpha, x, beta, y, y_ref, 0, matrixName, 10);
        t4 = microtime();

        printPerformance(matrixName, "COO", mkl_threads, t2-t1, (t4 - t3)/numIterations);

        deleteCOOMatrix(cooMatrix);
    }
///////////////////////
//      CSC
///////////////////////
    std::cout<<"**************"<<endl;
    std::cout<<"      CSC     "<<endl;
    std::cout<<"**************"<<endl;
    struct _PAVA_CSCMatrix* cscMatrix = (struct _PAVA_CSCMatrix* ) malloc (sizeof (struct _PAVA_CSCMatrix));
    numIterations = SMALL_ITERS;

    omp_threads = omp_get_max_threads();
    mkl_set_num_threads_local(omp_threads);
    mkl_threads = mkl_get_max_threads();

    std::cout<<" Converting CSR->CSC"<<endl;
    t1 = microtime();
    res = convertCSR2CSC(csrMatrix, cscMatrix);
    t2 = microtime();
    if(res!=0)
    {
        std::cout<<" CSC Converting Failed"<<std::endl;

        printPerformance(matrixName, "CSC", mkl_threads, -1, -1);
        free(cscMatrix);
    }
    else
    {

        initVectors( cscMatrix->numRows, cscMatrix->numCols, NULL, y, NULL );
//        referenceSpMV(csrMatrix, x, y_ref);

        std::cout<<" Executing CSC"<<endl;
        benchmark_CSC_SpMV( cscMatrix, alpha, x, beta, y, y_ref, 0, matrixName, 1);
        res = checkResults(cscMatrix->numRows, y, y_ref);
        
        initVectors( cscMatrix->numRows, cscMatrix->numCols, x, y, y_ref );
        referenceSpMV(csrMatrix, x, y_ref);

        t3 = microtime();
        benchmark_CSC_SpMV( cscMatrix, alpha, x, beta, y, y_ref, 0, matrixName, 10);
        t4 = microtime();

        printPerformance(matrixName, "CSC", mkl_threads, t2 - t1, (t4 - t3)/numIterations);
        deleteCSCMatrix(cscMatrix);
    }


///////////////////////
//      DIA
///////////////////////

    std::cout<<"**************"<<endl;
    std::cout<<"      DIA     "<<endl;
    std::cout<<"**************"<<endl;
    struct _PAVA_DIAMatrix* diaMatrix = (struct _PAVA_DIAMatrix* ) malloc (sizeof (struct _PAVA_DIAMatrix));
    numIterations = SMALL_ITERS;

    omp_threads = omp_get_max_threads();
    mkl_set_num_threads_local(omp_threads);
    mkl_threads = mkl_get_max_threads();

    std::cout<<" Converting CSR->DIA"<<endl;
    t1 = microtime();
    res = convertCSR2DIA(csrMatrix, diaMatrix);
    t2 = microtime();
    if(res!=0)
    {
        std::cout<<" DIA Converting Failed"<<std::endl;

        printPerformance(matrixName, "DIA", mkl_threads, -1, -1);
        free(diaMatrix);
    }
    else
    {
        initVectors( diaMatrix->numRows, diaMatrix->numCols, NULL, y, NULL );
//        referenceSpMV(csrMatrix, x, y_ref);

        std::cout<<" Executing DIA"<<endl;
        benchmark_DIA_SpMV( diaMatrix, alpha, x, beta, y, y_ref, 0, matrixName, 1);
        res = checkResults(csrMatrix->numRows, y, y_ref);

        t3 = microtime();
        benchmark_DIA_SpMV( diaMatrix, alpha, x, beta, y, y_ref, 0, matrixName, 10);
        t4 = microtime();

        printPerformance(matrixName, "DIA", mkl_threads, t2 - t1, (t4 - t3)/numIterations);
        deleteDIAMatrix(diaMatrix);
    }
///////////////////////
//      IE
///////////////////////
    std::cout<<"**************"<<endl;
    std::cout<<"      IE      "<<endl;
    std::cout<<"**************"<<endl;
    numIterations = LARGE_ITERS;

    initVectors( csrMatrix->numRows, csrMatrix->numCols, NULL, y, NULL );

    benchmark_IE_SpMV( csrMatrix, alpha, x, beta, y, y_ref, 0, matrixName, 10);

//    printPerformance(matrixName, "IE", mkl_threads, -1, -1);
    res = checkResults(csrMatrix->numRows, y, y_ref);

///////////////////////
//      BSR
///////////////////////

    std::cout<<"**************"<<endl;
    std::cout<<"      BSR      "<<endl;
    std::cout<<"**************"<<endl;
    struct _PAVA_BSRMatrix* bsrMatrix = (struct _PAVA_BSRMatrix* ) malloc (sizeof (struct _PAVA_BSRMatrix));
    numIterations = SMALL_ITERS;
    
    omp_threads = omp_get_max_threads();
    mkl_set_num_threads_local(omp_threads);
    mkl_threads = mkl_get_max_threads();

    std::cout<<" Converting CSR->BSR"<<endl;
    t1 = microtime();
    res = convertCSR2BSR(csrMatrix, bsrMatrix, 4);
    t2 = microtime();

    if(res!=0)
    {
        std::cout<<" BSR Converting Failed"<<std::endl;

        printPerformance(matrixName, "BSR", mkl_threads, -1, -1);
        free(bsrMatrix);
    }
    else
    {
        initVectors( bsrMatrix->numRows, bsrMatrix->numCols, NULL, y, NULL );
//        referenceSpMV(csrMatrix, x, y_ref);
        
        std::cout<<" Executing BSR"<<endl;

        benchmark_BSR_SpMV( bsrMatrix, alpha, x, beta, bsrMatrix->y_bsr, y_ref, 0, matrixName, 1);
        res = checkResults(csrMatrix->numRows, bsrMatrix->y_bsr, y_ref);

        t3 = microtime();
        benchmark_BSR_SpMV( bsrMatrix, alpha, x, beta, bsrMatrix->y_bsr, y_ref, 0, matrixName, 10);
        t4 = microtime();

        printPerformance(matrixName, "BSR", mkl_threads, t2 - t1, (t4 - t3)/numIterations);

        deleteBSRMatrix(bsrMatrix);

    }



///////////////////////
//      ESB
///////////////////////
    
    std::cout<<"**************"<<endl;
    std::cout<<"      ESB      "<<endl;
    std::cout<<"**************"<<endl;
    numIterations = LARGE_ITERS;

//    initVectors( csrMatrix->numRows, csrMatrix->numCols, NULL, y, NULL );

//    benchmark_ESB_SpMV(csrMatrix, alpha, x, beta, y, NULL, 0.0, 0.0, INTEL_SPARSE_SCHEDULE_DYNAMIC, matrixName, numIterations);
//    res = checkResults(csrMatrix->numRows, y, y_ref);

//    benchmark_ESB_SpMV(csrMatrix, alpha, x, beta, y, NULL, 0.0, 0.0, INTEL_SPARSE_SCHEDULE_STATIC, matrixName, numIterations);
//    res = checkResults(csrMatrix->numRows, y, y_ref);

    printPerformance(matrixName, "ESB", mkl_threads, -1, -1);
///////////////////////////////////////
///////////////////////
//      CVR
///////////////////////

    std::cout<<"**************"<<endl;
    std::cout<<"      CVR      "<<endl;
    std::cout<<"**************"<<endl;
    numIterations = LARGE_ITERS;
    if(csrMatrix->numRows > 4096 && csrMatrix->numCols > 4096)
    {

        initVectors( csrMatrix->numRows, csrMatrix->numCols, NULL, y, NULL );

        conduct_cvr(csrMatrix->numRows, csrMatrix->numCols, csrMatrix->nnz, csrMatrix->rowOffsets, csrMatrix->cols, csrMatrix->vals, x, y, alpha, matrixName, numIterations);

        res = checkResults(csrMatrix->numRows, y, y_ref);
    }
    else
    {
        omp_threads = omp_get_max_threads();
        printPerformance(matrixName, "CVR", omp_threads, -1,-1);
    }

////////////////////////////////////////
///////////////////////
//      CSR5
///////////////////////

    std::cout<<"**************"<<endl;
    std::cout<<"     CSR5     "<<endl;
    std::cout<<"**************"<<endl;
    numIterations = LARGE_ITERS;

    initVectors( csrMatrix->numRows, csrMatrix->numCols, NULL, y, NULL );

    conduct_csr5(csrMatrix->numRows, csrMatrix->numCols, csrMatrix->nnz, csrMatrix->rowOffsets, csrMatrix->cols, csrMatrix->vals, x, y, alpha, matrixName, numIterations);

    res = checkResults(csrMatrix->numRows, y, y_ref);

////////////////////////////////////////
///////////////////////
//      VHCC
///////////////////////
    std::cout<<"**************"<<endl;
    std::cout<<"     VHCC     "<<endl;
    std::cout<<"**************"<<endl;
    numIterations = LARGE_ITERS;

//    initVectors( fileCOOMatrix->numRows, fileCOOMatrix->numCols, NULL, y, NULL );

//    conduct_vhcc(fileCOOMatrix->numRows, fileCOOMatrix->numCols, fileCOOMatrix->nnz, fileCOOMatrix->rows, fileCOOMatrix->cols, fileCOOMatrix->vals, x, y, alpha, matrixName, numIterations);
//
    printPerformance(matrixName, "VHCC", omp_threads, -1, -1);

//    res = checkResults(fileCOOMatrix->numRows, y, y_ref);
    
////////////////////////////////////////
    MKL_free( x );
    MKL_free( y );
    MKL_free( y_ref );
    deleteCSRMatrix(csrMatrix);
    free(tmpMatrixName);

    std::cout<<" x.x.x.x.x.x.x.x.x.x.x.x"<<endl<<endl;

    return 0;
}




