#ifndef PAVA_FORMATS_H
#define PAVA_FORMATS_H


/////////////////////////////////
//
//      Format Conversion
//
////////////////////////////////

int convertCSR2COO ( const struct _PAVA_CSRMatrix *csrMatrix, struct _PAVA_COOMatrix *cooMatrix );

int convertCSR2CSC ( const struct _PAVA_CSRMatrix *csrMatrix, struct _PAVA_CSCMatrix *cscMatrix );

int convertCSR2DIA ( const struct _PAVA_CSRMatrix *csrMatrix, struct _PAVA_DIAMatrix *diaMatrix );

int convertCSR2BSR ( const struct _PAVA_CSRMatrix *csrMatrix, struct _PAVA_BSRMatrix *bsrMatrix, int blockSize );





/////////////////////////////////
//
//      SpMV Execution
//
////////////////////////////////


int  benchmark_CSR_SpMV ( struct _PAVA_CSRMatrix *csrMatrix, double alpha, double *x, double beta, double *y, double *y_ref, double matrixFrobeniusNorm, char* filename, int numIterations );

int  benchmark_COO_SpMV ( struct _PAVA_COOMatrix *cooMatrix, double alpha, double *x, double beta, double *y, double *y_ref, double matrixFrobeniusNorm, char* filename, int numIterations );

int  benchmark_CSC_SpMV ( struct _PAVA_CSCMatrix *cscMatrix, double alpha, double *x, double beta, double *y, double *y_ref, double matrixFrobeniusNorm, char* filename, int numIterations );

int  benchmark_DIA_SpMV ( struct _PAVA_DIAMatrix *diaMatrix, double alpha, double *x, double beta, double *y, double *y_ref, double matrixFrobeniusNorm, char* filename, int numIterations );

int  benchmark_BSR_SpMV ( struct _PAVA_BSRMatrix *bsrMatrix, double alpha, double *x, double beta, double *y, double *y_ref, double matrixFrobeniusNorm, char* filename, int numIterations );

int  benchmark_IE_SpMV ( struct _PAVA_CSRMatrix *csrMatrix, double alpha, double *x, double beta, double *y, double *y_ref, double matrixFrobeniusNorm, char* filename, int numIterations );

#endif
