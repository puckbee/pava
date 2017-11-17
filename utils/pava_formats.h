#ifndef PAVA_FORMATS_H
#define PAVA_FORMATS_H

int  benchmark_MKL_SpMV ( struct _PAVA_CSRMatrix *csrMatrix, double alpha, double *x, double beta, double *y, double *y_ref, double matrixFrobeniusNorm, char* filename, int numIterations );



#endif
