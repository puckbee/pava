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
#include <math.h>
#include <unistd.h>
#include <sys/mman.h>

#include <mkl.h>

#ifdef __KNC__
    #include "./spmv_interface.h"
#endif



#include "../utils/mmio.h"
#include "../utils/basicMatrix.h"


int  benchmark_ESB_SpMV ( const struct _PAVA_CSRMatrix *csrMatrix, double alpha, double *x, double beta, double *y, double *y_ref, double mflop, double matrixFrobeniusNorm, 
                                 sparseSchedule_t schedule, char* filename, int numIterations );
