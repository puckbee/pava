
##===================       Setting MKL     ========================
ifndef MKLROOT
  $(error Path to Intel MKL is not defined in MKLROOT)
endif

MKL_LIB=$(MKLROOT)/lib/intel64_lin
MKL_LINK= -Wl,--start-group $(MKL_LIB)/libmkl_intel_lp64.a $(MKL_LIB)/libmkl_intel_thread.a $(MKL_LIB)/libmkl_core.a -Wl,--end-group
##==================================================================

esblib=lib/libesb.a

#OPT_FLAG= -ansi-alias -O3 -xMIC-AVX512 -qopenmp -D__MIC__ -std=c++0x
OPT_FLAG= -ansi-alias -xMIC-AVX512 -qopenmp -D__MIC__ -std=c++0x -D__KNC__ -mkl -O3
#OPT_FLAG= -ansi-alias -xMIC-AVX512 -D__MIC__ -std=c++0x -D__KNC__ -D_SpMV_ESB_ -mkl -g
EXE= pava


all: pava

pava: mmio.o basicMatrix.o pava_formats.o esb.o
	icpc $(OPT_FLAG) mmio.o basicMatrix.o pava_formats.o esb.o main.cpp -o pava $(esblib)

esb.o:
	icpc  $(OPT_FLAG) -c -qopenmp esb/esb.c -I$(MKLROOT)/include -D_SpMV_ESB_

mmio.o: 
	icpc $(OPT_FLAG) -c utils/mmio.c 

basicMatrix.o: 
	icpc $(OPT_FLAG) -c utils/basicMatrix.cpp

pava_formats.o:
	icpc $(OPT_FLAG) -c utils/pava_formats.cpp

#debug:
#	icpc -O0 -ansi-alias -xMIC-AVX512 -qopenmp spmv.cpp -o spmv.g -g
clean:
	rm *.o $(EXE)
