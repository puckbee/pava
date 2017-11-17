
##===================       Setting MKL     ========================
ifndef MKLROOT
  $(error Path to Intel MKL is not defined in MKLROOT)
endif

MKL_LIB=$(MKLROOT)/lib/intel64_lin
MKL_LINK= -Wl,--start-group $(MKL_LIB)/libmkl_intel_lp64.a $(MKL_LIB)/libmkl_intel_thread.a $(MKL_LIB)/libmkl_core.a -Wl,--end-group
##==================================================================


OPT_FLAG= -O3 -ansi-alias -xMIC-AVX512 -qopenmp
EXE= pava


all: pava

pava: mmio.o basicMatrix.o pava_formats.o
	icpc $(OPT_FLAG) mmio.o basicMatrix.o pava_formats.o main.cpp $(MKL_LINK) -o pava

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
