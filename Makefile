all:
	nvcc -I /opt/apps/cuda_SDK/4.1/CUDALibraries/common/inc/ -I /opt/apps/cuda_SDK/4.1/shared/inc/ -I /work/01940/prankur/SC/Install/prankur/cudpp_src_2.0/include/ -L /work/01940/prankur/SC/Install/prankur/installation/lib/ prob.cu -lcudpp -I/opt/apps/cuda_SDK/4.1/C/common/inc/ -o prob
