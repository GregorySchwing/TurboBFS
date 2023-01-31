#!/bin/bash

#sycl standard build
#clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda *.cpp *.c -o sycl-app
#For NVCC Compiler

#CUDA_DIR=/usr/local/cuda
#NVCCFLAGS = -O3 -arch=sm_61 -I$(CUDA_DIR)/include  -I/$(CUDA_DIR)/samples/common/inc -lineinfo  
#NVCCLIBS = -L$(CUDA_DIR)/lib64 -lcufft -lcudart -lcudadevrt  -lcusparse 
#CFLAGS = $(NVCCFLAGS) -Xcompiler -std=gnu99  --use_fast_math --compiler-options
#CXXFLAGS = -O3 --use_fast_math
#LDLIBS = $(NVCCLIBS) -lgomp

# O3
#clang++ -fsycl -O3 -fsycl-targets=nvptx64-nvidia-cuda *.cpp *.c -o sycl-app

# Fast math, not targetable to cuda
#clang++ -fsycl -ffast-math *.cpp *.c -o sycl-app

# O3 with early opt
clang++ -fsycl -fast -fsycl-targets=nvptx64-nvidia-cuda *.cpp *.c -o sycl-app

