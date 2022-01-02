# 并行结算结课论文

高斯消元以及图像卷积的CUDA实现 

## 文件结构

### Gauss

h_common.h -|-makeEquationData.cpp
            |
            |-h_cpu.h-|-gauss_cpu.cpp
            |                |-gauss_omp.cpp
            |
            |-h_gpu.h-|-gauss1.cu
                      |-gauss2.1.cu
                      |-gauss2.2.cu
                      |-gauss2.3.cu
                      |-gauss3.cu
            

### Convolution

mask.h-|-convol.cpp
       |-convol_omp.h
	   |-mycuda.h-|-main.cpp + |-convol_plain.cu
	   			  |			                  |-convol_shared.cu
	   			  |
				     |-main_2gpu.cpp + convol_with_omp.cu



## 运行

要运行请使用Gauss/a.sh或Convolution/z.sh
