# 并行结算结课论文

此为OUC2021年秋并行计算课程，我的结课论文代码，包括高斯消去法以及图像卷积的CUDA实现 。

## 文件结构

### Gauss

* h_common.h
  * makeEquationData.cpp
  * h_cpu.h
    * gauss_cpu.cpp
    * gauss_omp.cpp
  * h_gpu.h
    * gauss1.cu
    * gauss2.1.cu
    * gauss2.2.cu
    * gauss2.3.cu
    * gauss3.cu
                  

### Convolution

* mask.h
  * convol.cpp
  * convol_omp.h
  * mycuda.h
    * convol_plain.cu + main.cpp
    * convol_shared.cu + main.cpp
    * convol_with_omp.cu + main_2gpu.cpp



## 运行

高斯消去，请使用Gauss/a.sh  
图像卷积，请先把a.jpg移入Convolution下，然后使用Convolution/z.sh
