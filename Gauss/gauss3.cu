#include "h_gpu.h"

/** 多线程, 并行规约从[Ak,k ... An-1,k]中寻找列主元
 * only 1 block, 1(blockDim.y) * FindMain_BlockDim_X(blockDim.x) threads per Block
 * */
#define FindMain_BlockDim_X 128
__global__ void FindMainElem_kCol(double* a, int n, int k, int* id)
{
    int tx = threadIdx.x; // 0 ~ FindMain_BlockDim_X - 1
    __shared__ int l[FindMain_BlockDim_X];
    __shared__ double mx[FindMain_BlockDim_X];

    /* Block内全部线程并行求解该列的部分元素的最大元 */
    int i = k + tx;
    mx[tx] = 0;
    while (i < n) {
        if (fabs(a[(n + 1) * i + k]) > mx[tx]) {
            mx[tx] = fabs(a[(n + 1) * i + k]);
            l[tx] = i;
        }
        i += FindMain_BlockDim_X;
    }
    __syncthreads();

    /* 归约块内结果 */
    if (tx == 0) {
        for (int i = 1; i < FindMain_BlockDim_X; i++) {
            if (mx[i] > mx[0]) {
                mx[0] = mx[i];
                l[0] = l[i];
            }
        }
        *id = l[0];
    }
}

#define Swap(a, b)    \
    {                 \
        double t = a; \
        a = b;        \
        b = t;        \
    }
/**
 * 多线程, 并行换行
 * [Ak,k ... Ak,n]<-->[Al,k ... Al,n]
 * 1(gridDim.y) * ceil((n - k + 1) / SwapRowBlockDim_x)(gridDim.x) blocks per Grid
 * 1(blockDim.y) * SwapRowBlockDim_x(blockDim.x) threads per Block
 * */
#define SwapRowBlockDim_X 32
__global__ void SwapRow(double* a, int n, int k, int* l)
{
    if (k != (*l)) {
        int j = k + (blockIdx.x * blockDim.x + threadIdx.x);
        if (j < n + 1) {
            Swap(a[(n + 1) * k + j], a[(n + 1) * (*l) + j]);
        }
    }
}

/**
 * 多线程, 并行消元
 * 需要操作的元素为
 * ----> x
 * | |Ak+1,k+1 ... Ak+1,n|
 * ' |                   |
 * y |    .           .  |
 *   |    .           .  |
 *   |An-1,k+1 ... An-1,n|
 * 
 * (n - k - 1)(gridDim.y) * ceil((n - k) / EliminationBlockDim_X)(gridDim.x) blocks per Grid
 * 1(blockDim.y) * EliminationBlockDim_X(blockDim.x) threads per Block
 * */
#define EliminationBlockDim_Y 1
#define EliminationBlockDim_X 1024
__global__ void GaussElimination(double* a, int n, int k)
{
    int i = (k + 1) + blockIdx.y; // threadIdx.y 恒为0
    int j = (k + 1) + blockIdx.x * blockDim.x + threadIdx.x; // 从Ak+1,k+1开始

    __shared__ double factor;
    if (threadIdx.x == 0) {
        factor = a[(n + 1) * i + k] / a[(n + 1) * k + k];
    }
    __syncthreads();

    if (i < n && j < n + 1) {
        a[(n + 1) * i + j] -= factor * a[(n + 1) * k + j];
    } // 消元
}

int main(int argc, char* argv[])
{
    /* 方程组（增广矩阵）输入 */
    int n;
    double* A; // 主机内存中的增广矩阵
    Input(A, n);

    clock_t tic, toc;

    /* 内存分配 */
    tic = clock();
    int* l = NULL;
    double* a = NULL; // GPU内存中的增广矩阵
    cudaMalloc((int**)&l, sizeof(int));
    cudaMalloc((void**)&a, sizeof(double) * (n + 1) * n);
    toc = clock();
    double malloc_time = double(toc - tic) / CLOCKS_PER_SEC;

    /* CPU->GPU */
    tic = clock();
    cudaMemcpy(a, A, sizeof(double) * (n + 1) * n, cudaMemcpyHostToDevice);
    toc = clock();
    double copy_time = double(toc - tic) / CLOCKS_PER_SEC;

    /* 消元过程 */
    GpuTimer timer;
    float kernel_time1 = 0, kernel_time2 = 0, kernel_time3 = 0;
    for (int k = 0; k < n - 1; k++) {

        /* 多线程寻找, k列的列主元 */
        timer.Start();
        FindMainElem_kCol<<<1, FindMain_BlockDim_X>>>(a, n, k, l);
        timer.Stop();
        kernel_time1 += timer.Elapsed() / 1000.0;

        /* 交换k,l两行 */
        timer.Start();
        SwapRow<<<ceil(1.0 * (n + 1 - k) / SwapRowBlockDim_X), SwapRowBlockDim_X>>>(a, n, k, l);
        timer.Stop();
        kernel_time2 += timer.Elapsed() / 1000.0;

        /**
         * 高斯消元
         * 线程划分: GRID_DIM.x = 矩阵第k+1列 ~ 第n列被分成多少块, GRID_DIM.y = 矩阵的行被分成多少块
         * */
        timer.Start();
        dim3 GRID_DIM(ceil(1.0 * (n - k) / EliminationBlockDim_X), (n - 1 - k), 1);
        dim3 BLOCK_DIM(EliminationBlockDim_X, 1, 1);
        GaussElimination<<<GRID_DIM, BLOCK_DIM>>>(a, n, k);
        timer.Stop();
        kernel_time3 += timer.Elapsed() / 1000.0;

    } // 此时增广矩阵已变换成了下三角阵

    /* GPU->CPU */
    tic = clock();
    cudaMemcpy(A, a, sizeof(double) * n * (n + 1), cudaMemcpyDeviceToHost);
    toc = clock();
    copy_time += double(toc - tic) / CLOCKS_PER_SEC;

    /* 回代求解方程组的解x */
    tic = clock();
    backSubstitution(A, n);
    toc = clock();
    double backSub_time = double(toc - tic) / CLOCKS_PER_SEC;

    /* 输出 */
    Output(A, n);
    PrintTime("gauss3", n, malloc_time, copy_time, kernel_time1, kernel_time2, kernel_time3, backSub_time);
    cout << endl;

    delete[] A;
    cudaFree(a);
    cudaFree(l);
    return 0;
}
