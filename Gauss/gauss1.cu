#include "h_gpu.h"

#define Swap(a, b)    \
    {                 \
        double t = a; \
        a = b;        \
        b = t;        \
    }

/**
 * 单线程，从[Ak,k ... An-1,k]中寻找列主元的索引l,
 * 后交换k,l两行 [Ak,k ... Ak,n]<-->[Al,k ... Al,n]
 * */
__global__ void SwapRow(double* a, int n, int k)
{
    /* 求第k列的最大元 */
    int l = k;
    double mx = fabs(a[(n + 1) * k + k]);
    for (int i = k + 1; i < n; i++) {
        if (fabs(a[(n + 1) * i + k]) > mx) {
            mx = fabs(a[(n + 1) * i + k]);
            l = i;
        }
    }
    /* 交换k,l两行 */
    if (k != l) {
        for (int j = k; j < n + 1; j++) {
            Swap(a[(n + 1) * k + j], a[(n + 1) * l + j]);
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
 * ceil(n / EliminationBlockDim_Y)(gridDim.y) * ceil((n + 1) / EliminationBlockDim_X)(gridDim.x) blocks per Grid
 * EliminationBlockDim_Y(blockDim.y) * EliminationBlockDim_X(blockDim.x) threads per Block
 * */
#define EliminationBlockDim_Y 32
#define EliminationBlockDim_X 32
__global__ void GaussElimination(double* a, int n, int k)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > k && i < n && j > k && j < n + 1) {
        a[(n + 1) * i + j] -= (a[(n + 1) * i + k] / a[(n + 1) * k + k]) * a[(n + 1) * k + j];
    } // 消元
}

int main(int argc, char* argv[])
{
    clock_t tic, toc;

    /* 方程组（增广矩阵）输入 */
    int n;
    double* A; // 主机内存中的增广矩阵
    Input(A, n);

    /**
     * 消元过程线程划分
     * GRID_DIM.x = 矩阵的列数n+1被分成多少块, GRID_DIM.y = 矩阵的行数n被分成多少块
     * */
    const dim3 GRID_DIM(ceil(1.0 * (n + 1) / EliminationBlockDim_X), ceil(1.0 * n / EliminationBlockDim_Y), 1);
    const dim3 BLOCK_DIM(EliminationBlockDim_X, EliminationBlockDim_Y, 1);

    /* 内存分配 */
    tic = clock();
    double* a = NULL; // GPU内存中的增广矩阵
    cudaMalloc((void**)&a, sizeof(double) * n * (n + 1));
    toc = clock();
    double malloc_time = double(toc - tic) / CLOCKS_PER_SEC;

    /* CPU->GPU */
    tic = clock();
    cudaMemcpy(a, A, sizeof(double) * n * (n + 1), cudaMemcpyHostToDevice);
    toc = clock();
    double copy_time = double(toc - tic) / CLOCKS_PER_SEC;

    /* 消元过程 */
    GpuTimer timer;
    float kernel_time1 = 0, kernel_time2 = 0, kernel_time3 = 0;
    for (int k = 0; k < n - 1; k++) {

        /* 交换两行 */
        timer.Start();
        SwapRow<<<1, 1>>>(a, n, k);
        timer.Stop();
        kernel_time2 += timer.Elapsed() / 1000.0;

        /* 高斯消元 */
        timer.Start();
        GaussElimination<<<GRID_DIM, BLOCK_DIM>>>(a, n, k);
        timer.Stop();
        kernel_time3 += timer.Elapsed() / 1000.0;

    } // 此时增广矩阵已变换成了上三角阵

    /* GPU->CPU */
    tic = clock();
    cudaMemcpy(A, a, sizeof(double) * n * (n + 1), cudaMemcpyDeviceToHost);
    toc = clock();
    copy_time += double(toc - tic) / CLOCKS_PER_SEC;

    /* cpu内回代过程 */
    tic = clock();
    backSubstitution(A, n);
    toc = clock();
    double backSub_time = double(toc - tic) / CLOCKS_PER_SEC;

    /* 输出 */
    Output(A, n);
    PrintTime("gauss1", n, malloc_time, copy_time, kernel_time1, kernel_time2, kernel_time3, backSub_time);

    delete[] A;
    cudaFree(a);
    return 0;
}
