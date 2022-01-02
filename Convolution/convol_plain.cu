#include "mask.h"
#include "mycuda.h"

#define BLOCK_WIDTH_Y 32
#define BLOCK_WIDTH_X 32
/** 
 * ceil(height / BLOCK_WIDTH_Y)(gridDim.y) * ceil(width /
 * BLOCK_WIDTH_X)(gridDim.x) blocks per Grid BLOCK_WIDTH_Y * BLOCK_WIDTH_X threads per Block
 **/
__global__ void convol(char* InputData, char* OutputData, int height, int width, int widthStep)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // 对应图像的i行, 有效范围0~height-1
    int j = blockIdx.x * blockDim.x + threadIdx.x; // 对应图像的j列, 有效范围0~width-1

    /* 卷积核存储在每个线程私有memory中 */
    getMask();

    /* 对InputData进行卷积, 并将结果返回给OutputData */
    if (i < height && j < width) {
        int b = 0;
        int g = 0;
        int r = 0;

        /**
         * 此时被卷积的像素点是InputData[i][j]
         * 与卷积核中心mask[1][1]正对
         * */
#pragma unroll
        for (int k = 0; k < 3; k++) {
            for (int l = 0; l < 3; l++) {
                int row = i + (k - 1); // 行号相对偏移核中心k - 1
                int col = j + (l - 1); // 列号相对偏移核中心l - 1
                if (row >= 0 && row < height && col >= 0 && col < width) {
                    b += 1 * mask[3 * k + l] * InputData[widthStep * row + 3 * col + 0];
                    g += 1 * mask[3 * k + l] * InputData[widthStep * row + 3 * col + 1];
                    r += 1 * mask[3 * k + l] * InputData[widthStep * row + 3 * col + 2];
                }
            }
        }
        OutputData[widthStep * i + 3 * j + 0] = b;
        OutputData[widthStep * i + 3 * j + 1] = g;
        OutputData[widthStep * i + 3 * j + 2] = r;
    }
}

void beforeConvol(char imgData[], int height, int width, int widthStep, int imgSize)
{
    clock_t tic, toc;

    /**
     * 线程划分
     * GRID_DIM.x = 图像width被分成多少块, GRID_DIM.y = 图像height被分成多少块
     * */
    const dim3 GRID_DIM(ceil(1.0 * width / BLOCK_WIDTH_X), ceil(1.0 * height / BLOCK_WIDTH_Y), 1);
    const dim3 BLOCK_DIM(BLOCK_WIDTH_X, BLOCK_WIDTH_Y, 1);

    /* 内存申请 */
    tic = clock();
    char* InputData = NULL;
    char* OutputData = NULL;
    cudaMalloc((void**)&InputData, imgSize);
    cudaMalloc((void**)&OutputData, imgSize);
    toc = clock();
    double malloc_time = double(toc - tic) / CLOCKS_PER_SEC;

    /* cpu->gpu */
    tic = clock();
    cudaMemcpy(InputData, imgData, imgSize, cudaMemcpyHostToDevice);
    toc = clock();
    double copy_time = double(toc - tic) / CLOCKS_PER_SEC;

    /* 调用gpu核函数 */
    GpuTimer timer;
    timer.Start();
    convol<<<GRID_DIM, BLOCK_DIM>>>(InputData, OutputData, height, width, widthStep);
    timer.Stop();
    float kernel_time = timer.Elapsed() / 1000.0;

    /* gpu->cpu */
    tic = clock();
    cudaMemcpy(imgData, OutputData, imgSize, cudaMemcpyDeviceToHost);
    toc = clock();
    copy_time += double(toc - tic) / CLOCKS_PER_SEC;

    cout << endl
         << "====运行convol_global====" << endl
         << "|-图像高宽: " << height << " * " << width << endl;
    PrintTime(kernel_time, copy_time, malloc_time);

    cudaFree(InputData);
    cudaFree(OutputData);
}
