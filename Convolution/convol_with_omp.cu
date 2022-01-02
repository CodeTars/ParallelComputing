#include "mask.h"
#include "mycuda.h"
#include <omp.h>

#define BLOCK_WIDTH_Y 32
#define BLOCK_WIDTH_X 32
#define TILE_WIDTH_X (BLOCK_WIDTH_X - 2)
#define TILE_WIDTH_Y (BLOCK_WIDTH_Y - 2)

/**
 * 线程划分
 * GRID_DIM.x = 图像width被分成多少块, GRID_DIM.y = 图像height被分成多少块
 * const dim3 GRID_DIM(ceil(1.0 * width / TILE_WIDTH_X), ceil(1.0 * height / TILE_WIDTH_Y), 1) 
 * const dim3 BLOCK_DIM(BLOCK_WIDTH_X, BLOCK_WIDTH_Y,1)
 *
 * 每个block处理的图像片段tile高宽 TILE_WIDTH_Y * TILE_WIDTH_X,
 * 每个block内的线程数是 (TILE_WIDTH_Y + 2) * (TILE_WIDTH_X + 2),
 * 因为要有线程处理读入图像片段边缘外一圈的值.
 * */
__global__ void convol(char* InputData, char* OutputData, int height, int width,
    int widthStep, int tag)
{
    /**
   * InputData被分为若干个片段tile
   * (ty, tx) = (1, 1)时,
   * 对应当前tile内左上角像素点, 坐标为InputData中的(TILE_WIDTH_Y * blockIdx.y, TILE_WIDTH_X * blockIdx.x)
   * 
   * 1 <= ty <= TILE_WIDTH_Y, 1 <= ty <= TILE_WIDTH_X 时, 
   * 偏移左上角 ty - 1 和 tx - 1 对应当前tile内, (TILE_WIDTH_Y * blockIdx.y, TILE_WIDTH_X * blockIdx.x) + (ty - 1, tx - 1)
   * */
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = TILE_WIDTH_Y * blockIdx.y + (ty - 1);
    int col = TILE_WIDTH_X * blockIdx.x + (tx - 1);

    /* 卷积核存储在每个线程私有memory中 */
    getMask();

    /**
   * InputData内除了边缘, 每个点要被访问9次
   * 先读入block内的 shared memory, 再进行访问, 能降低访问延迟
   * */
    __shared__ char b[BLOCK_WIDTH_Y][BLOCK_WIDTH_X];
    __shared__ char g[BLOCK_WIDTH_Y][BLOCK_WIDTH_X];
    __shared__ char r[BLOCK_WIDTH_Y][BLOCK_WIDTH_X];

    /** 如果没有越界, 读入; 否者设为纯黑点
   * 对于tag == 0, 要读入的行号范围是  0 <= row <= height, InputData指向0行
   * 对于tag == 1, 要读入的行号范围是 -1 <= row <= height - 1. InputData指向-1行
   */
    if (row >= 0 - tag && row <= height - tag && col >= 0 && col < width) {
        b[ty][tx] = InputData[widthStep * (tag + row) + 3 * col + 0];
        g[ty][tx] = InputData[widthStep * (tag + row) + 3 * col + 1];
        r[ty][tx] = InputData[widthStep * (tag + row) + 3 * col + 2];
    } else {
        b[ty][tx] = 0;
        g[ty][tx] = 0;
        r[ty][tx] = 0;
    }

    /* 等待b, g, r全部被初始化 */
    __syncthreads();

    /* 对tile进行卷积, 并将结果返回给OutputData */
    if (ty >= 1 && ty <= TILE_WIDTH_Y && tx >= 1 && tx <= TILE_WIDTH_X) {
        int bb = 0;
        int gg = 0;
        int rr = 0;
        /** 
         * 此时被卷积的像素点是InputData[tag + row][col] = (bgr)[ty][tx]
         * 与卷积核中心mask[1][1]正对
         */
#pragma unroll
        for (int k = 0; k < 3; k++) {
            for (int l = 0; l < 3; l++) {
                int i = ty + (k - 1); // 行号相对偏移核中心k - 1
                int j = tx + (l - 1); // 列号相对偏移核中心l - 1
                bb += 1 * mask[3 * k + l] * b[i][j];
                gg += 1 * mask[3 * k + l] * g[i][j];
                rr += 1 * mask[3 * k + l] * r[i][j];
            }
        }
        /* 如果没有越界, 写入 */
        if (row >= 0 && row < height && col >= 0 && col < width) {
            OutputData[widthStep * row + 3 * col + 0] = bb;
            OutputData[widthStep * row + 3 * col + 1] = gg;
            OutputData[widthStep * row + 3 * col + 2] = rr;
        }
    }
}

void beforeConvol(char imgData[], int height, int width, int widthStep,
    int imgSize, int tag)
{
    // cudaSetDevice(tag);
    cudaSetDevice(0);

    /**
     * 线程划分
     * GRID_DIM.x = 图像width被分成多少块, GRID_DIM.y = 图像height被分成多少块
     * grid内block的划分根据是, 每个block处理的图像片段tile高宽 TILE_WIDTH_Y *
     * TILE_WIDTH_X block的线程数是 (TILE_WIDTH_Y + 2) * (TILE_WIDTH_X + 2),
     * 因为要有线程处理读入图像片段边缘外一圈的值.
     * */
    const dim3 GRID_DIM(ceil(1.0 * width / TILE_WIDTH_X), ceil(1.0 * height / TILE_WIDTH_Y), 1);
    const dim3 BLOCK_DIM(BLOCK_WIDTH_X, BLOCK_WIDTH_Y, 1);

    /*  内存申请 */
    char* InputData = NULL;
    char* OutputData = NULL;
    // 对于上半图像, 要包含下边缘的下一行; 对于下半图像, 要包含上边缘的上一行
    cudaMalloc((void**)&InputData, imgSize + widthStep);
    // 输出只需要height行
    cudaMalloc((void**)&OutputData, imgSize);

    /* cpu->gpu */
    cudaMemcpy(InputData, imgData - tag * widthStep, imgSize + widthStep, cudaMemcpyHostToDevice);

    /* 调用gpu核函数 */
    /** InputData 指向上半图像0行, if tag == 0
     * InputData 指向下半图像-1行, if tag == 1
     **/
    convol<<<GRID_DIM, BLOCK_DIM>>>(InputData, OutputData, height, width, widthStep, tag);

    /* gpu->cpu */
    cudaMemcpy(imgData, OutputData, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(InputData);
    cudaFree(OutputData);
}
