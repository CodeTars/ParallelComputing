#include <cuda.h>
#include <iostream>

using namespace std;

/* 核函数执行时间计时器 */
struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;
    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void Start() { cudaEventRecord(start, 0); }
    void Stop() { cudaEventRecord(stop, 0); }
    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void PrintTime(double kernel_time, double copy_time, double malloc_time)
{
    cout << "|-总共耗时: " << (kernel_time + copy_time + malloc_time) << " s" << endl
         << "    |" << endl
         << "    |-GPU卷积: " << kernel_time << " s" << endl
         << "    |-内存传输: " << copy_time << " s" << endl
         << "    |-内存申请: " << malloc_time << " s" << endl
         << endl;
}