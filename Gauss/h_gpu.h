#include "h_common.h"
#include <cuda.h>

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

/* 回代求解方程组的解x */
void backSubstitution(double A[], int n)
{
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += A[(n + 1) * i + j] * A[(n + 1) * j + n];
        }
        A[(n + 1) * i + n] = (A[(n + 1) * i + n] - sum) / A[(n + 1) * i + i];
    }
}

/* 输出时耗 */
void PrintTime(const char* name, int n, double malloc_time, double copy_time, double kernel_time1, double kernel_time2, double kernel_time3, double backSub_time) 
{
    double kernel_time = kernel_time1 + kernel_time2 + kernel_time3;
    #ifdef FORMAT
        cout << setw(FORMAT) << (malloc_time + copy_time + kernel_time + backSub_time);
    #else
        cout << endl
            << "====运行" << name << "====" << endl
            << "|" << endl
            << "|-方程规模: " << n << "元一次方程组"<< endl
            << "|" << endl
            << "|-总共耗时: " << (malloc_time + copy_time + kernel_time + backSub_time) << " s" << endl
            << "    |" << endl
            << "    |-GPU中化三角阵: " << kernel_time << " s" << endl
            << "    |     |" << endl
            << "    |     |-找列主元: " << kernel_time1 << " s" << endl
            << "    |     |-交换两行: " << kernel_time2 << " s" << endl
            << "    |     |-高斯消元: " << kernel_time3 << " s" << endl
            << "    |" << endl
            << "    |-内存传输: " << copy_time << " s" << endl
            << "    |" << endl
            << "    |-内存申请: " << malloc_time << " s" << endl
            << "    |" << endl
            << "    |-CPU中回代解x: " << backSub_time << " s" << endl
            << endl;
    #endif
}

/* 增广矩阵的输入 */
void Input(double*& A, int& n)
{
    ifstream fin("matrix_ofEquation.txt");
    fin >> n;
    A = new double[(n + 1) * n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n + 1; j++) {
            fin >> A[(n + 1) * i + j];
        }
    }
    fin.close();
    return;
}

/* 方程组解x的输出 */
void Output(const double A[], int n)
{
    ofstream fout("x.txt");
    for (int i = 0; i < n; i++) {
        fout << A[(n + 1) * i + n] << endl;
    }
    fout.close();
}

/* 增广矩阵的输出 */
void PrintMatrix(const double A[], int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n + 1; j++) {
            cout << A[(n + 1) * i + j] << '\t';
        }
        cout << endl;
    }
}