#include "h_cpu.h"
#include <omp.h>

int main(int argc, char* argv[])
{
    int n;
    double *A, *b;
    Input(A, b, n);

    double time1 = 0, time2 = 0;
    for (int k = 0; k < n - 1; k++) {

        /* 交换两行 */
        time1 -= omp_get_wtime();
        int l = k;
        double mx = fabs(A[n * k + k]);
        for (int i = k + 1; i < n; i++) {
            if (fabs(A[n * i + k]) > mx) {
                mx = fabs(A[n * i + k]);
                l = i;
            }
        }
        if (l != k) {
            for (int j = k; j < n; j++) {
                swap(A[n * l + j], A[n * k + j]);
            }
            swap(b[l], b[k]);
        }
        time1 += omp_get_wtime();

        /* 高斯消元 */
        time2 -= omp_get_wtime();
#pragma omp parallel for num_threads(29)
        for (int i = k + 1; i < n; i++) {
            double m = A[n * i + k] / A[n * k + k];
            for (int j = k + 1; j < n; j++) {
                A[n * i + j] -= m * A[n * k + j];
            }
            b[i] -= m * b[k];
        }
        time2 += omp_get_wtime();
    }

    /* 回代求解X */
    double time3 = 0;
    time3 -= omp_get_wtime();
    backSubstitution(A, b, n);
    time3 += omp_get_wtime();

    /* 输出 */
    Output(b, n);
    PrintTime("gauss_omp", n, time1, time2, time3);

    delete[] A, b;
    return 0;
}