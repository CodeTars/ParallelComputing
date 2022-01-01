#include "h_cpu.h"

int main(int argc, char* argv[])
{
    int n;
    double *A, *b;
    Input(A, b, n);

    clock_t tic, toc;
    double time1 = 0, time2 = 0;
    for (int k = 0; k < n - 1; k++) {

        /* 交换两行 */
        tic = clock();
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
        toc = clock();
        time1 += double(toc - tic) / CLOCKS_PER_SEC;

        /* 高斯消元 */
        tic = clock();
        for (int i = k + 1; i < n; i++) {
            double m = A[n * i + k] / A[n * k + k];
            for (int j = k + 1; j < n; j++) {
                A[n * i + j] -= m * A[n * k + j];
            }
            b[i] -= m * b[k];
        }
        toc = clock();
        time2 += double(toc - tic) / CLOCKS_PER_SEC;
    }

    /* 回代求解X */
    tic = clock();
    backSubstitution(A, b, n);
    toc = clock();
    double time3 = double(toc - tic) / CLOCKS_PER_SEC;

    /* 输出 */
    Output(b, n);
    PrintTime("gauss_cpu", n, time1, time2, time3);

    delete[] A, b;
    return 0;
}
