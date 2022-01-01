#include "h_common.h"

/* 回代求X */
void backSubstitution(const double A[], double b[], int n)
{
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += A[n * i + j] * b[j];
        }
        b[i] = (b[i] - sum) / A[n * i + i];
    }
}

/* 增广矩阵的输入 */
void Input(double*& A, double*& b, int& n)
{
    ifstream fin("matrix_ofEquation.txt");
    fin >> n;
    A = new double[n * n];
    b = new double[n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fin >> A[n * i + j];
        }
        fin >> b[i];
    }
    fin.close();
    return;
}

/* 方程组解x的输出 */
void Output(const double* b, int n)
{
    ofstream fout("x.txt");
    for (int i = 0; i < n; i++) {
        fout << b[i] << endl;
    }
    fout.close();
    return;
}

/* 输出时耗 */
void PrintTime(const char* name, int n, double time1, double time2, double time3)
{
#ifdef FORMAT
    cout << setw(FORMAT) << time1 + time2 + time3;
#else
    cout << endl
         << "====运行" << name << "====" << endl
         << "|" << endl
         << "|-方程规模: " << n << "元一次方程组" << endl
         << "|" << endl
         << "|-总共耗时: " << time1 + time2 + time3 << " s" << endl
         << "    |" << endl
         << "    |-交换两行: " << time1 << " s" << endl
         << "    |-高斯消元: " << time2 << " s" << endl
         << "    |-回代解X: " << time3 << " s" << endl
         << endl;
#endif
}