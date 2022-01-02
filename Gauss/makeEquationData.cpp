#include "h_common.h"

int main(int argc, char* argv[])
{

#ifdef FORMAT
    cout << "n = ";
#else
    cout << endl
         << "====运行makeEquationData====" << endl
         << "将随机生成一个n元线性方程组Ax=b, A和x中各元取值于1, 2, ..., 1000。" << endl
         << "方程增广矩阵[A|b]输出到matrix_ofEquation.txt, 方程的解x输出到stdx.txt。 n = ";
#endif

    int n;
    if (argc == 1) {
        cin >> n;
    } else {
        n = atoi(argv[1]);
        cout << n << endl;
    }

    cout << endl;

    ofstream fout;
    srand((int)time(0));
    fout.open("stdx.txt");
    int* x = new int[n];
    for (int i = 0; i < n; i++) {
        x[i] = 1 + rand() % 1000;
        fout << x[i] << endl;
    }
    fout.close();

    fout.open("matrix_ofEquation.txt");
    fout << n << endl;
    for (int i = 0; i < n; i++) {
        long long b = 0;
        for (int j = 0; j < n; j++) {
            int a = 1 + rand() % 1000;
            b += x[j] * a;
            fout << a << '\t';
        }
        fout << b << endl;
    }

    fout.close();
    delete[] x;

    return 0;
}