#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// Чтение
vector<vector<double>> readMatrix(const string& filename, int& n) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "Error: not open file '" << filename << "'" << endl;
        exit(1);
    }

    if (!(fin >> n) || n <= 0) {
        cerr << "Error: invalid matrix size in '" << filename << "'" << endl;
        exit(1);
    }

    vector<vector<double>> mat(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!(fin >> mat[i][j])) {
                cerr << "Error: insufficient data in '" << filename << "'" << endl;
                exit(1);
            }
        }
    }
    fin.close();
    return mat;
}

// Запись
void writeMatrix(const string& filename, const vector<vector<double>>& mat, double elapsed_ms = -1.0) {
    ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "Error: cannot create file '" << filename << "'" << endl;
        exit(1);
    }

    fout << "# Execution time: " << fixed << setprecision(8) << elapsed_ms << " ms\n";

    int n = mat.size();
    fout << n << "\n";
    fout << fixed << setprecision(8);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fout << mat[i][j] << (j == n - 1 ? "" : " ");
        }
        fout << "\n";
    }
    fout.close();
}

int main(int argc, char* argv[]) {
    int n = 0;
    int num_threads = 0;

    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) num_threads = atoi(argv[2]);

    if (n <= 0) {
        cerr << "Usage: " << argv[0] << " <matrix_size> [num_threads]" << endl;
        return 1;
    }

    char filename_a[256], filename_b[256];
    snprintf(filename_a, sizeof(filename_a), "A%d.txt", n);
    snprintf(filename_b, sizeof(filename_b), "B%d.txt", n);

    vector<vector<double>> A, B;
    int n1, n2;

    A = readMatrix(filename_a, n1);
    B = readMatrix(filename_b, n2);

    vector<vector<double>> BT(n, vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            BT[j][i] = B[i][j];

    if (n1 != n2) {
        cerr << "Error: matrix sizes do not match (" << n1 << " vs " << n2 << ")" << endl;
        return 1;
    }
    n = n1;

    vector<vector<double>> C(n, vector<double>(n, 0.0));

    // [OMP] Установка числа потоков и замер времени
    omp_set_num_threads(num_threads);
    double start_time = omp_get_wtime();

    // [OMP] Параллелизация умножения
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i][k] * BT[j][k];
            }
            C[i][j] = sum;
        }
    }

    double elapsed_ms = (omp_get_wtime() - start_time) * 1000.0;

    long long flops = 2LL * n * n * n - 1LL * n * n;

    writeMatrix("result.txt", C, elapsed_ms);

    // [OMP] Вывод информации о потоках
    cout << "Matrix size: " << n << "x" << n << endl;
    cout << "Threads used: " << num_threads << endl;
    cout << "Execution time: " << fixed << setprecision(4) << elapsed_ms << " ms" << endl;
    cout << "Task volume (FLOPs): " << flops << endl;
    cout << "Performance: " << fixed << setprecision(2)
         << (flops / 1e9) / (elapsed_ms / 1000.0) << " GFLOPS" << endl;
    cout << "Result saved to: result.txt" << endl;

    return 0;
}