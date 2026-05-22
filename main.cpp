#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdlib>

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

    fout << "# Execution time: " << fixed << setprecision(4) << elapsed_ms << " ms\n";

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

int main() {
    int n1, n2;

    vector<vector<double>> A = readMatrix("A.txt", n1);
    vector<vector<double>> B = readMatrix("B.txt", n2);

    if (n1 != n2) {
        cerr << "Error: matrix sizes do not match (" << n1 << " vs " << n2 << ")" << endl;
        return 1;
    }
    int n = n1;

    vector<vector<double>> C(n, vector<double>(n, 0.0));


    auto start = high_resolution_clock::now();
    // Перемножение матриц
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    auto end = high_resolution_clock::now();
    double elapsed_ms = duration<double, milli>(end - start).count();

    long long flops = 2LL * n * n * n - 1LL * n * n;

    writeMatrix("result.txt", C, elapsed_ms);

    cout << "Matrix size: " << n << "x" << n << endl;
    cout << "Execution time: " << fixed << setprecision(4) << elapsed_ms << " ms" << endl;
    cout << "Task volume (FLOPs): " << flops << endl;
    cout << "Performance: " << fixed << setprecision(2)
         << (flops / 1e9) / (elapsed_ms / 1000.0) << " GFLOPS" << endl;
    cout << "Result saved to: result.txt" << endl;

    return 0;
}