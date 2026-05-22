#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// Проверка ошибок CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                 << cudaGetErrorString(err) << endl; \
            exit(1); \
        } \
    } while(0)

// Чтение матрицы из файла
vector<vector<double>> readMatrix(const string& filename, int& n) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "Error: cannot open file '" << filename << "'" << endl;
        exit(1);
    }
    if (!(fin >> n) || n <= 0) {
        cerr << "Error: invalid matrix size in '" << filename << "'" << endl;
        exit(1);
    }
    vector<vector<double>> mat(n, vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (!(fin >> mat[i][j])) {
                cerr << "Error: insufficient data in '" << filename << "'" << endl;
                exit(1);
            }
    fin.close();
    return mat;
}

// Запись результата
void writeMatrix(const string& filename, const vector<vector<double>>& mat, double elapsed_ms = -1.0) {
    ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "Error: cannot create file '" << filename << "'" << endl;
        exit(1);
    }
    if (elapsed_ms > 0)
        fout << "# Execution time: " << fixed << setprecision(4) << elapsed_ms << " ms\n";
    int n = mat.size();
    fout << n << "\n" << fixed << setprecision(8);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            fout << mat[i][j] << (j == n - 1 ? "" : " ");
        fout << "\n";
    }
    fout.close();
}

// Ядро CUDA: каждый поток вычисляет один элемент C[i][j]
__global__ void matMulKernel(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    // Параметры по умолчанию
    int n = 0;
    int block_size = 16;  // блок 16×16 потоков
    
    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) block_size = atoi(argv[2]);
    
    if (n <= 0) {
        cerr << "Usage: " << argv[0] << " <matrix_size> [block_size]" << endl;
        cerr << "  block_size: 16, 32, etc. (default: 16)" << endl;
        return 1;
    }

    // Чтение матриц
    char filename_a[256], filename_b[256];
    snprintf(filename_a, sizeof(filename_a), "A%d.txt", n);
    snprintf(filename_b, sizeof(filename_b), "B%d.txt", n);
    
    int n1, n2;
    auto A_2d = readMatrix(filename_a, n1);
    auto B_2d = readMatrix(filename_b, n2);
    
    if (n1 != n2 || n1 != n) {
        cerr << "Error: matrix sizes mismatch" << endl;
        return 1;
    }

    // Преобразование в плоские массивы (row-major)
    vector<double> A_flat(n * n), B_flat(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            A_flat[i * n + j] = A_2d[i][j];
            B_flat[i * n + j] = B_2d[i][j];
        }

    // Выделение памяти на устройстве
    double *d_A, *d_B, *d_C;
    size_t bytes = n * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Копирование данных на GPU
    CUDA_CHECK(cudaMemcpy(d_A, A_flat.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B_flat.data(), bytes, cudaMemcpyHostToDevice));

    // Настройка сетки блоков
    dim3 block(block_size, block_size);
    dim3 grid((n + block_size - 1) / block_size, 
              (n + block_size - 1) / block_size);

    // CUDA events для точного замера времени ядра
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Запуск ядра
    CUDA_CHECK(cudaEventRecord(start));
    matMulKernel<<<grid, block>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, start, stop));

    // Копирование результата обратно на хост
    vector<double> C_flat(n * n);
    CUDA_CHECK(cudaMemcpy(C_flat.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // Освобождение ресурсов
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Преобразование результата в 2D и запись
    vector<vector<double>> C_2d(n, vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C_2d[i][j] = C_flat[i * n + j];

    writeMatrix("result.txt", C_2d, kernel_time_ms);

    // Статистика
    long long flops = 2LL * n * n * n;
    double elapsed_sec = kernel_time_ms / 1000.0;
    double gflops = (flops / 1e9) / elapsed_sec;

    cout << "=== CUDA Matrix Multiplication ===" << endl;
    cout << "Matrix size: " << n << "×" << n << endl;
    cout << "Block configuration: " << block_size << "×" << block_size << endl;
    cout << "Grid size: " << grid.x << "×" << grid.y << " blocks" << endl;
    cout << "Execution time: " << fixed << setprecision(4) << kernel_time_ms << " ms" << endl;
    cout << "Computational volume: " << flops << " FLOPs" << endl;
    cout << "Performance: " << fixed << setprecision(2) << gflops << " GFLOPS" << endl;
    cout << "Result saved to: result.txt" << endl;

    return 0;
}