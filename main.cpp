#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

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
    if (elapsed_ms > 0) {
        fout << "# Execution time: " << fixed << setprecision(8) << elapsed_ms << " ms\n";
    }
    int n = mat.size();
    fout << n << "\n" << fixed << setprecision(8);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            fout << mat[i][j] << (j == n - 1 ? "" : " ");
        fout << "\n";
    }
    fout.close();
}

// Расчёт количества строк для процесса с данным рангом
pair<int, int> getRowRange(int rank, int size, int n) {
    int rows_per_proc = n / size;
    int remainder = n % size;
    int start_row = rank * rows_per_proc + min(rank, remainder);
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    return {start_row, local_rows};
}

int main(int argc, char* argv[]) {
    // Инициализация MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 0;
    if (argc >= 2) n = atoi(argv[1]);
    
    // Рассылка размера матрицы всем процессам
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (n <= 0) {
        if (rank == 0) {
            cerr << "Usage: mpirun -np <procs> " << argv[0] << " <matrix_size>" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Используем массивы для эффективной работы с MPI
    vector<double> A_flat, BT_flat;
    
    // Читает файлы и подготавливает данные
    if (rank == 0) {
        char filename_a[256], filename_b[256];
        snprintf(filename_a, sizeof(filename_a), "A%d.txt", n);
        snprintf(filename_b, sizeof(filename_b), "B%d.txt", n);

        int n1, n2;
        auto A_2d = readMatrix(filename_a, n1);
        auto B_2d = readMatrix(filename_b, n2);

        if (n1 != n2 || n1 != n) {
            cerr << "Error: matrix sizes do not match (" << n1 << " vs " << n2 << " vs " << n << ")" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Транспонирование B для оптимизации доступа к памяти
        A_flat.resize(n * n);
        BT_flat.resize(n * n);
        
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A_flat[i * n + j] = A_2d[i][j];
                BT_flat[i * n + j] = B_2d[j][i]; 
            }
        }
    }
    else {
        // Остальные процессы выделяют память для приёма данных
        A_flat.resize(n * n);
        BT_flat.resize(n * n);
    }

    // Рассылка матриц всем процессам
    MPI_Bcast(A_flat.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(BT_flat.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Распределение строк между процессами
    auto [start_row, local_rows] = getRowRange(rank, size, n);
    
    // Локальная часть результата
    vector<double> C_local_flat(local_rows * n, 0.0);

    // Замер времени
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Вычисление только строк
    for (int i_local = 0; i_local < local_rows; ++i_local) {
        int i_global = start_row + i_local;
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C_local_flat[i_local * n + j] += A_flat[i_global * n + k] * BT_flat[j * n + k];
            }
        }
    }

    double elapsed_ms = (MPI_Wtime() - start_time) * 1000.0;

    // Подготовка параметров для MPI_Gatherv
    vector<int> recv_counts(size), displs(size);
    for (int p = 0; p < size; ++p) {
        auto [p_start, p_rows] = getRowRange(p, size, n);
        recv_counts[p] = p_rows * n;           // количество элементов от процесса p
        displs[p] = p_start * n;               // смещение в итоговом массиве
    }

    // Сбор результатов
    vector<double> C_flat(n * n);
    MPI_Gatherv(C_local_flat.data(), local_rows * n, MPI_DOUBLE,
                C_flat.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // [MPI] Записывает результат и выводит статистику
    if (rank == 0) {
        vector<vector<double>> C_2d(n, vector<double>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                C_2d[i][j] = C_flat[i * n + j];

        writeMatrix("result.txt", C_2d, elapsed_ms);

        // Расчёт производительности
        long long flops = 2LL * n * n * n;
        double elapsed_sec = elapsed_ms / 1000.0;
        double gflops = (flops / 1e9) / elapsed_sec;

        cout << "=== MPI Matrix Multiplication ===" << endl;
        cout << "Matrix size: " << n << "×" << n << endl;
        cout << "MPI processes: " << size << endl;
        cout << "Execution time: " << fixed << setprecision(4) << elapsed_ms << " ms" << endl;
        cout << "Computational volume: " << flops << " FLOPs" << endl;
        cout << "Performance: " << fixed << setprecision(2) << gflops << " GFLOPS" << endl;
        cout << "Result saved to: result.txt" << endl;
    }

    MPI_Finalize();
    return 0;
}