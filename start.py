import subprocess
import numpy as np
import sys
import os
import csv
import re


EXEC = "matmul_omp.exe"
MATRIX_SIZES = [200, 400, 800, 1200, 1600, 2000]
THREAD_COUNTS = [1, 2, 4, 6, 8, 12]
RESULTS_FILE = "results_table.csv"
VERIFY_UP_TO = 1200

def run_program(n, num_threads):
    """
    Запускает matmul_omp.exe с заданными параметрами.
    Возвращает словарь с метриками или None при ошибке.
    """
    cmd = [EXEC, str(n), str(num_threads)]
    print(f"  → Запуск: {' '.join(cmd)}", flush=True)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
    except subprocess.TimeoutExpired:
        print("Таймаут выполнения", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(f"Ошибка выполнения:\n{result.stderr}", file=sys.stderr)
        return None

    metrics = {}
    for line in result.stdout.split('\n'):
        if 'Execution time:' in line:
            match = re.search(r'([\d.]+)\s*ms', line)
            if match:
                metrics['time_ms'] = float(match.group(1))
        elif 'Performance:' in line:
            match = re.search(r'([\d.]+)\s*GFLOPS', line)
            if match:
                metrics['gflops'] = float(match.group(1))

    if 'time_ms' not in metrics or 'gflops' not in metrics:
        print("Не удалось распарсить вывод программы", file=sys.stderr)
        return None

    return metrics


def read_matrix_from_file(filepath):
    """Читает матрицу из файла в формате: первая строка - размер, далее - данные"""
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f.readlines()
                 if line.strip() and not line.strip().startswith('#')]

    if not lines:
        raise ValueError(f"Файл {filepath} пуст или не содержит данных.")

    data = []
    for line in lines:
        data.extend(line.split())

    n = int(data[0])
    if len(data) != n * n + 1:
        raise ValueError(f"Неверное количество элементов в {filepath}.")

    values = np.array(data[1:], dtype=np.float64)
    return values.reshape(n, n)


def verify_results(n):
    """
    Сравнивает результат C++ программы с эталонным умножением через numpy.
    Возвращает True, если погрешность в допустимых пределах.
    """
    try:
        A = read_matrix_from_file(f"A{n}.txt")
        B = read_matrix_from_file(f"B{n}.txt")
        C_cpp = read_matrix_from_file("result.txt")

        C_ref = np.dot(A, B)

        is_match = np.allclose(C_cpp, C_ref, atol=1e-5, rtol=1e-5)
        max_err = np.max(np.abs(C_cpp - C_ref))

        status = "PASSED" if is_match else "FAILED"
        print(f"Верификация: {status}, погрешность: {max_err:.2e}")
        return is_match
    except FileNotFoundError as e:
        print(f"Файл не найден, пропускаем верификацию: {e}")
        return False
    except Exception as e:
        print(f"Ошибка верификации: {e}")
        return False


def init_results_file():
    """Создаёт новый CSV-файл с заголовками столбцов"""
    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Matrix_Size',      # Размер матрицы (N×N)
            'Threads',          # Число потоков
            'Time_ms',          # Время выполнения (мс)
            'GFLOPS',           # Производительность (млрд оп/сек)
            'Verified',         # Результат верификации (YES/NO)
        ])


def save_result(n, threads, metrics, verified):
    """Добавляет одну строку с результатами в конец CSV-файла"""
    with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            n,
            threads,
            f"{metrics['time_ms']:.4f}",
            f"{metrics['gflops']:.2f}",
            'YES' if verified else 'NO',
        ])

def print_summary():
    """Выводит удобочитаемую таблицу результатов в консоль"""
    print("\nводная таблица результатов (время в мс):")
    print(f"{'Size':>7} | " + " ".join(f"{t:>9}" for t in THREAD_COUNTS))
    print("-" * (7 + 3 + 10 * len(THREAD_COUNTS)))

    try:
        results = {}
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                n = int(row['Matrix_Size'])
                t = int(row['Threads'])
                time_val = float(row['Time_ms'])
                if n not in results:
                    results[n] = {}
                results[n][t] = time_val

        for n in MATRIX_SIZES:
            if n in results:
                times = [results[n].get(t, '-') for t in THREAD_COUNTS]
                times_str = [f"{val:>9.2f}" if isinstance(val, (int, float)) else f"{val:>9}" for val in times]
                print(f"{n:>7} | " + " ".join(times_str))
    except Exception as e:
        print(f"(Не удалось вывести сводку: {e})")


def main():
    print("=== OpenMP Matrix Multiplication Experiments ===")
    print(f"Executable: {EXEC}")
    print(f"Matrix sizes: {MATRIX_SIZES}")
    print(f"Thread counts: {THREAD_COUNTS}")
    print(f"Results file: {RESULTS_FILE}\n")

    if not os.path.isfile(EXEC):
        print(f"Ошибка: файл '{EXEC}' не найден.", file=sys.stderr)
        sys.exit(1)

    init_results_file()

    successful_runs = 0

    for n in MATRIX_SIZES:
        print(f"\n[SIZE] {n}×{n}")

        if not os.path.isfile(f"A{n}.txt") or not os.path.isfile(f"B{n}.txt"):
            print(f"Файлы A{n}.txt или B{n}.txt не найдены, пропускаем размер")
            continue

        for threads in THREAD_COUNTS:
            print(f"  [THREADS] {threads}", end=" ... ", flush=True)

            metrics = run_program(n, threads)
            if not metrics:
                print(" - FAILED")
                continue

            if n <= VERIFY_UP_TO:
                verified = verify_results(n)
            else:
                verified = True
                print(" (верификация пропущена)", end="")

            save_result(n, threads, metrics, verified)
            successful_runs += 1

            print(f" + {metrics['time_ms']:.2f} мс, {metrics['gflops']:.2f} GFLOPS")

    print("\n=== Эксперименты завершены ===")
    print(f"Успешных запусков: {successful_runs}")
    print(f"Результаты сохранены в: {RESULTS_FILE}")

    print_summary()


if __name__ == "__main__":
    main()