import subprocess
import numpy as np
import sys

EXEC = "matmul.exe"

def run_program():
    print(f"Запуск {EXEC}")
    result = subprocess.run([EXEC], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Ошибка выполнения:\n{result.stderr}")
        sys.exit(1)

def read_matrix_from_file(filepath):
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f.readlines()
                 if line.strip() and not line.strip().startswith('#')]

    if not lines:
        raise ValueError(f"Файл {filepath} пуст или не содержит данных.")
    data = []
    for line in lines:
        data.extend(line.split())
    if not data:
        raise ValueError(f"Не найдено числовых данных в {filepath}.")
    n = int(data[0])
    if len(data) != n * n + 1:
        raise ValueError(f"Неверное количество элементов в {filepath}. Ождалось {n*n+1}, найдено {len(data)}.")

    values = np.array(data[1:], dtype=np.float64)
    return values.reshape(n, n)

def verify_results():
    print("Верификация.")
    try:
        A = read_matrix_from_file("A.txt")
        B = read_matrix_from_file("B.txt")
        C_cpp = read_matrix_from_file("result.txt")
        C_ref = np.dot(A, B)
        is_match = np.allclose(C_cpp, C_ref, atol=1e-6, rtol=1e-6)

        status_line = "VERIFICATION: PASSED" if is_match else "VERIFICATION: FAILED"
        with open("result.txt", "a") as f:
            f.write(f"# {status_line}\n")

        if is_match:
            max_error = np.max(np.abs(C_cpp - C_ref))
            print("+ Верификация пройдена успешно.")
            print(f"   Максимальная абсолютная погрешность: {max_error:.2e}")
        else:
            max_error = np.max(np.abs(C_cpp - C_ref))
            print("- Верификация НЕ пройдена.")
            print(f"   Максимальная абсолютная погрешность: {max_error:.2e}")
            sys.exit(1)
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)

def main():
    run_program()
    verify_results()

if __name__ == "__main__":
    main()