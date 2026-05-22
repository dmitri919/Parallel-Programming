import numpy as np

def generate_matrix_file(filename, n, seed=None):
    """
    Генерирует файл с матрицей NxN.
    Формат: первая строка - размер N, далее N*N чисел с плавающей точкой.
    """
    if seed is not None:
        np.random.seed(seed)  # Для воспроизводимости результатов

    # Диапазоне [0, 100)
    matrix = np.random.uniform(0, 100, size=(n, n))

    with open(filename, "w") as f:
        f.write(f"{n}\n")
        # Записываем элементы с 8 знаками после запятой
        for row in matrix:
            f.write(" ".join(f"{val:.8f}" for val in row) + "\n")

    print(f"✅ Создан файл '{filename}' ({n}x{n}, {n*n} элементов)")

if __name__ == "__main__":
    N = 2000  # Размер матрицы

    print(f"Генерация матриц {N}x{N}...")
    generate_matrix_file("A.txt", N, seed=42)
    generate_matrix_file("B.txt", N, seed=123)
    print("Готово! Файлы A.txt и B.txt можно использовать для теста.")