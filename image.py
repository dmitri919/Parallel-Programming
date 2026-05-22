#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Построение графиков для MPI-версии:
1. Time vs Matrix Size
2. Time vs Computational Volume (FLOPs)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Настройки
CSV_FILE = "results_table.csv"
OUTPUT_DIR = "image"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Читаем данные
df = pd.read_csv(CSV_FILE)

# Вычисляем вычислительный объём: FLOPs = 2*N³ - N²
df['Volume_FLOPs'] = 2 * df['Matrix_Size']**3 - df['Matrix_Size']**2
df['Volume_GFLOPs'] = df['Volume_FLOPs'] / 1e9  # для удобной подписи оси

# Словарь стилей для процессов MPI
PROC_STYLES = {
    1:  {'color': '#e41a1c', 'marker': 'o', 'label': '1 proc'},
    2:  {'color': '#377eb8', 'marker': 's', 'label': '2 procs'},
    4:  {'color': '#4daf4a', 'marker': '^', 'label': '4 procs'},
    6:  {'color': '#984ea3', 'marker': 'd', 'label': '6 procs'},
    8:  {'color': '#ff7f00', 'marker': 'v', 'label': '8 procs'},
    12: {'color': '#a65628', 'marker': '*', 'label': '12 procs'},
}

# График 1: Time vs Matrix Size
plt.figure(figsize=(10, 6))

# используем столбец 'Processes' или 'Threads'
for procs in sorted(df['Processes'].unique()):
    subset = df[df['Processes'] == procs]
    style = PROC_STYLES.get(procs, {'color': 'gray', 'marker': 'x', 'label': f'{procs} procs'})

    plt.plot(
        subset['Matrix_Size'],
        subset['Time_ms'],
        marker=style['marker'],
        color=style['color'],
        label=style['label'],
        linewidth=2,
        markersize=6
    )

plt.xlabel('Размер матрицы (N × N)', fontsize=12)
plt.ylabel('Время выполнения (мс)', fontsize=12)
plt.title('Зависимость времени выполнения от размера матрицы (MPI)', fontsize=14, fontweight='bold')
plt.legend(title='Число процессов', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

path1 = os.path.join(OUTPUT_DIR, "time_vs_size.png")
plt.savefig(path1, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Сохранён график: {path1}")

# График 2: Time vs Computational Volume (FLOPs)
plt.figure(figsize=(10, 6))

for procs in sorted(df['Processes'].unique()):
    subset = df[df['Processes'] == procs]
    style = PROC_STYLES.get(procs, {'color': 'gray', 'marker': 'x', 'label': f'{procs} procs'})

    plt.plot(
        subset['Volume_GFLOPs'],
        subset['Time_ms'],
        marker=style['marker'],
        color=style['color'],
        label=style['label'],
        linewidth=2,
        markersize=6
    )

plt.xlabel('Вычислительный объём (GFLOPs)', fontsize=12)
plt.ylabel('Время выполнения (мс)', fontsize=12)
plt.title('Зависимость времени выполнения от вычислительного объёма (MPI)', fontsize=14, fontweight='bold')
plt.legend(title='Число процессов', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

path2 = os.path.join(OUTPUT_DIR, "time_vs_volume.png")
plt.savefig(path2, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Сохранён график: {path2}")

print(f"\nГрафики сохранены в папку '{OUTPUT_DIR}/'")