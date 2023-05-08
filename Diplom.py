import pandas as pd
import time
import numpy as np
import statistics
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import *
from multiprocessing import Process
import multiprocessing
import os
import matplotlib.pyplot as plt

def slope(xs, ys):
    """Вычисление наклона линии (углового коэффициента)"""
    return xs.cov(ys) / xs.var()

def add_values_to_table(se, l, w, k):
    final_table = pd.DataFrame()

    final_table.insert(0, "EXP/CONTR", -1)
    final_table.insert(1, "NUM_OF_RAT", -1)
    final_table.insert(2, "AMPLITUDE", -1)
    final_table.insert(3, "COUNT_1000", -1)
    final_table.insert(4, "COUNT_3000", -1)
    final_table.insert(5, "TONUS", -1)
    final_table.insert(6, "PERCENTILE_0.995", -1)
    final_table.insert(7, "PERCENTILE_0.05", -1)
    final_table.insert(8, "PERCENTILE_0.85", -1)
    final_table.insert(9, "PARAMETR_1", -1)
    final_table.insert(10, "PARAMETR_2", -1)
    final_table.insert(11, "PARAMETR_3", -1)

    se["AVGT"] = se.iloc[:, 3].rolling(window=9, min_periods=9, center=True).mean()
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=3000)
    se["MI"] = se["AVGT"].rolling(window=indexer).min()
    se["MA"] = se["AVGT"].rolling(window=indexer).max()
    se["DIF"] = se["MA"] - se["MI"]
    se.insert(11, "SLOPE_1000", 2)
    se.insert(12, "COUNT_1000", 2)
    se.insert(13, "SLOPE_3000", 2)
    se.insert(14, "COUNT_3000", 2)

    for i in range(1, se.shape[0] - 1001):
        se.iloc[i, 11] = np.sign(
            slope(se.iloc[i : (i + 1000), 2], se.iloc[i : (i + 1000), 7])
        )
        if (se.iloc[i, 11] * se.iloc[i - 1, 11]) < 0:
            se.iloc[i, 12] = 1
        else:
            se.iloc[i, 12] = 0

    for i in range(1, se.shape[0] - 3001):
        se.iloc[i, 13] = np.sign(
            slope(se.iloc[i : (i + 3000), 2], se.iloc[i : (i + 3000), 7])
        )
        if (se.iloc[i, 13] * se.iloc[i - 1, 13]) < 0:
            se.iloc[i, 14] = 1
        else:
            se.iloc[i, 14] = 0

    mean = se["AVGT"]
    

    final_table.loc[len(final_table.index)] = [
        se.iloc[0, 0], # 0,0 -> exp/contr
        se.iloc[0, 1], # 0, 1 -> 6
        round(statistics.median(se.iloc[5 : (se.shape[0] - 3001), 10]), 2), # mediana po amplitudam, округляем до двух значений после запятой
        sum(se.iloc[5 : (se.shape[0] - 1001), 12]) / 2, # SIG1K sum / 2
        sum(se.iloc[5 : (se.shape[0] - 3001), 14]) / 2, # SIG3K sum / 2
        round(statistics.median(se.iloc[5 : (se.shape[0] - 3001), 8]), 2), # median po minimum
        round(np.percentile(se.iloc[4 : (se.shape[0] - 4), 7], 99.5), 2), # precentile po mean
        round(np.percentile(se.iloc[4 : (se.shape[0] - 4), 7], 5), 2),
        round(np.percentile(se.iloc[4 : (se.shape[0] - 4), 7], 85), 2),
        se.iloc[0, 4], # dist_before ach
        se.iloc[0, 5], # ...
        se.iloc[0, 6], # ...
    ]

    return [final_table, mean, l, w, k]

def file_calculation(file_path):
    #EXP = pd.read_csv(file_path, delimiter=";", encoding="latin-1")
    # Поменять на это для чтения эксель файлов
    EXP = pd.read_excel(file_path)
    EXP = EXP.dropna()

    Type_v = EXP.iloc[:, 4].unique()
    Time_v = EXP.iloc[:, 5].unique()
    Subst_v = EXP.iloc[:, 6].unique()

    for l in Type_v:
        print("l:" + str(l))
        for w in Time_v:
            print("w:" + str(w))
            for k in Subst_v:
                print("k:" + str(k))
                se = EXP[
                    (EXP.iloc[:, 4] == l)
                    & (EXP.iloc[:, 5] == w)
                    & (EXP.iloc[:, 6] == k)
                ]
                taskResults.append(pool.apply_async(add_values_to_table, args=[se, l, w, k]))


if __name__ == '__main__':
    cpuCoreCount = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpuCoreCount)

    taskResults = []

    root = tk.Tk()
    root.withdraw()

    # Open file dialog
    file_paths_in = filedialog.askopenfilenames(title='Select experiment files')
    dir_path_out = filedialog.askdirectory(title='Select folder for output files')
    file_name_out = "results.xlsx"
    graph_name = "graph"
    file_path_out = os.path.join(dir_path_out, file_name_out)

    start = time.time()
    for file_path_in in file_paths_in:
        file_calculation(file_path_in)
    pool.close()
    pool.join()
    end = time.time()
    print(f"Multiprocess execution took {end - start} seconds")

    tables = []

    counter = 0
    for result in taskResults:
        result = result.get()

        table = result[0]
        mean = result[1]
        l = str(result[2])
        w = str(result[3])
        k = str(result[4])

        fig, ax = plt.subplots()
        ax.plot(mean)
        ax.set_title(f'Smooth mean')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')

        graph_path = dir_path_out + "/" + l + '_' + w + '_' + k + ".png"

        fig.savefig(graph_path)

        tables.append(table)
        counter = counter + 1

    final_table = pd.concat(tables).reset_index(drop=True)

    final_table.to_excel(file_path_out, index=False)

