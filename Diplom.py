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

def add_values_to_table(se, t, n, l, w, k):
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

    return [final_table, mean, t, n, l, w, k]

def file_calculation(file_path):
    #EXP = pd.read_csv(file_path, delimiter=";", encoding="latin-1")
    # Поменять на это для чтения эксель файлов
    EXP = pd.read_excel(file_path)
    EXP = EXP.dropna()

    Type_exp = EXP.iloc[:, 0].unique()
    Num_rat = EXP.iloc[:, 1].unique()
    Type_v = EXP.iloc[:, 4].unique()
    Time_v = EXP.iloc[:, 5].unique()
    Subst_v = EXP.iloc[:, 6].unique()

    counter_ = 0
    for t in Type_exp:
        print("t:" + str(t))
        for n in Num_rat:
            print("n:" + str(n))
            for l in Type_v:
                print("l:" + str(l))
                for w in Time_v:
                    print("w:" + str(w))
                    for k in Subst_v:
                        print("k:" + str(k))
                        se = EXP[(EXP.iloc[:, 0] == t)
                            & (EXP.iloc[:, 1] == n)
                            & (EXP.iloc[:, 4] == l)
                            & (EXP.iloc[:, 5] == w)
                            & (EXP.iloc[:, 6] == k)
                        ]
                       
                        if se.empty: 
                            continue
                       
                        taskResults.append(pool.apply_async(add_values_to_table, args=[se, t, n, l, w, k]))
                        counter_ = counter_ + 1


def index_calculation(temp):
    #temp.iloc[:, 2] = [x.replace(',', '.') for x in temp.iloc[:, 2]]
    #temp.iloc[:, 3] = [x.replace(',', '.') for x in temp.iloc[:, 3]]
    #temp.iloc[:, 4] = [x.replace(',', '.') for x in temp.iloc[:, 4]]
    #temp.iloc[:, 5] = [x.replace(',', '.') for x in temp.iloc[:, 5]]
    #temp.iloc[:, 6] = [x.replace(',', '.') for x in temp.iloc[:, 6]]
    #temp.iloc[:, 7] = [x.replace(',', '.') for x in temp.iloc[:, 7]]
    #temp.iloc[:, 8] = [x.replace(',', '.') for x in temp.iloc[:, 8]]


    #temp.iloc[:, 2]  = temp.iloc[:, 2].astype(float)
    #temp.iloc[:, 3]  = temp.iloc[:, 3].astype(float)
    #temp.iloc[:, 4]  = temp.iloc[:, 4].astype(float)
    #temp.iloc[:, 5]  = temp.iloc[:, 5].astype(float)
    #temp.iloc[:, 6]  = temp.iloc[:, 6].astype(float)
    #temp.iloc[:, 7]  = temp.iloc[:, 7].astype(float)
    #temp.iloc[:, 8]  = temp.iloc[:, 8].astype(float)

    Type_exp_1 = temp.iloc[:, 0].unique()
    Type_v_1 = temp.iloc[:, 9].unique()
    Time_v_1 = temp.iloc[:, 10].unique()
    Subst_v_1 = temp.iloc[:, 11].unique()

    agr_final_table = pd.DataFrame()

    agr_final_table.insert(0, "EXP/CONTR", -1)
    agr_final_table.insert(1, "TYPE_VALUE", -1)
    agr_final_table.insert(2, "TIME_VALUE", -1)
    agr_final_table.insert(3, "SUBST_VALUE", -1)
    agr_final_table.insert(4, "AMPLITUDE", -1)
    agr_final_table.insert(5, "COUNT_1000", -1)
    agr_final_table.insert(6, "COUNT_3000", -1)
    agr_final_table.insert(7, "TONUS", -1)
    agr_final_table.insert(8, "PERCENTILE_0.995", -1)
    agr_final_table.insert(9, "PERCENTILE_0.05", -1)
    agr_final_table.insert(10, "PERCENTILE_0.85", -1)

    for k in Type_exp_1:
        for l in Type_v_1:
            for s in Subst_v_1:
                 for e in Time_v_1:
                    temp1 = temp[(temp.iloc[:, 0] == k)
                                & (temp.iloc[:, 9] == l)
                                & (temp.iloc[:, 10] == e)
                                & (temp.iloc[:, 11] == s)
                              ]  
                    if temp1.empty: 
                        continue
                    agr_final_table.loc[len(agr_final_table.index)] = [
                    k, l, e, s,   

                    round(statistics.median(temp1.iloc[:, 2]), 2),
                    round(statistics.median(temp1.iloc[:, 3]), 2),
                    round(statistics.median(temp1.iloc[:, 4]), 2),
                    round(statistics.median(temp1.iloc[:, 5]), 2),
                    round(statistics.median(temp1.iloc[:, 6]), 2),
                    round(statistics.median(temp1.iloc[:, 7]), 2),
                    round(statistics.median(temp1.iloc[:, 8]), 2)
                    ]

    for i in range(len(agr_final_table.index) - 1):
        i = i + 1
        if ((agr_final_table.iloc[i, 0] == agr_final_table.iloc[i - 1, 0]) & 
           (agr_final_table.iloc[i, 1] == agr_final_table.iloc[i - 1, 1]) &
            (agr_final_table.iloc[i, 3] == agr_final_table.iloc[i - 1, 3])):
         
            agr_final_table.loc[len(agr_final_table.index)] = [
                    agr_final_table.iloc[i, 0],
                    agr_final_table.iloc[i, 1],
                    ' ',
                    agr_final_table.iloc[i, 3],
                    round((agr_final_table.iloc[i, 4] / agr_final_table.iloc[i - 1, 4]), 2),
                    round((agr_final_table.iloc[i, 5] / agr_final_table.iloc[i - 1, 5]), 2),
                    round((agr_final_table.iloc[i, 6] / agr_final_table.iloc[i - 1, 6]), 2),
                    round((agr_final_table.iloc[i, 7] / agr_final_table.iloc[i - 1, 7]), 2),
                    round((agr_final_table.iloc[i, 8] / agr_final_table.iloc[i - 1, 8]), 2),
                    round((agr_final_table.iloc[i, 9] / agr_final_table.iloc[i - 1, 9]), 2),
                    round((agr_final_table.iloc[i, 10] / agr_final_table.iloc[i - 1,10]), 2)]
        else:
            continue
   
    return agr_final_table





if __name__ == '__main__':
    cpuCoreCount = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpuCoreCount)

    taskResults = []

    root = tk.Tk()
    root.withdraw()

    # Open file dialog
    file_paths_in = filedialog.askopenfilenames(title='Select experiment files')
    dir_path_out = filedialog.askdirectory(title='Select folder for output files')
    file_name_out = "Summary_table.xlsx"
    file_name_out_arg = "Index_table.xlsx"
    graph_name = "graph"
    file_path_out = os.path.join(dir_path_out, file_name_out)
    file_path_out_arg = os.path.join(dir_path_out, file_name_out_arg)

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
        t = str(result[2])
        n = str(result[3])
        l = str(result[4])
        w = str(result[5])
        k = str(result[6])

        fig, ax = plt.subplots()
        ax.plot(mean)
        ax.set_title(f'Smooth mean')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')

        graph_path = dir_path_out + "/" + t + '_' + n + '_' + l + '_' + w + '_' + k + ".png"

        fig.savefig(graph_path)

        tables.append(table)
        counter = counter + 1

    final_table = pd.concat(tables).reset_index(drop=True)
    
    arg_final_table = index_calculation(final_table)

    final_table.to_excel(file_path_out, index=False)
    arg_final_table.to_excel(file_path_out_arg, index=False)

