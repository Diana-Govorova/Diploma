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


def arg_calculation(temp):

    Type_exp_1 = temp.iloc[:, 0].unique()
    Type_v_1 = temp.iloc[:, 9].unique()
    Time_v_1 = temp.iloc[:, 10].unique()
    Subst_v_1 = temp.iloc[:, 11].unique()

    agr_final_table = pd.DataFrame()

    agr_final_table.insert(0, "PARAM", -1)
    agr_final_table.insert(1, "AMPLITUDE_BEFORE", -1)
    agr_final_table.insert(2, "AMPLITUDE_AFTER", -1)
    agr_final_table.insert(3, "COUNT_1000_BEFORE", -1)
    agr_final_table.insert(4, "COUNT_1000_AFTER", -1)
    agr_final_table.insert(5, "COUNT_3000_BEFORE", -1)
    agr_final_table.insert(6, "COUNT_3000_AFTER", -1)
    agr_final_table.insert(7, "TONUS_BEFORE", -1)
    agr_final_table.insert(8, "TONUS_AFTER", -1)
    agr_final_table.insert(9, "PERCENTILE_0.995_BEFORE", -1)
    agr_final_table.insert(10, "PERCENTILE_0.995_AFTER", -1)
    agr_final_table.insert(11, "PERCENTILE_0.05_BEFORE", -1)
    agr_final_table.insert(12, "PERCENTILE_0.05_AFTER", -1)
    agr_final_table.insert(13, "PERCENTILE_0.85_BEFORE", -1)
    agr_final_table.insert(14, "PERCENTILE_0.85_AFTER", -1)

   

    for k in Type_exp_1:
        for l in Type_v_1:
            for s in Subst_v_1:
                temp1 = temp[(temp.iloc[:, 0] == k)
                            & (temp.iloc[:, 9] == l)
                            & (temp.iloc[:, 10] == 'BEFORE')
                            & (temp.iloc[:, 11] == s)
                            ] 
                
                temp2 = temp[(temp.iloc[:, 0] == k)
                            & (temp.iloc[:, 9] == l)
                            & (temp.iloc[:, 10] == 'AFTER')
                            & (temp.iloc[:, 11] == s)
                            ] 

                if temp1.empty: 
                    continue
        
                agr_final_table.loc[len(agr_final_table.index)] = [
                k + '_' + l + '_' + s,   

                round(statistics.median(temp1.iloc[:, 2]), 2),
                round(statistics.median(temp2.iloc[:, 2]), 2),
                round(statistics.median(temp1.iloc[:, 3]), 2),
                round(statistics.median(temp2.iloc[:, 3]), 2),
                round(statistics.median(temp1.iloc[:, 4]), 2),
                round(statistics.median(temp2.iloc[:, 4]), 2),
                round(statistics.median(temp1.iloc[:, 5]), 2),
                round(statistics.median(temp2.iloc[:, 5]), 2),
                round(statistics.median(temp1.iloc[:, 6]), 2),
                round(statistics.median(temp2.iloc[:, 6]), 2),
                round(statistics.median(temp1.iloc[:, 7]), 2),
                round(statistics.median(temp2.iloc[:, 7]), 2),
                round(statistics.median(temp1.iloc[:, 8]), 2),
                round(statistics.median(temp2.iloc[:, 8]), 2)
                ]

    
    return agr_final_table


def index_calculation(agr_final_table):

    index_final_table = pd.DataFrame()

    index_final_table.insert(0, "PARAM", -1)
    index_final_table.insert(1, "AMPLITUDE", -1)
    index_final_table.insert(2, "COUNT_1000", -1)
    index_final_table.insert(3, "COUNT_3000", -1)
    index_final_table.insert(4, "TONUS", -1)
    index_final_table.insert(5, "PERCENTILE_0.995", -1)
    index_final_table.insert(6, "PERCENTILE_0.05", -1)
    index_final_table.insert(7, "PERCENTILE_0.85", -1)



    for i in range(len(agr_final_table.index)):
        index_final_table.loc[len(index_final_table.index)] = [
                    agr_final_table.iloc[i, 0],
                    round((agr_final_table.iloc[i, 2] / agr_final_table.iloc[i, 1]), 2),
                    round((agr_final_table.iloc[i, 4] / agr_final_table.iloc[i, 3]), 2),
                    round((agr_final_table.iloc[i, 6] / agr_final_table.iloc[i, 5]), 2),
                    round((agr_final_table.iloc[i, 8] / agr_final_table.iloc[i, 7]), 2),
                    round((agr_final_table.iloc[i, 10] / agr_final_table.iloc[i, 9]), 2),
                    round((agr_final_table.iloc[i, 12] / agr_final_table.iloc[i, 11]), 2),
                    round((agr_final_table.iloc[i, 14] / agr_final_table.iloc[i, 13]), 2),
         
                    ]

    return index_final_table


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
    file_name_out_arg = "Agr_table.xlsx"
    file_name_out_ind = "Index_table.xlsx"


    file_name_out_amp = "AVG_AMPLITUDE.png"
    file_name_out_s_count = "AVG_COUNT_1000.png"
    file_name_out_b_count = "AVG_COUNT_3000.png"
    file_name_out_tonus = "AVG_TONUS.png"
    file_name_out_pers_0995 = "AVG_PERCENTILE_0.995.png"
    file_name_out_pers_005 = "AVG_PERCENTILE_0.05.png"
    file_name_out_pers_085 = "AVG_PERCENTILE_0.85.png"

    file_name_out_amp1 = "INDEX_AMPLITUDE.png"
    file_name_out_s_count1 = "INDEX_COUNT_1000.png"
    file_name_out_b_count1 = "INDEX_COUNT_3000.png"
    file_name_out_tonus1 = "INDEX_TONUS.png"
    file_name_out_pers_0995_1 = "INDEX_PERCENTILE_0.995.png"
    file_name_out_pers_005_1 = "INDEX_PERCENTILE_0.05.png"
    file_name_out_pers_085_1 = "INDEX_PERCENTILE_0.85.png"

    graph_name = "graph"

    file_path_out = os.path.join(dir_path_out, file_name_out)
    file_path_out_arg = os.path.join(dir_path_out, file_name_out_arg)
    file_path_out_ind = os.path.join(dir_path_out, file_name_out_ind)

    file_path_out_amp = os.path.join(dir_path_out, file_name_out_amp)
    file_path_out_s_count = os.path.join(dir_path_out, file_name_out_s_count)
    file_path_out_b_count = os.path.join(dir_path_out, file_name_out_b_count)
    file_path_out_tonus = os.path.join(dir_path_out, file_name_out_tonus)
    file_path_out_pers_0995 = os.path.join(dir_path_out, file_name_out_pers_0995)
    file_path_out_pers_005 = os.path.join(dir_path_out, file_name_out_pers_005)
    file_path_out_pers_085 = os.path.join(dir_path_out, file_name_out_pers_085)

    file_path_out_amp1 = os.path.join(dir_path_out, file_name_out_amp1)
    file_path_out_s_count1 = os.path.join(dir_path_out, file_name_out_s_count1)
    file_path_out_b_count1 = os.path.join(dir_path_out, file_name_out_b_count1)
    file_path_out_tonus1 = os.path.join(dir_path_out, file_name_out_tonus1)
    file_path_out_pers_0995_1 = os.path.join(dir_path_out, file_name_out_pers_0995_1)
    file_path_out_pers_005_1 = os.path.join(dir_path_out, file_name_out_pers_005_1)
    file_path_out_pers_085_1 = os.path.join(dir_path_out, file_name_out_pers_085_1)

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
    
    arg_final_table = arg_calculation(final_table)

                                                                                                                                                                                                      

    arg_final_table[['PARAM', 'AMPLITUDE_BEFORE', 'AMPLITUDE_AFTER']].plot(x='PARAM', kind='bar', title = 'Усредненное значение амплитуды до/после эксперимента для каждой группы', rot=15).get_figure().savefig(file_path_out_amp)
    arg_final_table[['PARAM', 'COUNT_1000_BEFORE', 'COUNT_1000_AFTER']].plot(x='PARAM', kind='bar', title = 'Усредненное значение малой частоты до/после эксперимента для каждой группы', rot=15).get_figure().savefig(file_path_out_s_count)
    arg_final_table[['PARAM', 'COUNT_3000_BEFORE', 'COUNT_3000_AFTER']].plot(x='PARAM', kind='bar', title = 'Усредненное значение большой частоты до/после эксперимента для каждой группы', rot=15).get_figure().savefig(file_path_out_b_count)
    arg_final_table[['PARAM', 'TONUS_BEFORE', 'TONUS_AFTER']].plot(x='PARAM', kind='bar', title = 'Усредненное значение тонуса до/после эксперимента для каждой группы', rot=15).get_figure().savefig(file_path_out_tonus)
    arg_final_table[['PARAM', 'PERCENTILE_0.995_BEFORE', 'PERCENTILE_0.995_AFTER']].plot(x='PARAM', kind='bar', title = 'Усредненное значение 0,995 персентили до/после эксперимента для каждой группы', rot=15).get_figure().savefig(file_path_out_pers_0995)
    arg_final_table[['PARAM', 'PERCENTILE_0.05_BEFORE', 'PERCENTILE_0.05_AFTER']].plot(x='PARAM', kind='bar', title = 'Усредненное значение 0,05 персентили до/после эксперимента для каждой группы', rot=15).get_figure().savefig(file_path_out_pers_005)
    arg_final_table[['PARAM', 'PERCENTILE_0.85_BEFORE', 'PERCENTILE_0.85_AFTER']].plot(x='PARAM', kind='bar', title = 'Усредненное значение 0,85 персентили до/после эксперимента для каждой группы', rot=15).get_figure().savefig(file_path_out_pers_085)
    plt.close('all')

    index_final_table = index_calculation(arg_final_table)

    fig, ax = plt.subplots()
    plt.title("Индекс изменения амплитуды до/после эксперимента для каждой группы") 
    ax.bar(index_final_table['PARAM'], index_final_table['AMPLITUDE'])
    fig.savefig(file_path_out_amp1)
    plt.close()

    fig, ax = plt.subplots()
    plt.title("Индекс изменения малой частоты до/после эксперимента для каждой группы") 
    ax.bar(index_final_table['PARAM'], index_final_table['COUNT_1000'])
    fig.savefig(file_path_out_s_count1)
    plt.close()

    fig, ax = plt.subplots()
    plt.title("Индекс изменения большой частоты до/после эксперимента для каждой группы") 
    ax.bar(index_final_table['PARAM'], index_final_table['COUNT_3000'])
    fig.savefig(file_path_out_b_count1)
    plt.close()

    fig, ax = plt.subplots()
    plt.title("Индекс изменения тонуса до/после эксперимента для каждой группы") 
    ax.bar(index_final_table['PARAM'], index_final_table['TONUS'])
    fig.savefig(file_path_out_tonus1)
    plt.close()

    fig, ax = plt.subplots()
    plt.title("Индекс изменения 0.995 персентили до/после эксперимента для каждой группы") 
    ax.bar(index_final_table['PARAM'], index_final_table['PERCENTILE_0.995'])
    fig.savefig(file_path_out_pers_0995_1) 
    plt.close()

    fig, ax = plt.subplots()
    plt.title("Индекс изменения 0.05 персентили до/после эксперимента для каждой группы") 
    ax.bar(index_final_table['PARAM'], index_final_table['PERCENTILE_0.05'])
    fig.savefig(file_path_out_pers_005_1)
    plt.close()

    fig, ax = plt.subplots()
    plt.title("Индекс изменения 0.85 персентили до/после эксперимента для каждой группы") 
    ax.bar(index_final_table['PARAM'], index_final_table['PERCENTILE_0.85'])
    fig.savefig(file_path_out_pers_085_1) 
    plt.close()

   
    final_table.to_excel(file_path_out, index=False)
    arg_final_table.to_excel(file_path_out_arg, index=False)
    index_final_table.to_excel(file_path_out_ind, index=False)
