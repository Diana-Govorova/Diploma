import pandas as pd
import time
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
import statistics
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import *
from multiprocessing import Process
import time

root = tk.Tk()
root.title("Data")
root.geometry("1000x500") 
#root.withdraw()

file_path = filedialog.askopenfilename()

def slope(xs, ys):
    '''Вычисление наклона линии (углового коэффициента)'''
    return xs.cov(ys) / xs.var()

final_table = pd.DataFrame() 
final_table.insert(0, "EXP/CONTR", -1)
final_table.insert(1, "NUM_OF_RAT",-1)
final_table.insert(2, "AMPLITUDE", -1)
final_table.insert(3, "COUNT_1000",-1)
final_table.insert(4, "COUNT_3000", -1)
final_table.insert(5, "TONUS", -1)
final_table.insert(6, "PERCENTILE_0.995", -1)
final_table.insert(7, "PERCENTILE_0.05", -1)
final_table.insert(8, "PERCENTILE_0.85", -1)
final_table.insert(9, "PARAMETR_1", -1)
final_table.insert(10, "PARAMETR_2", -1)
final_table.insert(11, "PARAMETR_3", -1)


def add_values_to_table(se, final_table):
    se['AVGT'] = se.iloc[:, 3].rolling(window=9, min_periods=9, center=True).mean()
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=3000)
    se['MI'] = se['AVGT'].rolling(window=indexer).min()
    se['MA'] = se['AVGT'].rolling(window=indexer).max()
    se['DIF'] = se['MA']-se['MI']
    se.insert(11, "SLOPE_1000", 2)
    se.insert(12, "COUNT_1000", 2)
    se.insert(13, "SLOPE_3000", 2)
    se.insert(14, "COUNT_3000", 2)
            
    start = time.time()

    for i in range(1, se.shape[0] - 1001):
        se.iloc[i, 11] = np.sign(slope(se.iloc[i:(i+1000), 2], se.iloc[i:(i+1000), 7]))
        if (se.iloc[i, 11] *  se.iloc[i - 1, 11]) < 0:
            se.iloc[i, 12] = 1
        else:
            se.iloc[i, 12] = 0
    end = time.time()
    print(end - start)


    start = time.time()
    for i in range(1, se.shape[0] - 3001):
        se.iloc[i, 13] = np.sign(slope(se.iloc[i:(i+3000), 2], se.iloc[i:(i+3000), 7]))
        if (se.iloc[i, 13] *  se.iloc[i - 1, 13]) < 0:
            se.iloc[i, 14] = 1
        else:
            se.iloc[i, 14] = 0 
    end = time.time()
    print(end - start)
 
    final_table.loc[len(final_table.index)] = [se.iloc[0, 0], se.iloc[0, 1], 
                                           round(statistics.median(se.iloc[5:(se.shape[0]-3001), 10]), 2), 
                                           sum(se.iloc[5:(se.shape[0]-1001), 12]) / 2,
                                           sum(se.iloc[5:(se.shape[0]-3001), 14]) / 2, 
                                           round(statistics.median(se.iloc[5:(se.shape[0]-3001), 8]), 2), 
                                           round(np.percentile(se.iloc[4:(se.shape[0]-4), 7], 99.5), 2), 
                                           round(np.percentile(se.iloc[4:(se.shape[0]-4), 7], 5), 2),
                                           round(np.percentile(se.iloc[4:(se.shape[0]-4), 7], 85), 2), se.iloc[0, 4],
                                           se.iloc[0, 5], se.iloc[0, 6] ]

    
def file_calculation(file_path):

    EXP = pd.read_csv(file_path, delimiter=';', encoding='latin-1')
    EXP = EXP.dropna()

    Type_v = EXP.iloc[:, 4].unique()
    Time_v = EXP.iloc[:, 5].unique()
    Subst_v = EXP.iloc[:, 6].unique()

    r = 0
    for l in Type_v:
        print("l:" + str(l))
        for w in Time_v:
            print("w:" + str(w))
            for k in Subst_v:
                print("k:" + str(k))
                r = r + 1
                se = EXP[(EXP.iloc[:, 4] == l) & (EXP.iloc[:, 5] == w) & (EXP.iloc[:, 6] == k)]
                add_values_to_table(se, final_table)
                
    print(final_table)

final_table =  file_calculation(file_path)


#se = EXP2[(EXP2['TYPE_VALUE'] == Type_v[0]) & (EXP2['TIME_VALUE(BEFORE/AFTER)'] == Time_v[0]) & (EXP2['SUBSTANCE_VALUE (BUT/ACH)'] == Subst_v[0])]
#add_values_to_table(se, final_table)

file_path_save = filedialog.asksaveasfilename()
final_table.to_csv(file_path_save, sep=',', index= False )



list_of_res = final_table.values.tolist()


 
# определяем столбцы
columns = ("EXP/CONTR", "NUM_OF_RAT", "AMPLITUDE", "COUNT_1000", "COUNT_3000", "TONUS", "PERCENTILE_0.995", "PERCENTILE_0.05", "PERCENTILE_0.85", "PARAMETR_1", "PARAMETR_2", "PARAMETR_3")


tree = ttk.Treeview(columns=columns, show="headings")
tree.pack(fill=BOTH, expand=1)
 
# определяем заголовки
tree.heading("EXP/CONTR", text="EXP/CONTR")
tree.heading("NUM_OF_RAT", text="NUM_OF_RAT")
tree.heading("AMPLITUDE", text="AMPLITUDE")
tree.heading("COUNT_1000", text="COUNT_1000")
tree.heading("COUNT_3000", text="COUNT_3000")
tree.heading("TONUS", text="TONUS")
tree.heading("PERCENTILE_0.995", text="PERCENTILE_0.995")
tree.heading("PERCENTILE_0.05", text="PERCENTILE_0.05")
tree.heading("PERCENTILE_0.85", text="PERCENTILE_0.85")
tree.heading("PARAMETR_1", text="PARAMETR_1")
tree.heading("PARAMETR_2", text="PARAMETR_2")
tree.heading("PARAMETR_3", text="PARAMETR_3")

tree.column("#1", stretch=YES, width=70)
tree.column("#2", stretch=YES, width=70)
tree.column("#3", stretch=YES, width=70)
tree.column("#4", stretch=YES, width=70)
tree.column("#5", stretch=YES, width=70)
tree.column("#6", stretch=YES, width=70)
tree.column("#7", stretch=YES, width=70)
tree.column("#8", stretch=YES, width=70)
tree.column("#9", stretch=YES, width=70)
tree.column("#10", stretch=YES, width=70)
tree.column("#11", stretch=YES, width=70)
tree.column("#12", stretch=YES, width=70)

 
# добавляем данные
for i in list_of_res:
    print(i)
    tree.insert("", END, values=i)


root.mainloop()


