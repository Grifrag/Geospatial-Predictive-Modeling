import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import csv

def drop(df):
    df.drop('radius', axis=1, inplace=True)
    df.drop('name', axis=1, inplace=True)
    df.drop('id', axis=1, inplace=True)
    return df


def pivot(df):
    New = (df.pivot_table(index='label',
                          columns=df.groupby('label').cumcount(),
                          aggfunc='first')
           .sort_index(axis=1, level=1))
    New = New.set_axis([f'{x}{y}' if y != 0 else x
                        for x, y in New.columns],
                       axis=1).reset_index()
    return New
def append(df,df2):
    New = df.append(df2, ignore_index=True)

    New=New.fillna(method="ffill")
    #New = New.loc[:, ['id', 'lat', 'lon','label']]
    return New



df_in = pd.read_csv('Cruise_Haugesund_Skudefjorden_In_20210212.csv')
df_in = drop(df_in)
df_in['label']='Hau-Skud'
df_in=pivot(df_in)
df_out=pd.read_csv("Cruise_Haugesund_Skudefjorden_Out_20210212.csv")
df_out = drop(df_out)
df_out['label']='Hau-Skud'
df_out=pivot(df_out)
New =append(df_in,df_out)

df_in = pd.read_csv('Cruise_Stavanger_Feistein_In_20210212.csv')
df_in = drop(df_in)
df_in['label']='Stav-Feis'
df_in=pivot(df_in)
New =append(New,df_in)
df_out=pd.read_csv("Cruise_Stavanger_Feistein_Out_20210212.csv")
df_out = drop(df_out)
df_out['label']='Stav-Feis'
df_out=pivot(df_out)
New =append(New,df_out)

df_in = pd.read_csv('Cruise_Stavanger_Skudefjorden_In_20210212.csv')
df_in = drop(df_in)
df_in['label']='Stav-Skude'
df_in=pivot(df_in)
New =append(New,df_in)
df_out=pd.read_csv("Cruise_Stavanger_Skudefjorden_Out_20210212.csv")
df_out = drop(df_out)
df_out['label']='Stav-Skude'
df_out=pivot(df_out)
New =append(New,df_out)


print(New)

#columns_titles=['id','lon','lat','label']
#New=New.reindex(columns=columns_titles)

New.to_csv('waypoints-withLabelsVol2.csv',index=False)