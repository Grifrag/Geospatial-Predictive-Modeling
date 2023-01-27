import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import csv

def append(df,df2):
    New = df.append(df2, ignore_index=True)
    New = New.loc[:, ['id', 'lat', 'lon','label']]
    return New



df_in = pd.read_csv('Cruise_Haugesund_Skudefjorden_In_20210212.csv')
df_in['label']='Hau-Skud'
df_out=pd.read_csv("Cruise_Haugesund_Skudefjorden_Out_20210212.csv")
df_out['label']='Hau-Skud'
New =append(df_in,df_out)

df_in = pd.read_csv('Cruise_Stavanger_Feistein_In_20210212.csv')
df_in['label']='Stav-Feis'
New =append(New,df_in)
df_out=pd.read_csv("Cruise_Stavanger_Feistein_Out_20210212.csv")
df_out['label']='Stav-Feis'
New =append(New,df_out)

df_in = pd.read_csv('Cruise_Stavanger_Skudefjorden_In_20210212.csv')
df_in['label']='Stav-Skud'
New =append(New,df_in)
df_out=pd.read_csv("Cruise_Stavanger_Skudefjorden_Out_20210212.csv")
df_out['label']='Stav-Skud'
New =append(New,df_out)

print(New)

#columns_titles=['id','lon','lat','label']
#New=New.reindex(columns=columns_titles)

New.to_csv('waypoints-withLabels.csv',index=False)