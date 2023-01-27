import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import csv

df_ais = pd.read_csv("ais_202102.csv",sep=';',engine='python')

print(df_ais.head())
df_ais_new=df_ais[["mmsi",'lat','lon']]

print(df_ais_new.head())
print(df_ais_new.tail())

df_ais_new.to_csv('ais_lat-lon.csv',index=False)



