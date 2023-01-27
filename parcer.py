import csv
import xml.etree.ElementTree as ET

import pandas as pd
import lxml
from lxml import etree
from lxml import objectify

tree = ET.parse("Cruise_Stavanger_Skudefjorden_Out_20210212.rtz")
root = tree.getroot()





# PANDA TABLE

df_cols = ['id', 'name', 'lat', 'lon', 'radius']
rows = []
df = pd.DataFrame(rows, columns=df_cols)

x = 0
for item in root:
    if item.tag == ('waypoint'):
        print(item.tag, item.attrib)
        s_id = item.attrib['id']
        s_name = item.attrib['name']
        if s_name == "":
            s_name = "None"
        s_radius=item.get('radius')
        if s_radius is not None:
            s_radius = item.attrib['radius']
            df.loc[x, 'radius'] = s_radius
        df.loc[x, 'id'] = s_id
        df.loc[x, 'name'] = s_name
        x = x + 1

x = 0
for position in root.iter("position"):
    print(position.tag, position.attrib)
    s_lat = position.attrib['lat']
    s_lon = position.attrib['lon']
    df.loc[x, 'lat'] = s_lat
    df.loc[x, 'lon'] = s_lon
    x = x + 1

df.set_index('id',inplace=True)
print(df)

df.to_csv('Cruise_Stavanger_Skudefjorden_Out_20210212.csv',index=True)