"""
The purpose of this module is to assess the content of ./ESEM - Final.ods (look for duplicates, overlap, etc.)
"""
import os
from pyexcel_ods3 import get_data
import code
import pandas as pd

# constants
ods_filepath = os.path.abspath(os.path.join(__file__,'../','ESEM - Final.ods'))

# ask questions
print('filepath is',ods_filepath)
data = get_data(ods_filepath)
print('tabs are:',[key for key in data.keys()])
print('number of repos:')
dataframes = {}
for tab,sheet in list(data.items())[:-1]: # exclude the last tab as it's only the regular expressions
    headers = sheet[0]
    sheet_data = [row for row in sheet[1:] if len(row) != 0]
    d = {}
    for index,header in enumerate(headers):
        column_data = [row[index] for row in sheet_data]
        d[header] = column_data
    df = pd.DataFrame(data=d)
    print(df.shape[0],' repos in ',tab)
    dataframes[tab] = df
print()
# look into overlaps
repo_tabs = [tab for tab in data.keys() if tab != 'Regular expressions']
repo_tabs.sort()
print('overlaps are...')
overlaps = {}
for tab0 in repo_tabs:
    overlaps[tab0] = {}
    for tab1 in repo_tabs:
        s0 = set(dataframes[tab0]['Repository'].values)
        s1 = set(dataframes[tab1]['Repository'].values)
        intercept = s0.intersection(s1)
        overlaps[tab0][tab1] = len(intercept)
df = pd.DataFrame(data=overlaps)
print(df)

print()
print('Model stats')
recall_df = dataframes['Model']
uniques = {x for x in recall_df['Archived']}
print('vals',uniques)
for val in uniques:
    c = recall_df.loc[recall_df['Archived'] == val].count()
    print(val,c['Archived'])

print()
print('Recall stats')
recall_df = dataframes['Recall']
uniques = {x for x in recall_df['Classification']}
print('vals',uniques)
for val in uniques:
    c = recall_df.loc[recall_df['Classification'] == val].count()
    print(val,c['Classification'])

print()
print('Precision stats')
recall_df = dataframes['Precision']
uniques = {x for x in recall_df['Result']}
print('vals',uniques)
for val in uniques:
    c = recall_df.loc[recall_df['Result'] == val].count()
    print(val,c['Result'])
