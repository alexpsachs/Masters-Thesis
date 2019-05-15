"""
The purpose of this script is to create the list of repositories to crawl for the final testing set which are:
    112 repos from /home/alex/Thesis/Experiment/Analysis/ESEM - Dataset.ods
    129 repos from /home/alex/Thesis/Experiment/Analysis/ESEM - Survey - dataset.csv
"""
import os
from pyexcel_ods3 import get_data
import pandas as pd
import code

ods_filepath = os.path.abspath(os.path.join(__file__,'../../Analysis/ESEM - Dataset.ods'))
csv_filepath = os.path.abspath(os.path.join(__file__,'../../Analysis/ESEM - Survey - dataset.csv'))
out_filepath = os.path.abspath(os.path.join(__file__,'../../Data/repo_list.txt'))

# get the data
ods_data = get_data(ods_filepath)['Recall']
csv_df = pd.read_csv(csv_filepath)

# pull out the repo names
header = ods_data[0]
index = header.index('Repository')
print('index',index)
repos = []
for line in [ln for ln in ods_data[1:] if len(ln) != 0]:
    repos.append(line[index])
for val in csv_df['Repository'].values:
    repos.append(val)
print('Total repos:',len(repos))
unique_repos = list(set(repos))
print('unique repos:',len(unique_repos))

# create the list
with open(out_filepath, 'w') as f:
    for repo in unique_repos:
        f.write(repo+'\n')

print(out_filepath, ' created with ',len(unique_repos),' repositories')
