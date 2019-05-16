"""
The purpose of this script is to create the list of repositories to crawl for the final testing set which are:
    112 repos from /home/alex/Thesis/Experiment/Analysis/ESEM - Dataset.ods
    129 repos from /home/alex/Thesis/Experiment/Analysis/ESEM - Survey - dataset.csv
"""
import os
from pyexcel_ods3 import get_data
import pandas as pd
import code
import sys


def main():
    ods_filepath = os.path.abspath(os.path.join(__file__,'../../Analysis/ESEM - Dataset.ods'))
    csv_filepath = os.path.abspath(os.path.join(__file__,'../../Analysis/ESEM - Survey - dataset.csv'))
    out_filepath = os.path.abspath(os.path.join(__file__,'../../Data/repo_list.txt'))

    # get the data
    ods_data = get_data(ods_filepath)['Recall']
    ods_dict = {label:[row[ods_data[0].index(label)] for row in ods_data[1:] if len(row) != 0] for label in ods_data[0]}
    # create dataframes
    ods_df = pd.DataFrame(data=ods_dict)
    csv_df = pd.read_csv(csv_filepath)
    csv_df = csv_df.rename(columns={'Result':'Classification'})

    # check the data
    print('what are the lengths of each?')
    print('\nods\n',ods_df.describe())
    print('\ncsv\n',csv_df.describe())

    print('\nwhat are the common repositories?')
    # common_repos = list(set(ods_df['Repository'].to_list() + csv_df['Repository'].to_list()))
    common_repos = [repo for repo in ods_df['Repository'].to_list() if repo in csv_df['Repository'].to_list()]
    print(len(common_repos),'common repos')

    print('\ndo the common repos have the same classification?')
    agree = []
    disagree = [] 
    for repo in common_repos:
        ods_val = ods_df[ods_df['Repository'] == repo]['Classification'].values[0]
        csv_val = csv_df[csv_df['Repository'] == repo]['Classification'].values[0]
        if ods_val == csv_val:
            agree.append(repo)
        else:
            disagree.append(repo)
    print('repo classifications that agree',len(agree))
    print('repo classifications that disagree',len(disagree))
    print('disagree:',disagree)

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
    out_dict = {'Repository':[],'Classification':[]}
    for repo in unique_repos:
        out_dict['Repository'].append(repo)
        if repo in ods_df['Repository'].to_list():
            out_dict['Classification'].append(ods_df[ods_df['Repository'] == repo]['Classification'].values[0])
        else:
            out_dict['Classification'].append(csv_df[csv_df['Repository'] == repo]['Classification'].values[0])
    out_df = pd.DataFrame(data=out_dict)

    # create the csv
    out_df.to_csv(out_filepath)

    print(out_filepath, ' created with ',len(unique_repos),' repositories')

if __name__ == '__main__':
    main()
