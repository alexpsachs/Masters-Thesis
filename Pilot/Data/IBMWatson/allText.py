"""
The objective of this script is to aggregate all of the text as it was used for the annotations
found in '/home/alex/a2sachs/Dropbox/Masters/Thesis/Scripts/ThesisSiteDev/VotingBank/Oni_SYMLOG.ods'
"""
LIB = '/home/a2sachs/Documents/Library'
import sys
sys.path.append(LIB)
import gitHubAPI
import logger
log = logger.log
from pyexcel_ods3 import get_data
import json

if __name__ == '__main__':
    logger.deleteLogs()

# 0. Set up your paths
oni_SYMLOG_filepath = '/home/a2sachs/Dropbox/Masters/Thesis/Scripts/ThesisSiteDev/VotingBank/Oni_SYMLOG.ods'
dest_filepath = '/home/a2sachs/Documents/Experiment1/Data/IBMWatson/allText.json'

# 1. Get the common urls
data = get_data(oni_SYMLOG_filepath)
annotations = data['Annotations']
url_i = annotations[0].index('Thread URL')
common_urls = [row[url_i] for row in annotations[1:]]

# 2. Convert these urls into comments
out = {} # {person:text}
for url in common_urls:
    lst = gitHubAPI.pullUrlToList(url) # -> [{'name':'Akin909', 'comment': 'Reverts onivim/oni...'}, ...]
    for row in lst:
        person = row['name']
        comment = row['comment']
        if person in out:
            out[person] += ' ' + comment
        else:
            out[person] = comment

# 3. Dump {person:text} into the json
with open(dest_filepath, 'w') as f:
    json.dump(out, f, indent=4)

