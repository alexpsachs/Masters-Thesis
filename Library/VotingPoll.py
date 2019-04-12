"""
This module is responsible for converting the annotation data from
the numerous formats into one
"""
#Imports
import json
import os
from collections import OrderedDict

#Constants
curr_filepath = os.path.abspath(__file__)
parent = os.path.dirname(os.path.abspath(__file__))
SYMLOG_LABELS = json.load(open(os.path.join(parent,'SYMLOG_LABELS.json'),'r'))
HOME = os.sep.join(parent.split(os.sep)[:-2])
VOTING_BANK = os.path.join(HOME,'Dropbox','Masters','Thesis','Scripts','ThesisSiteDev','VotingBank')
data = None

#Debug function
import sys
sys.path.append(os.path.join(HOME,'Dropbox','Logging'))
import Logger
Debug = True
def p(*args,level=0):
    if Debug:
        Logger.log(*args,level=level,filepath=curr_filepath)
p('somefn',1,2,3,level=1)

#Functions
def aggregateDataFolder(folderPath,judge):
    """(str,str) -> (dictof str:str:str:(dictof str:int,str:int))

    This function takes in a folder path and the name of the judge who created
    these records and returns a dictionary of the data the folder contained.

    (folder_path, name_of_judge) -> {judge:repo:person:{'labels':{label:count},'conversationCount':int}}
    """
    ans = {}
    recordNames = [path for path in os.listdir(folderPath) if path.startswith('record')]
    recordPaths = [os.path.join(folderPath,name) for name in recordNames]
    record = None
    for recordPath in recordPaths:
        with open(recordPath,'r') as f:
            record = json.load(f)
        repo = getRepoName(record)
        person = record['conversation']['target']
        vote = record['opinion']
        emptyEntry = {'labels':{label:0 for label in SYMLOG_LABELS},'conversationCount':0}
        if judge not in ans.keys():
            ans[judge] = {}
        if repo not in ans[judge].keys():
            ans[judge][repo] = {}
        if person not in ans[judge][repo].keys():
            ans[judge][repo][person] = emptyEntry
        for label,val in vote.items():
            ans[judge][repo][person]['labels'][label] += int(val)
        ans[judge][repo][person]['conversationCount'] += 1
    return ans

def aggregateVotingBank():
    """() -> (dictof str:str:str:(dictof str:int,str:int))

    This function aggregates all of the data in the VotingBank and returns
    it as a dictionary.

    () -> {judge:repo:person:{'labels':{label:count},'conversationCount':int}}
    """
    fn = 'aggregateVotingBank'
    global data
    if data == None:
        data = {}
        folderJudgeLst = [('AlexData','Alex'),
                ('AndrewAtom0','Andrew'),
                ('AndrewAtom1','Andrew'),
                ('AndrewChromedeveditor','Andrew'),
                ('AndrewDolphin','Andrew'),
                ('AndrewOni','Andrew')]
        # folderJudgeLst = [('AndrewChromedeveditor','Andrew')]
        # folderJudgeLst = [('TestData','TestJudge')]
        for folder,judge in folderJudgeLst:
            p(fn,'folder,judge',folder,judge)
            ext = aggregateDataFolder(os.path.join(VOTING_BANK,folder), judge)
            if judge == 'Andrew':
                p(fn,'andrew ext',ext)
            # Update data
            if judge not in data:
                data.update(ext) 
            else:
                for repo in ext[judge]:
                    if repo not in data[judge]:
                        data[judge][repo] = {}
                    for person in ext[judge][repo]:
                        if person not in data[judge][repo]:
                            data[judge][repo][person] = ext[judge][repo][person]
                        else:
                            #merge the person data
                            dataEntry = data[judge][repo][person]
                            extEntry = ext[judge][repo][person]
                            for label in dataEntry['labels']:
                                dataEntry['labels'][label] += extEntry['labels'][label]
                            dataEntry['conversationCount'] += extEntry['conversationCount']
            if 'Andrew' in data:
                p(fn,'data.update andrew keys',data['Andrew'].keys())
        p(fn,'Andrew data',data['Andrew'].keys())
    return data
    
def getRepoName(record):
    """(Record) -> str

    This function takes in a record dict and returns the full repo name
    """
    urlStr = record['conversation']['url']
    ans = '/'.join(urlStr.split('/')[3:5])
    return ans

def getVotingBankData():
    """() -> (dictof str:str:str:(dictof str:int,str:int))

    This function returns all the voting bank data in the form of a dictionary.

    () -> {judge:repo:person:{'labels':{label:count},'conversationCount':int}}
    """
    global data
    if data == None:
        aggregateVotingBank()
    return data

