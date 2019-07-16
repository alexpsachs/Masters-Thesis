"""
The purpose of this script is to trim down the tables in ~/Documents/Library/Data/GitHub to
just the rows we actually need (then put those rows in Selection directory)
"""

import time
import os
from pyexcel_ods3 import get_data
from pyexcel_io import get_data as csv_get_data
from csv import reader
LIB = '/home/a2sachs/Documents/Library'
import sys
sys.path.append(LIB)
import logger
import pymysql.cursors
def log(*args,pre=None):
    if pre == None:
        pre = 'Selection.py'
    else:
        pre = 'Selection.py'+pre
    logger.log(*args,pre=pre)

# 1. Setup the paths
DATA = '/home/a2sachs/Documents/Library/Data/GitHub/mysql-2019-02-01'
OUT = '/home/a2sachs/Documents/Experiment2.1/Data/Selection'
SAMPLE = '/home/a2sachs/Documents/Experiment2.1/Data/SampleSelection.ods'

projects_csv = '/home/a2sachs/Documents/Experiment2.1/Data/Selection/projects.csv'
pull_requests_csv = '/home/a2sachs/Documents/Experiment2.1/Data/Selection/pull_requests.csv'

# 2. Make the folder if it doesn't already exist
if not os.path.exists(OUT):
    os.mkdir(OUT)

# RETRIEVAL FUNCTIONS
def getPullRequestIds():
    data = csv_get_data(pull_requests_csv)['pull_requests.csv']
    headers = data[0]
    idI = headers.index('id')
    ids = {e[idI] for e in data[1:]}
    log('pull request ids',ids)
    return ids

def getRepoIds():
    # returns {repo:id}
    data = csv_get_data(projects_csv)['projects.csv']
    headers = data[0]
    idI = headers.index('id')
    ids = {e[idI] for e in data[1:]}
    log('repo ids',ids)
    return ids

def getRepos():
    repos = set()
    data = get_data(SAMPLE)
    activeSheet, inactiveSheet = data['active'], data['inactive']
    header = activeSheet[0]
    repoI = header.index('Repository')
    for sheet in [activeSheet, inactiveSheet]:
        for row in sheet[1:]:
            repo = row[repoI]
            repos.add(repo)
    log('repos of interest')
    log(repos)
    return repos

def getTableFields(tbl):
    conn = pymysql.connect(host='localhost',
                                 user='ghtorrentuser',
                                 password='ghtorrentpassword',
                                 db='ghtorrent',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
    cur = conn.cursor()
    cur.execute("SHOW COLUMNS FROM {0}".format(tbl))
    fields = []
    for response in cur:
        fields.append(response['Field'])
    cur.close()
    conn.close()
    return fields

def readAndParseCSVLine(f,delimiter=',',eol='\n',quotes='"'):
    ignore = False
    original = ''
    curr = ''
    lst = []
    newline = False
    while not newline:
        c = f.read(1)
        original += c
        if c == delimiter:
            if not ignore:
                lst.append(curr)
                curr = ''
            else:
                curr += c
        elif c == '\n':
            if not ignore:
                lst.append(curr)
                newline = True
                curr = ''
            else:
                curr += c
        elif c == quotes:
            ignore = not ignore
        elif c == '':
            print('end of file?...')
            return None
        else:
            curr += c
    return lst

def trimCSV(filename, col_label, match_citeria):
    lst = match_citeria
    inpath = os.path.join(DATA,filename)
    outpath = os.path.join(OUT,filename)
    line = 0
    f = open(inpath,'r')
    g = open(outpath,'w')
    l = open(os.path.join(OUT,filename+'.log'),'w')

    headers = getTableFields(filename[:-4])
    col = None
    if type(col_label) == list:
        col = [headers.index(e) for e in col_label]
    else:
        col = headers.index(col_label)  
    g.write(','.join(headers)+'\n')

    start = time.time()
    # for row in f:
    while True:
        items = readAndParseCSVLine(f)
        if items == None:
            break
        if line % 1000000 == 0:
            log('line',line)
            log('items',items)
            log('dir f',dir(f))
        try:
            doWrite = False
            if type(col) == list:
                xs = [items[c] for c in col]
                for x in xs:
                    if x in lst:
                        doWrite = True
            else:
                x = items[col]
                if x in lst:
                    doWrite = True
            if doWrite:
                log('wrote:',x)
                g.write(','.join(items)+'\n')
        except:
            l.write(str(line)+'\n')
        line += 1
    end = time.time()
    # close the connections
    # f.close()
    g.close()
    l.close()
    print('It took',end-start,'seconds to do all rows for',filename)

# TRIM FUNCTIONS
def trim_projects(repos):
    lst = repos
    filename = 'projects.csv'
    inpath = os.path.join(DATA,filename)
    outpath = os.path.join(OUT,filename)
    repoI = 1
    repoStartI = len('https://api.github.com/repos/')
    line = 0
    f = open(inpath,'r')
    g = open(outpath,'w')
    l = open(os.path.join(OUT,filename+'.log'),'w')

    headers = getTableFields(filename[:-4])
    g.write(','.join(headers)+'\n')

    start = time.time()
    for row in f:
        if line % 1000000 == 0:
            log('line',line)
        items = row.split(',',maxsplit=2)
        try:
            x = items[repoI][repoStartI+1:-1]
            if x in lst:
                g.write(row)
        except:
            l.write(str(line)+'\n')
        line += 1
    end = time.time()
    # close the connections
    f.close()
    g.close()
    l.close()
    print('It took',end-start,'seconds to do all rows for',filename)


# def trim_pull_requests(repoIds):
def trim_pull_requests():
    # lst = repoIds
    lst = ['16311']
    filename = 'pull_requests.csv'
    inpath = os.path.join(DATA,filename)
    outpath = os.path.join(OUT,filename)
    line = 0
    f = open(inpath,'r')
    g = open(outpath,'w')
    l = open(os.path.join(OUT,filename+'.log'),'w')

    headers = getTableFields(filename[:-4])
    # col = headers.index('head_repo_id')
    col = headers.index('id')
    g.write(','.join(headers)+'\n')

    start = time.time()
    for row in f:
        if line % 1000000 == 0:
            log('line',line)
        items = row.split(',')
        try:
            x = items[col]
            # print('col',col)
            # print('x',x)
            if x in lst:
                g.write(row)
        except:
            l.write(str(line)+'\n')
        line += 1
    end = time.time()
    # close the connections
    f.close()
    g.close()
    l.close()
    log('done')
    print('It took',end-start,'seconds to do all rows for',filename)

def trim_pull_request_comments():
    lst = ["And voila!",'And voila!']
    filename = 'commit_comments.csv'
    inpath = os.path.join(DATA,filename)
    outpath = os.path.join(OUT,filename)
    line = 0
    f = open(inpath,'r')
    g = open(outpath,'w')
    l = open(os.path.join(OUT,filename+'.log'),'w')

    headers = getTableFields(filename[:-4])
    col = headers.index('body')
    g.write(','.join(headers)+'\n')

    start = time.time()
    for row in f:
        if line % 1000000 == 0:
            log('line',line)
        items = row.split(',')
        try:
            x = items[col]
            # print('col',col)
            # print('x',x)
            # if x in lst:
            if x in lst:
                g.write(row)
        except:
            l.write(str(line)+'\n')
        line += 1
    end = time.time()
    # close the connections
    f.close()
    g.close()
    l.close()
    log('done')
    print('It took',end-start,'seconds to do all rows for',filename)
#MAIN
if __name__ == '__main__':
    logger.deleteLogs()

    # repos = getRepos()
    # trim_projects(repos)

    # repoIds = getRepoIds()
    # repoIdStrs = {str(x) for x in repoIds}
    # trimCSV('pull_requests.csv',['head_repo_id','base_repo_id'],repoIdStrs)

    
    #test
    # s = '1,2,3,45,"I think blah\nboop bee"\n'
    # print('parse',s)
    # print(parseCSVLine(s))
    prIds = getPullRequestIds()
    prIdStrs = {str(x) for x in prIds}
    trimCSV('pull_request_comments.csv','pull_request_id',prIdStrs)

    print('done done done all done')

