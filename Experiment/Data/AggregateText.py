"""
The purpose of this script is to crawl github for all of those comments on all the pull
requests for the repos listed in AggregateText_repo_list.txt
"""
import os
HOME = os.path.abspath(os.path.join(__file__,'../../../'))
LIB = os.path.join(HOME,'Library')
import sys
sys.path.append(LIB)
import json
from pyexcel_ods3 import get_data
import code
import gitHubAPI
import logger
def log(*args):
    logger.log(args,pre='AggregateText.py')
def logJSON(*args):
    logger.logJSON(args,pre='AggregateText.py.json')
def main():
    # 0. set up the paths and direcory
    REPO_LST = os.path.join(HOME,'Experiment','Data','AggregateText_repo_list.txt')
    OUTDIR = os.path.join(HOME,'Experiment','Data','AggregateText')
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)
    #HERE
    DATA = '/home/a2sachs/Documents/Library/Data/GitHub/comments_for_pull_reqs'
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)
    # 1. Import the data
    # Create the mapping
    sample = get_data(REPO_LST)
    logJSON(sample)
    activeSheet, inactiveSheet = [sample[x] for x in ['active','inactive']]
    header = inactiveSheet[0]
    repoI = header.index('Repository')
    statusI = header.index('Status')
    repo_lst = []
    """
    {<repo_name>:{'status':active/inactive,
        'core contributors':{<name>:<all_text>,...}
        }
    }
    """
    for sheet in [activeSheet,inactiveSheet]:
        rows = sheet[1:]
        for row in rows:
            repo = row[repoI]
            repo_lst.append(repo)


    # 2. Get the core contributors for each repo
    # for repo in repos:
    #     core_contributors = gitHubAPI.getCoreContributors(repo)
    #     entry = [{item['author']:''} for item in core_contributors]
    #     log('core_contributors for',repo,'are',core_contributors)
    #     repos[repo]['core contributors'] = entry

    # 3. Aggregate the text for each of these contributors
    for repo in repo_lst:
        out_path = os.path.join(OUTDIR,repo.replace('/','__') + '.json')
        issue_comments_path = os.path.join(DATA,repo.replace('/','__'),'issue_comments.json')
        review_comments_path = os.path.join(DATA,repo.replace('/','__'),'review_comments.json')
        issues = json.load(open(issue_comments_path,'r'))
        reviews = json.load(open(review_comments_path,'r'))
        issues_lst = []
        for entry in issues:
            try:
                if entry not in ['documentation_url','message']:
                    name = entry['user']['login']

                    time_str = entry['created_at']
                    temp = time_str.split('-')
                    yr = temp[0]
                    mth = temp[1]
                    rest = temp[2]
                    temp = rest.split('T')
                    day = temp[0]
                    rest = temp[1][:-1] # take out the Z
                    hr,m,sec = rest.split(':')
                    timestamp = (yr,mth,day,hr,m,sec)
                    timestamp = tuple(int(x) for x in timestamp)

                    issue_url = entry['issue_url']
                    issue_num = int(issue_url[issue_url.rfind('/')+1:])
                    text = entry['body']
                    new_entry = {'name':name,'time':timestamp,'issue_num':issue_num,'text':text}
                    issues_lst.append(new_entry)
            except:
                print('error in issues in AggregateText')
                log('failed entry at repo',repo,'is',entry)
                log(entry)
        reviews_lst = []
        for entry in reviews:
            if entry == None:
                continue
            try:
                if entry not in ['documentation_url','message']:
                    user = entry['user']
                    if user == None: #some reivews don't have a user apparently
                        continue
                    name = user['login']

                    time_str = entry['created_at']
                    temp = time_str.split('-')
                    yr = temp[0]
                    mth = temp[1]
                    rest = temp[2]
                    temp = rest.split('T')
                    day = temp[0]
                    rest = temp[1][:-1] # take out the Z
                    hr,m,sec = rest.split(':')
                    timestamp = (yr,mth,day,hr,m,sec)
                    timestamp = tuple(int(x) for x in timestamp)

                    pr_url = entry['pull_request_url']
                    pr_num = int(issue_url[issue_url.rfind('/')+1:])
                    text = entry['body']
                    new_entry = {'name':name,'time':timestamp,'pr_num':issue_num,'text':text}
                    reviews_lst.append(new_entry)
            except Exception as err:
                print('error from AggregateText at 2',err)
                code.interact(local=locals())
                log('failed entry at repo',repo,'is',entry)
                log(entry)
        issues_lst.sort(key=lambda x: (x['issue_num'],x['time']))
        reviews_lst.sort(key=lambda x: (x['pr_num'],x['time']))
        
        allText = {} #{name:text}
        for entry in [*issues_lst,*reviews_lst]:
            name = entry['name']
            text = entry['text']
            if name not in allText:
                allText[name] = ''
            allText[name] += text + '. '
        # code.interact(local=locals())
        json.dump(allText,open(out_path,'w'),indent=4)


if __name__ == '__main__':
    logger.deleteLogs()
    main()
