"""
This is Rahul Iyer's script (with some modifications) for crawling comments
"""
import requests, json, time, random
global tokens,auth_header
from pathlib import Path
import pandas as pd
import datetime
from multiprocessing import Pool
import multiprocessing as mp
import os
import code
import sys
HOME = os.path.abspath(os.path.join(__file__,'../../../'))
LIB = os.path.join(HOME,'Library')
sys.path.append(LIB)
DATA = os.path.join(HOME,'Experiment','Data','comments_for_pull_reqs')
import logger
def log(*args,pre=None):
    pre = ('comments_for_pull_reqs.py' if pre == None else 'comments_for_pull_reqs.py.'+pre)
    logger.log(*args,pre=pre)
tokens_path = os.path.join(HOME,'Experiment','Data','gitHubAPITokens.txt')
tokens = [line.strip() for line in open(tokens_path,'r')]
# tokens = ['159b497bb00dbd384daeed3bb580570b30202ad8','5cf8f5a2fa29362b2dc4f2c0466b76e0e342c4ea','d083963d6bc40d48463e195764ebaaab6ce716c4','3f144d71a9acb23a76a3747a2619f5467c107128'] #rahul-iyer,iyer-rahul,dishant,alex sun,
auth_header = {'Authorization':'token ' + tokens[0]}


# AS; don't know why this is here....
# df = pd.read_csv('/home/r3iyer/Desktop/whole.csv')

def contributors(full_repo_name,i,num_pages=100000):
    # GET /repos/:owner/:repo/stats/contributors
    """
    Get all the contributors
    """
    request_url = 'https://api.github.com/repos/%s/stats/contributors' % full_repo_name
    return generic(request_url, i, num_pages=num_pages)


def generate_token(index):
    """
    This function generates a token?
    """
    pre = 'generate_token'
    log('start index',index,pre=pre)
    global tokens
    limit_nos = []
    url= 'https://api.github.com/rate_limit'
    index = index % len(tokens)
    log('end index',index,pre=pre)
    time_remaining = [] 
    for token in tokens[index:index+1]:
        log('token',token,'for index',index,pre=pre)
        try:
            res = requests.get(url,headers={'Authorization':'token ' + token}).json()
        except Exception as e:
            time.sleep(6)
            res = requests.get(url,headers={'Authorization':'token ' + token}).json()

        #print(res)
        if not "resources" in  res:
            auth_header = {'Authorization':'token ' + random.choice(tokens)}
            return "Positive RateLimit"
        limit_nos.append(res["resources"]["core"]["remaining"])
        time_remaining.append(res["resources"]["core"]["reset"] -time.time())

    if sum(i > 3  for i in limit_nos) < 1:
        # selected_token = time_remaining.index(min(time_remaining))
        selected_token = index #AS: needs to be the token of the index provided
        print('Sleeping until %s' % (datetime.datetime.now() + datetime.timedelta(0,time_remaining[selected_token])))
        time.sleep(time_remaining[selected_token] + 2)
        curr_token = tokens[selected_token]
    else:
        # selected_token = limit_nos.index(max(limit_nos))
        selected_token = index
        curr_token = tokens[selected_token]
        
    return curr_token
    


def rate_reset_wait(headers,token):
    """
    This function waits for cooldowns?
    """
    if 'X-RateLimit-Remaining' in headers:
        ratelimit_remaining = int(headers['X-RateLimit-Remaining'])
    else:
        ratelimit_remaining = 1
    if ratelimit_remaining <= 0:
        print("Waiting for %d minutes..." % ((int(headers['X-RateLimit-Reset']) - time.time())//60))
        return "RateLimit Reset"
    else:
        if ratelimit_remaining % 100 == 0:
            print('X-RateLimit-Remaining:', ratelimit_remaining,token)
        return "Positive RateLimit"



def generic(request_url, i, headers=None, num_pages=1, iterable=True):
    """
    This function is merely a REST call
    Returns the responses of all of the requests in the chain
    """
    global tokens
    pre = 'generic'
    log('obtain token',i,'from generate_token',pre=pre)
    token = generate_token(i)
    auth_header = {'Authorization':'token ' + token}
    merged_response = list()
    for i0 in range(num_pages):
        # print("Request:", request_url)
        response = None # AS: seems to need to be carried over
        try:
            if headers is not None:
                response = requests.get(request_url, headers=auth_header)
            else:
                response = requests.get(request_url, headers=auth_header)
            wait_status = rate_reset_wait(response.headers,token)
            if wait_status == "Positive RateLimit":
                if iterable:
                    merged_response.extend(response.json())
                else:
                    merged_response.append(response.json())
            else:
                token = generate_token(i)
                auth_header = {'Authorization':'token ' + token}
                
                if headers is not None:
                    response = requests.get(request_url, headers=auth_header)
                else:
                    response = requests.get(request_url, headers=auth_header)
                if iterable:
                    merged_response.extend(response.json())
                else:
                    merged_response.append(response.json())

            # Change request_url to next url in the link
            if 'Link' in response.headers:
                raw_links = response.headers['Link'].split(',')
                next_url = None
                for link in raw_links:
                    split_link = link.split(';')
                    if split_link[1][-6:] == '"next"':
                        next_url = split_link[0].strip()[1:-1]
                        break
                if next_url is not None:
                    request_url = next_url
                else:
                    break
            else:
                break
        except Exception as e:
            print(e)
            log(e,response.json(),pre='generic') #AS: added to capture errors
            continue

    return merged_response    
    
def review_comments(full_repo_name,i, num_pages=100000, pr_number=None):
    """
    Obtain all of the review comments
    """
    if pr_number is None:
        request_url = 'https://api.github.com/repos/%s/pulls/comments' % full_repo_name
        return generic(request_url, i, num_pages=num_pages)
    else:
        request_url = 'https://api.github.com/repos/%s/pulls/%s/comments' % (full_repo_name, pr_number)
        return generic(request_url, i, num_pages=num_pages)


def issue_comments(full_repo_name,i,num_pages=100000, issue_number=None):
    """
    Get all the issues comments
    """
    pre = 'issue_comments'
    log('issue_comments for repo',full_repo_name,'uses token',i,pre=pre)
    if issue_number is None:
        request_url = 'https://api.github.com/repos/%s/issues/comments' % full_repo_name
        return generic(request_url, i,num_pages=num_pages)
    else:
        request_url = 'https://api.github.com/repos/%s/issues/%s/comments' % (full_repo_name, issue_number)
        return generic(request_url, i,num_pages=num_pages)


def crawl_comments(reponame):
    """
    This function will crawl for all of the comments for the repositories given
    and place them in the comments_for_pull_reqs/<reponame> folder for each reponame in repoLst
    """
    # 0. setup the paths
    datadir = '/home/a2sachs/Documents/Library/Data/GitHub/comments_for_pull_reqs'
    # repodirs = [os.path.join(datadir,reponame.replace('/','__')) for reponame in repoLst]
    repopath = os.path.join(datadir,reponame.replace('/','__'))
    # 1. setup the folders
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    if not os.path.exists(repopath):
        os.mkdir(repopath)
    # 2. get all the issue and review comments
    # issue comments
    repopath = os.path.join(datadir,reponame.replace('/','__'),'issue_comments.json')
    data = issue_comments(reponame)
    json.dump(data,open(repopath,'w'),indent=4)
    # review comments
    reviewpath = os.path.join(datadir,reponame.replace('/','__'),'review_comments.json')
    data = review_comments(reponame)
    json.dump(data,open(reviewpath,'w'),indent=4)
    # contributors
    contributors_path = os.path.join(datadir,reponame.replace('/','__'),'contributors.json')
    data = contributors(reponame)
    json.dump(data,open(contributors_path,'w'),indent=4)

def crawl_comments_2(args):
    """
    This function will crawl for all of the comments for the repositories given
    and place them in the comments_for_pull_reqs/<reponame> folder for each reponame in repoLst
    """
    pre = 'crawl_comments_2'
    repo_lst,i = args
    # 0. setup the paths
    datadir = DATA
    repodirs = [os.path.join(datadir,reponame.replace('/','__')) for reponame in repo_lst]
    # 1. setup the folders
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    for repodir in repodirs:
        if not os.path.exists(repodir):
            os.mkdir(repodir)
    # 2. get all the issue and review comments
    for reponame in repo_lst:
        start = time.time()
        # issue comments
        repopath = os.path.join(datadir,reponame.replace('/','__'),'issue_comments.json')
        log('getting issues for',i,pre=pre)
        data = issue_comments(reponame,i)
        json.dump(data,open(repopath,'w'),indent=4)

        # review comments
        reviewpath = os.path.join(datadir,reponame.replace('/','__'),'review_comments.json')
        data = review_comments(reponame,i)
        json.dump(data,open(reviewpath,'w'),indent=4)

        # contributors
        contributors_path = os.path.join(datadir,reponame.replace('/','__'),'contributors.json')
        data = contributors(reponame,i)
        json.dump(data,open(contributors_path,'w'),indent=4)

        # code.interact(local=locals())
        log('finished repo',reponame,pre=pre)
        end = time.time()
        print('It took',end-start,'seconds to crawl',reponame)

def crawl_comments_for_id(inputs):
    """
    Writes the issue and reivew comment ids to '/comments_pr/...json'
    """
    issue_output = issue_comments(inputs[0],issue_number = inputs[1])
    review_output = review_comments(inputs[0],pr_number = inputs[1])
    result  = {inputs[2] : issue_output + review_output}
    with open('comments_pr/' + str(inputs[2])+'.json','w') as f:
        json.dump(result,f)

#     with open('pr_comments/' + inputs[2] + '.json','w') as f:
#         json.dump(issue_output + review_output, f)


# In[25]:

#AS: df leads to fdf
# fdf = df[df['num_comments'] >= 10]


# In[26]:

# fdf.head()

def main():
    start = time.time()

    #0. setup the parameters
    repo_txt_file = os.path.join(HOME,'Experiment','Data','repo_list.txt')
    repo_lst = [line.strip() for line in open(repo_txt_file,'r')]
    done_repo_txt_file = os.path.join(HOME,'Experiment','Data','DoneRepos.txt')
    done_repo_lst = [] if not os.path.exists(done_repo_txt_file) else {line.strip() for line in open(done_repo_txt_file,'r')}
    print('type done',type(done_repo_lst))
    final_lst = [repo for repo in repo_lst if repo not in done_repo_lst]
    print('final list is',len(final_lst),'repos')
    # point = repo_lst.index('emberjs/ember.js')
    # repo_lst = repo_lst[point+1:point+10]

    # AS
    # crawl_comments(repo_lst)

    # d = issue_comments('tensorflow/tensorflow')
    # code.interact(local=locals())

    # repo_lst = ['symfony/symfony','pixijs/pixi.js']
    pool = Pool(processes=6)
    inputs = []
    for i,reponame in enumerate(final_lst):
        if i < 6:
            inputs.append([reponame])
        else:
            inputs[i%6].append(reponame)
    for i in range(len(inputs)):
        inputs[i] = [inputs[i][:],i]
    log('inputs',inputs)
    print('final_lst has',len(final_lst),'repos')
    pool.map(crawl_comments_2, inputs)

    #INACTIVE
    # repo_lst = ['Homebrew/legacy-homebrew']
    # repo_lst = ['Prinzhorn/skrollr']
    # repo_lst = ['Shopify/dashing']
    # repo_lst = ['JakeWharton/ActionBarSherlock']
    # repo_lst = ['omab/django-social-auth']
    #ACTIVE
    # repo_lst = ['facebook/react']
    # repo_lst = ['d3/d3']
    # repo_lst = ['facebook/react-native']
    # repo_lst = ['electron/electron']
    # repo_lst = ['tensorflow/tensorflow']

    # repo_lst = [['symfony/symfony'],0]
    # print('repo_lst',repo_lst)
    # crawl_comments_2(repo_lst)

    end = time.time()
    print('It took',end-start,'seconds to crawl the repos')

if __name__ == '__main__':
    logger.deleteLogs()
    main()
