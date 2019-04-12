"""
The purpose of this module is to provide an easy interface
to obtaining GitHub data
"""
import requests
import code
import json
import os
import time
import logger
import math
def log(*args,pre='gitHubAPI.py',fn=None):
    if fn != None:
        pre = 'gitHubAPI.{0}.py'.format(fn)
    logger.log(args,pre=pre)
def logJSON(*args,pre='gitHubAPI.py',fn=None):
    if fn != None:
        pre = 'gitHubAPI.{0}.py'.format(fn)
    logger.logJSON(args,pre=pre)

# requests can be seen here
token = ''
parent = os.path.dirname(os.path.realpath(__file__))
filename = '{0}/gitHubAPITokens.txt'.format(parent)
with open(filename, 'r') as f:
    tokens = [token.rstrip() for token in f.readlines()]
    # print('tokens', tokens)
tokenI = 0

def getContributors(repoPath):
    """
    (getContributors repoPath) -> [{'author':name,'numOfCommits':#_of_commits},...]) Given the repo path (ex. onivim/oni), return the list of contributors (most active first, etc.)
    getContributors: str str -> (listof (dictof str: (anyof str int)))
    ex.
    getContributors('onivim/oni') -> [{'author':'bryphe','numOfCommits':746}, {'author':'Akin909','numOfCommits':138},...]
    """
    fn = 'getContributors'
    url = 'https://api.github.com/repos/{0}/stats/contributors'.format(repoPath)
    resp = getJSON(url)
    logJSON(resp,fn=fn)
    log('resp',resp,fn=fn)
    contributors = resp
    log('type',type(contributors),fn=fn)
    ans = []
    for contributor in contributors[::-1]:
        # print('contributor', contributor['author'])
        author = contributor['author']['login']
        numOfCommits = contributor['total']
        entry = {
                'author': author,
                'numOfCommits': numOfCommits
                }
        ans.append(entry)
    return ans

def getCoreContributors(repoPath):
    """
    (getCoreContributors repoPath) -> [{'author':name,'numOfCommits'},...]
    Given a repo path (ex. onivim/oni) this function will return (in descending order) the core contributors (those who collectively contribute at least 80% of the total number of commits for the repository
    getCoreContributors: str -> (listof (dictof str (anyof str int)))
    ex
    getCoreContributors('onivim/oni') -> [{'author':'bryphe','numOfCommits':746}, ... {'author': 'keforbes', 'numOfCommits': 50}]
    """
    fn = 'getCoreContributors'
    log('called for ',repoPath,fn=fn)
    contributors = getContributors(repoPath)
    logJSON(contributors,fn=fn+'.contributors')
    totalNumCommits = sum([x['numOfCommits'] for x in contributors])
    soFar = 0
    ans = []
    for entry in contributors:
        soFar += entry['numOfCommits']
        ans.append(entry)
        log('so far',soFar/totalNumCommits,fn=fn)
        log('ans',ans,fn=fn)
        if soFar/totalNumCommits >= 0.8:
            return ans

def getJSON(urlPrefix):
    """
    (getJSON urlPrefix) -> resp This function obtains the json of the get request, or None if it's empty
    getJSON: str -> dict
    ex.
    getJSON('http://api.github.com/repos/onivim/oni') -> {'id': 73929422, 'node_id': 'MDEwOlJlcG9zaXRvcnk3MzkyOTQyMg==', 'name': 'oni', 'full_name': 'onivim/oni', 'private': False, 'owner': {'login': 'onivim', ...
    """
    fn = 'getJSON'
    log('called on',urlPrefix,fn=fn)
    global tokenI
    #Get the raw JSON response
    token = tokens[tokenI]
    if tokenI < 5:
        tokenI += 1
    else:
        tokenI = 0
    url = '{0}?access_token={1}'.format(urlPrefix, token)
    log('final url',url,fn=fn)
    response = requests.get(url)
    log('response',response,fn=fn)
    resp = response.json()
    logJSON(resp,fn=fn+'.resp')
    if response.status_code == 202: #202 indicates too many requests (I think...)
        print('gitHubAPI sleeping for',urlPrefix,'...')
        time.sleep(10)
        return getJSON(urlPrefix)
    return resp

def getRepoJSON(fullRepoName):
    """
    Get the raw GitHub reply for that repo's homepage
    """
    url = 'http://api.github.com/repos/{0}'.format(fullRepoName)
    # response = requests.get(url)
    resp = getJSON(url)
    return resp

def getStars(fullRepoName):
    url = 'http://api.github.com/repos/{0}?access_token={1}'.format(fullRepoName, token)
    response = requests.get(url)
    resp = response.json()
            # > http://api.github.com/repos/[username]/[reponame]
            # > obj['watchers_count']
def pullListToDict(urlLst):
    """
    (pullListToDict urlLst) Takes the url list of a pull request conversations (urlLst)
    and transforms it into a dictionary of comments of the form:
    {url: [{name: '...', comment: '...'},...],...}
    pullListToDict: str -> (dictof str: (listof (dictof str: str)))
    examples:
    pullListToDict('https://github.com/onivim/oni/pull/2478') -> 
    {'https://github.com/onivim/oni/pull/2478': [{'name':'Akin909', 'comment': '''Reverts onivim/oni#2472

    @CrossR @bryphe this awesome bugfix PR unfortunately breaks a users ability to to navigate too and from the sidebar using keyboard bindings, in my excitement to have this annoying issue fixed I totally missed this bug till just now.'
    '''}, 
    {'name':'CrossR', 'comment':'Not good! Guess we need to get this under CI tests as well.'}]}
    """
    ans = {}
    for url in urlLst:
        lst = pullUrlToList(url)
        ans[url] = lst
    return ans

def pullUrlToList(url):
    """
    (pullUrlToDict url) Takes the url of a pull request conversation (url) and
    transforms it into a list of comments of the form:
    [{name: '...', comment: '...'},...]
    pullUrlToList: str -> (listof (dictof str: str))
    examples:
    pullUrlToList('https://github.com/onivim/oni/pull/2478') -> 
    [{'name':'Akin909', 'comment': 'Reverts onivim/oni...'},
    'name':'CrossR', 'comment': 'Not good! Guess we...']
    """
    ans = []
    ans.append(getPullDescComment(url))
    rest = []
    rest.extend(getPullReviews(url))
    rest.extend(getPullComments(url))
    rest.sort(key=lambda d: d['timestamp'])
    ans.extend(rest)
    return ans


def getPullDescComment(url):
    """
    (getPullDescComment url) Takes the url of a pull request conversation (url) and
    extracts the description comment at the top of the conversation in the form a 
    a dictionary: {name: '...', comment: '...'}
    getPullDescComment: str -> (dictof str:str)
    example:
    getPullDescComment('https://github.com/onivim/oni/pull/2478') -> 
    {'name':'Akin909', 'comment': 'Reverts onivim/oni...'}
    """
    # Convert the provided url into the one the API needs
    url = url.replace('https://github.com/', 'https://api.github.com/repos/')
    url = url.replace('/pull/', '/issues/')
    # primeUrl = '{0}?access_token={1}'.format(url, token)
    # response = requests.get(primeUrl)
    # resp = response.json()
    resp = getJSON(url)
    log(resp)
    name = resp['user']['login']
    comment = resp['body']

    #Return the answer
    ans = {'name': name, 'comment': comment}
    return ans

def getPullComments(url):
    """
    (getPullComments url) Takes the url of a pull request conversation (url) and
    extracts the comments in the conversation and returns
    a list: [{name: '...', comment: '...'},...]
    getPullComments: str -> (listof (dictof str:str))
    example:
    getPullComments('https://github.com/onivim/oni/pull/2478') -> 
    [TODO
    """
    #Translate raw url into the appropriate comments url
    url = url.replace('https://github.com/', 'https://api.github.com/repos/')
    url = url.replace('/pull/', '/issues/')
    url += '/comments'
    # primeUrl = '{0}?access_token={1}'.format(url, token)
    # response = requests.get(primeUrl)
    # resp = response.json()
    resp = getJSON(url)
    ans = []
    for d in resp:
        name = d['user']['login']
        comment = d['body']
        timestamp = d['created_at']
        item = {'name': name, 'comment': comment, 'timestamp': timestamp}
        ans.append(item)
    return ans

def getPullReviews(url):
    """
    (getPullReviews url) Takes the url of a pull request conversation (url) and
    extracts the reviews in the conversation and returns
    a list: [{name: '...', comment: '...', timestamp:'...'},...]
    getPullComments: str -> (listof (dictof str:str))
    example:
    getPullReviews('https://github.com/onivim/oni/pull/2478') -> 
    [TODO
    """
    log('Provided url: {0}'.format(url))
    #Translate raw url into the appropriate comments url
    url = url.replace('https://github.com/', 'https://api.github.com/repos/')
    url = url.replace('/pull/', '/pulls/')
    url += '/reviews'
    log('url',url)
    # primeUrl = '{0}?access_token={1}'.format(url, token)
    # log('primeUrl: {0}'.format(primeUrl))
    # response = requests.get(primeUrl)
    # resp = response.json()
    resp = getJSON(url)
    #if the response was found then proceed, otherwise return an empty list (as there are no reviews)
    message = None
    if type(resp) == dict:
        message = resp.get('message')
    log('message: {0}'.format(message))
    if message == None:
        ans = []
        log('resp: {0}'.format(resp))
        for d in resp:
            name = d['user']['login']
            comment = d['body']
            timestamp = d['submitted_at']
            item = {'name': name, 'comment': comment, 'timestamp': timestamp}
            if comment != '':
                ans.append(item)
        return ans
    else:
        return []

def getClosedPullRequestUrls(repoName,datestr):
    """
    (getClosedPullRequestUrls repoName) -> listof Urls: This function returns a list of the urls (as per normal user interface)
    for all of the closed pull requests (given the optional filters).
    getClosedPullRequestUrls: str -> (listof str)
    ex.
    getClosedPullRequestUrls('onivim/oni') -> ['https://github.com/onivim/oni/pull/2705', 'https://github.com/onivim/oni/pull/2697',...]
    """
    fn = 'getClosedPullRequestUrls'
    #1. Get the number of entries
    prefix = 'https://api.github.com/search/issues'
    query = 'q=repo:"{0}"+is:pr+state:closed+created:<{1}'.format(repoName,datestr)
    sorting = 'sort=created'
    base_url = '{0}?{1}&{2}'.format(prefix,query,sorting)
    resp = getJSON(base_url)
    total_count = resp['total_count']
    #2 for each page append the urls to ans
    ans = []
    total_page_count = math.ceil(total_count/30)
    for i in range(total_page_count):
        pageNum = i+1
        pagination = 'page={0}&per_page=100'.format(pageNum)
        url = '{0}&{1}'.format(base_url,pagination)
        resp = getJSON(url)
        for item in resp['items']:
            ans.append(item['url'])
    return ans

# def getJson(url, filename):
#     primeUrl = '{0}?access_token={1}'.format(url, token)
#     response = requests.get(primeUrl)
#     resp = response.json()
#     f = open('./'+filename, 'w')
#     json.dump(resp, f, indent=4)
#     f.close() 
#     return resp

def outputTempJson(jsonObj):
    home = os.environ.get('HOMEPATH')
    if home == None:
        home = os.environ.get('HOME')
    filepath = os.path.join(home, 'Dropbox', 'Masters', 'Thesis', 'Scripts', 'ThesisSiteDev', 'output.json')
    with open(filepath, 'w') as writer:
        json.dump(jsonObj, writer, indent=4)

def searchRepositories(searchWords, sortBy='stars', order='desc'):
    # searchURL = 'https://api.github.com/search/repositories?q=tetris+language:assembly&sort=stars&order=desc'
    searchSequnce = '{0}&sort={1}&order={2}'.format(searchWords, sortBy, order)
    # searchWords = 'tetris+language:assembly&sort=stars&order=desc'
    # searchWords = 'hello&sort=stars&order=desc'
    baseURL = 'https://api.github.com/search/repositories'
    searchURL = '{0}?q={2}?access_token={1}'.format(baseURL, token, searchSequnce)
    response = requests.get(searchURL)
    resp = response.json()
    home = os.environ.get('HOMEPATH')
    if home == None:
        home = os.environ.get('HOME')
    filepath = os.path.join(home, 'Dropbox', 'Masters', 'Thesis', 'Scripts', 'ThesisSiteDev', 'searchResults.json')
    with open(filepath, 'w') as writer:
        json.dump(resp, writer, indent=4)


if __name__ == "__main__":
    logger.deleteLogs()
    d = getClosedPullRequestUrls('onivim/oni','2019-02-25')
    print(d)
    code.interact(local=locals())
