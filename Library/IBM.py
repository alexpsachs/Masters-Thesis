"""
The purpose of this library is to provide an API for accessing
information in the IBM json files
"""
#IMPORTS
import os
import json
import code
import time
import sys
from nltk.tokenize import word_tokenize
from watson_developer_cloud import PersonalityInsightsV3
import logger
def log(*args,pre=None):
    if pre == None:
        pre = 'IBM'
    logger.log(*arg,pre=pre)

personality_insights = PersonalityInsightsV3(
   version='2018-08-22',
   username='7d494188-7567-4bbd-aa26-94b5b1b3825f',
   password='wxZ7kmHzampN',
   # url='https://gateway.watsonplatform.net/personality-insights/api'
   url='https://gateway.watsonplatform.net/personality-insights/api'
)

HOME = os.path.abspath(os.path.join(__file__,'../../'))

#FUNCTIONS
def getBig5(filepath):
    """(str) -> (dictof str:number)

    This function returns a dictionary of the big 5 attributes for a particular
    json file (filepath).
    """
    data = None
    with open(filepath, 'r') as f:
        data = json.load(f)
    persona = data['personality']
    ans = {
            'openness':list(filter(lambda x: x['name'] == 'Openness', persona))[0]['raw_score'],
            'conscientiousness':list(filter(lambda x: x['name'] == 'Conscientiousness', persona))[0]['raw_score'],
            'extraversion':list(filter(lambda x: x['name'] == 'Extraversion', persona))[0]['raw_score'],
            'agreeableness':list(filter(lambda x: x['name'] == 'Agreeableness', persona))[0]['raw_score'],
            'neuroticism':list(filter(lambda x: x['name'] == 'Emotional range', persona))[0]['raw_score'],
            }
    return ans

def getResult(text,filename=None):
    words = text.split(' ',maxsplit=400)
    def report_error(s):
        with open(filename,'w') as f:
            f.write(s)
    def file_ans(s):
        json.dump(s,open(filename,'w'),indent=4)
    if len(words) < 300:
        if filename != None:
            report_error('AS: less than 300 words')
        return None
    status_code = None
    profile = None
    try:
        profile = personality_insights.profile(
               text,
               content_type='text/plain',
               consumption_preferences=True,
               raw_scores=True)
        status_code = profile.status_code
    except Exception as err:
        if err.code == 400: # not enough words
            if filename != None:
                report_error('AS: Error code 400, not enough words')
                file_ans(None)
            return None
        elif err.code == 413: # text is too big so watson can't process it
            all_words = text.split(' ')
            new_text = ' '.join(all_words[len(all_words)//2:])
            ans = getResult(new_text)
            if filename != None:
                file_ans(ans)
            return ans
        elif err.code == 500: # an internal error on IBM's server
            print('error code 500 encountered, gonna sleep and try again...')
            time.sleep(10)
            ans = getResult(text)
            if filename != None:
                file_ans(ans)
            return ans
        else:
            print('Unknown code in IBM',err.code)
            print('interacting with IBM.getResult...')
            code.interact(local=dict(globals(),**locals()))
    if status_code == 200:
        ans = profile.result
        if filename != None:
            file_ans(ans)
        return ans

def getBig5FromText(text,filename=None):
    """
    (getBig5FromText text) -> {<big 5 trait>:<raw_score>} This function takes in text and turns it into a dict of
    the Big5 variables.

    | kargs:
    | filename - An optional argument that will save the raw IBM output as a json to the designated file

    ex.
    getBig5FromText('This is some text' x 100) -> {'openness':0.72,'conscientiousness':0.43, blah...}
    """
    raw = getResult(text,filename=filename)
    if raw == None:
        return None
    ans = {}
    persona = raw['personality']
    ans['openness'] = [item for item in persona if item['name'] == 'Openness'][0]['raw_score']
    ans['conscientiousness'] = [item for item in persona if item['name'] == 'Conscientiousness'][0]['raw_score']
    ans['extraversion'] = [item for item in persona if item['name'] == 'Extraversion'][0]['raw_score']
    ans['agreeableness'] = [item for item in persona if item['name'] == 'Agreeableness'][0]['raw_score']
    ans['neuroticism'] = [item for item in persona if item['name'] == 'Emotional range'][0]['raw_score']
    return ans

if __name__ == '__main__':
    print('testing')
    TESTING = os.path.join(HOME,'Library','Testing')
    LOG = os.path.join(HOME,'Library','Logs')
    filepath = os.path.join(LOG,'test1.json')
    exp_ans = {'openness': 0.7648753447240669, 'conscientiousness': 0.5955417708434109, 'extraversion': 0.5398801872266267, 'agreeableness': 0.7218578735880791, 'neuroticism': 0.4782258167889502}
    exp_file = os.path.join(TESTING,'test1.json')
    exp_file_ans = json.load(open(exp_file,'r'))
    ans = getBig5FromText('hi how are you '*100,filename=filepath)
    file_ans = json.load(open(filepath,'r'))
    print('Test1 PASSED') if exp_ans == ans and exp_file_ans == file_ans else print('Test1 Failed got',ans,'instead of',exp_ans)
