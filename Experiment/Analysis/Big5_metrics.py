"""
The purpose of this script is to convert the data in ../Data/ProcessedText_sym into the final metrics
that will be used in the final analysis
"""
import os
import sys
import code
TOP = os.path.abspath(os.path.join(__file__,'../../../'))
LIB = os.path.join(TOP,'Library')
sys.path.append(LIB)
import SYMLOG
import json
import code
import math
import logger
from collections import OrderedDict
from pyexcel_ods3 import save_data, get_data
from multiprocessing import Pool
import multiprocessing as mp

# constants
ANALYSIS = os.path.join(TOP,'Experiment','Analysis')
OUTDIR = os.path.join(ANALYSIS,'Big5_metrics')
INDIR = os.path.join(TOP,'Experiment','Data','ProcessedText_Big5')
def log(*args,pre=None):
    logger.log(*args,pre='Big5_metrics.py' if pre==None else 'Big5_metrics.py.'+pre)
# Classes
class Big5Group:
    def __init__(self,data):
        self.data = data
        self.big5_attributes = [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism"]
        self.num_of_people = len(list(self.data.keys()))
    
    def get_average(self):
        # returns a 5 attribute dict of the group averages
        ans = {}
        for attribute in self.big5_attributes:
            tot = sum([d[attribute] for d in self.data.values()])
            avg = tot/self.num_of_people
            ans[attribute] = avg
        return ans

    def get_percentiles(self):
        # returns a 5 attribute dict of 5 percentiles in each category
        pre = 'get_percentiles'
        log('start',pre=pre)
        n = 5
        ans = {}
        for attribute in self.big5_attributes:
            all_vals = [d[attribute] for d in self.data.values()]
            all_vals.sort()
            interval = len(all_vals)//(n-1)
            percentiles = [all_vals[i*interval] for i in range(n-1)]
            percentiles.append(all_vals[-1])
            log('percentiles are',percentiles,'for',all_vals,pre=pre)
            ans[attribute] = percentiles
        log('returned',ans,pre=pre)
        return ans


# Functions
def calcMetrics(filename,reorient=False):
    """
    (calcMetrics filename) Here filename is a path to a json that contains all of the symlog attributes for
    all the people in a project. This
    function returns a dictionary of the form {name: {metric_name: metric_value}} where the metric_name can be:
    *openness
    *conscientiousness
    *extraversion
    *agreeableness
    *neuroticism

    OR None if the group is empty

    """
    pre='calcMetrics'
    log('start',pre=pre)
    try:
        data = json.load(open(filename,'r'))
    except:
        print('error at filename',filename)
    log('num of people',len(list(data.keys())),pre=pre)
    if len(list(data.keys())) in [0,1]:
        log('zero group contingency triggered',pre=pre)
        return None
    group = Big5Group(data)
    avg_metric = {'average_'+key:val for key,val in group.get_average().items()}
    perc_metric = {key+str(i):p for key,val in group.get_percentiles().items() for i,p in enumerate(val)}
    metrics = {}
    # metrics.update(avg_metric)
    metrics.update(perc_metric)
    return metrics

def run(exp_name,reorient=False,num_threads=4):
    pre='run'
    log('running',exp_name,pre=pre)
    filenames = os.listdir(INDIR)
    data = {} # {reponame:{metric_name: metric_value}}
    ods_dir = os.path.join(OUTDIR,exp_name)
    if not os.path.exists(ods_dir):
        os.mkdir(ods_dir)
    plots_dir = os.path.join(OUTDIR,exp_name,'plots')
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    # allocate the inputs for multithreading
    filenames_lsts = []
    for i,filename in enumerate(filenames):
        index = i % num_threads
        if index == len(filenames_lsts):
            filenames_lsts.append([])
        filenames_lsts[index].append(filename)
    inputs = [[lst,i,ods_dir] for i,lst in enumerate(filenames_lsts)]
    # run the multithreadable function
    pool = Pool(processes=num_threads)
    pool.map(run_helper,inputs)

    # aggregate the data together
    filepaths = [os.path.join(ods_dir,'Big5_metrics_{0}.ods'.format(i)) for i in range(num_threads)]
    all_data = get_data(filepaths[0])
    key = list(all_data.keys())[0]
    for filepath in filepaths[1:]:
        new_data = get_data(filepath)
        all_data[key].extend(new_data[key][1:]) # need to make sure to exclude the header of each file
    out_path = os.path.join(ods_dir,'Big5_metrics.ods')
    save_data(out_path,all_data)
    log('done the run',pre=pre)
    print('done',exp_name)

# create the function to multithread
def run_helper(args):
    lst,thread,ods_dir = args
    pre = 'run'+str(thread)
    data = {}
    drop_count = 0
    for filename in lst:
        log('Processing',filename,pre=pre)
        reponame = filename[:-5]
        inpath = os.path.join(INDIR,filename)
        metrics = calcMetrics(inpath)
        if metrics != None: # None indicates an empty group (or a group with ony 1 person)
            data[reponame] = metrics
        else:
            log('Dropped',filename,pre=pre)
            drop_count += 1
    # output the aggregated metrics
    metric_labels = list(list(data.values())[0].keys())
    symlog_header = ['repository']+metric_labels
    symlog_data = []
    for name,metrics in data.items():
        lst = [metrics[label] for label in metric_labels]
        symlog_data.append([name]+lst)
    out_data = OrderedDict()
    out_data.update({"Big5": [symlog_header, *symlog_data]})
    save_data(os.path.join(ods_dir,'Big5_metrics_{0}.ods'.format(thread)), out_data)
    log(drop_count,'repositories were dropped',pre=pre)

# Main script
if __name__ == '__main__':
    logger.deleteLogs()
    log('Big5_metrics start')
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)
    run('old',num_threads=3)
    run('reorient',num_threads=3)
