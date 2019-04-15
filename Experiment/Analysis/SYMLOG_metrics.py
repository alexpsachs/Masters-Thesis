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
OUTDIR = os.path.join(ANALYSIS,'SYMLOG_metrics')
INDIR = os.path.join(TOP,'Experiment','Data','ProcessedText_Sym')
def log(*args,pre=None):
    logger.log(*args,pre='SYMLOG_metrics.py' if pre==None else 'SYMLOG_metrics.py.'+pre)

# Functions
def calcSymMetrics(filename,reorient=False):
    """
    (calcSymMetrics filename) Here filename is a path to a json that contains all of the symlog attributes for
    all the people in a project. This
    function returns a dictionary of the form {name: {metric_name: metric_value}} where the metric_name can be:
    *dissidence: sum of squared deviation from SCG optimum
    *upf_corr: The intercorrelation between the U,P, and F poles
    *pf_prop: The proportion of individuals that end up in the PF region
    *opp_prop: The proportion of individuals that end up in the Opposite Circle
    *rot_regret: The amount of rotational deviation from Bale's "optimal" placement (NB<->PF) (1 indicates pi radian deviation from optimal)

    OR None if the group is empty

    kwargs:
        reorient: if True, this will call reorientCompass() before doing the calculations
    """
    pre='calcSymMetrics'
    log('start',pre=pre)
    personalities = json.load(open(filename,'r'))
    log('personality length',len(list(personalities.keys())),pre=pre)
    if len(list(personalities.keys())) in [0,1]:
        log('zero group contingency triggered',pre=pre)
        return None
    symPlot = SYMLOG.SYMLOGPlot(personalities,compassMethod='regression')
    if reorient:
        symPlot.reorientCompass()
    # symBest = SYMLOG.SYMLOGPlot(personalities,compassMethod='PF') # turns out this actually isn't used for anything
    num_people = symPlot.personalities.shape[0]
    metrics = {
            'dissidence':symPlot.getSCGDeviation(),
            'upf_corr':symPlot.getUPFCorrelation(),
            'pf_prop':len([attributes for attributes in symPlot.personalities[['p_n','f_b']].values
                if attributes[0] > 0 and attributes[1] > 0])/num_people,
            'opp_prop':len(list(symPlot.getMembersByRegion('opp')))/num_people,
            'rot_regret':abs(symPlot.angle - math.pi/4)/math.pi
            }
    return metrics

def plotPersonas(filename,outpath,reorient=False):
    personalities = json.load(open(filename,'r'))
    symPlot = SYMLOG.SYMLOGPlot(personalities,compassMethod='regression')
    if reorient:
        symPlot.reorientCompass()
    symPlot.draw(outpath)

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
    inputs = [[lst,i,plots_dir,ods_dir,reorient] for i,lst in enumerate(filenames_lsts)]
    # run the multithreadable function
    pool = Pool(processes=num_threads)
    pool.map(run_helper,inputs)

    # aggregate the data together
    filepaths = [os.path.join(ods_dir,'SYMLOG_metrics_{0}.ods'.format(i)) for i in range(num_threads)]
    all_data = get_data(filepaths[0])
    key = list(all_data.keys())[0]
    for filepath in filepaths[1:]:
        new_data = get_data(filepath)
        all_data[key].extend(new_data[key])
    out_path = os.path.join(ods_dir,'SYMLOG_metrics.ods')
    save_data(out_path,all_data)
    log('done the run',pre=pre)
    print('done',exp_name)

# create the function to multithread
def run_helper(args):
    lst,thread,plots_dir,ods_dir,reorient = args
    pre = 'run'+str(thread)
    data = {}
    drop_count = 0
    for filename in lst:
        log('Processing',filename,pre=pre)
        reponame = filename[:-5]
        inpath = os.path.join(INDIR,filename)
        png_outpath = os.path.join(plots_dir,reponame+'.png')
        metrics = calcSymMetrics(inpath,reorient)
        if metrics != None: # None indicates an empty group (or a group with ony 1 person)
            data[reponame] = metrics
            plotPersonas(inpath,png_outpath,reorient=reorient)
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
    out_data.update({"SYMLOG": [symlog_header, *symlog_data]})
    save_data(os.path.join(ods_dir,'SYMLOG_metrics_{0}.ods'.format(thread)), out_data)
    log(drop_count,'repositories were dropped',pre=pre)

# Main script
if __name__ == '__main__':
    logger.deleteLogs()
    log('SYMLOG_metrics start')
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)
    run('old',num_threads=3)
    run('reorient',reorient=True,num_threads=3)
