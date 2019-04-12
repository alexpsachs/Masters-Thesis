"""
The purpose of this script is to convert the data in ../Data/ProcessedText_sym into the final metrics
that will be used in the final analysis
"""
import sys
LIB = '/home/a2sachs/Documents/Library'
sys.path.append(LIB)
import SYMLOG
import json
import code
import math
import logger
import os
from collections import OrderedDict
from pyexcel_ods3 import save_data

# constants
OUTDIR = '/home/a2sachs/Documents/Experiment2.2/Analysis/SYMLOG_metrics'
INDIR = '/home/a2sachs/Documents/Experiment2.2/Data/ProcessedText_Sym'
ANALYSIS = '/home/a2sachs/Documents/Experiment2.2/Analysis'
def log(*args,pre=None):
    logger.log(*args,pre='SYMLOG_metrics.py' if pre==None else 'SYMLOG_metrics.py.'+pre)

# Functions
def calcSymMetrics(filename):
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
    """
    pre='calcSymMetrics'
    log('start',pre=pre)
    personalities = json.load(open(filename,'r'))
    log('personality length',len(list(personalities.keys())),pre=pre)
    if len(list(personalities.keys())) in [0,1]:
        log('zero group contingency triggered',pre=pre)
        return None
    symPlot = SYMLOG.SYMLOGPlot(personalities,compassMethod='regression')
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

def plotPersonas(filename,outpath):
    personalities = json.load(open(filename,'r'))
    symPlot = SYMLOG.SYMLOGPlot(personalities,compassMethod='regression')
    symPlot.reorientCompass()
    symPlot.draw(outpath)

# Main script
if __name__ == '__main__':
    logger.deleteLogs()
    log('SYMLOG_metrics start')
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)
    filenames = os.listdir(INDIR)
    # filename = '1N3__Sn1per.json'
    data = {} # {reponame:{metric_name: metric_value}}
    for filename in filenames:
        log('Processing',filename)
        reponame = filename[:-5]
        inpath = os.path.join(INDIR,filename)
        # json_outpath = os.path.join(OUTDIR,filename)
        png_outpath = os.path.join(OUTDIR,reponame+'.png')
        metrics = calcSymMetrics(inpath)
        if metrics != None: # None indicates an empty group (or a group with ony 1 person)
            data[reponame] = metrics
            plotPersonas(inpath,png_outpath)
    # output the aggregated metrics
    metric_labels = list(list(data.values())[0].keys())
    symlog_header = ['repository']+metric_labels
    symlog_data = []
    for name,metrics in data.items():
        lst = [metrics[label] for label in metric_labels]
        symlog_data.append([name]+lst)
    out_data = OrderedDict()
    out_data.update({"SYMLOG": [symlog_header, *symlog_data]})
    save_data(os.path.join(ANALYSIS,'SYMLOG_metrics.ods'), out_data)


