"""
The purpose of this script is to combine the relelant data in tables:
    ./SYMLOG_metrics.ods
    ./ESEM - Dataset.ods
so that there can be one datasource for the analysis
"""
import code
import sys
import os
LIB = os.path.abspath(os.path.join(__file__,'../../Library'))
print('LIB',LIB)
code.interact(local=locals())

sys.path.append(LIB)
import logger
from pyexcel_ods3 import save_data, read_data
from collections import OrderedDict

def log(*args,pre=None):
    logger.log(*args,pre='experiment_data.py' if pre==None else 'experiment_data.py.'+pre)

# constants
SYMLOG_metrics_path = '/home/a2sachs/Documents/Experiment2.2/Analysis/SYMLOG_metrics.ods'
ESEM_path = '/home/a2sachs/Documents/Experiment2.2/Analysis/ESEM - Dataset.ods'
out_path = '/home/a2sachs/Documents/Experiment2.2/Analysis/experiment_data.ods'

# functions

if __name__ == '__main__':
    logger.deleteLogs()
    pre='experiment_data.py'
    # aggregate data for experiments

    # congregate the SYMLOG metrics
    symlog_data = read_data(SYMLOG_metrics_path)['SYMLOG']
    headers = symlog_data[0]
    metrics = symlog_data[0][1:]
    repo_index = symlog_data[0].index('repository')
    symlog_dict = {}
    for row in symlog_data[1:]: #exclude the header
        reponame = row[repo_index].replace('__','/')
        value = {metric:row[headers.index(metric)] for metric in metrics}
        symlog_dict[reponame] = value

    # congregate the ESEM labels
    esem_data = read_data(ESEM_path)['Model']
    esem_dict = {}
    headers = esem_data[0]
    for row in esem_data[1:]:
        if len(row) != 0:
            name = row[headers.index('Repository')]
            status = row[headers.index('Archived')]
            active = 'active' if status=='Active' else 'inactive' #False is if it is 'Archived' or 'FSE' (which in the paper the say is inactive)
            esem_dict[name] = active
            log('processed',name,pre=pre)
        else:
            log('error, zero length row',row,pre=pre)

    # output the data
    common_repos = []
    symlog_only = []
    for key in esem_dict:
        if key in symlog_dict:
            common_repos.append(key)
        else:
            symlog_only.append(key)
    print(len(symlog_only),'repos not in symlog_dict')
    out_data = OrderedDict()
    symlog_metrics = list(list(symlog_dict.values())[0].keys())
    header = ['reponame','esem_status',*symlog_metrics]
    data = []
    for reponame in common_repos:
        esem_status = esem_dict[reponame]
        symlog_scores = [symlog_dict[reponame][metric] for metric in symlog_metrics]
        lst = [reponame,esem_status]+symlog_scores
        data.append(lst)
    # code.interact(local=locals())
    out_data.update({'Data':[header,*data]})
    save_data(out_path, out_data)

    print('done')

