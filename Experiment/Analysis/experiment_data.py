"""
The purpose of this script is to combine the relevant data in tables:
    ./SYMLOG_metrics/<experiment>/SYMLOG_metrics.ods
    ./ESEM - Final.ods
so that there can be one datasource for the analysis into
    ./experiment_data/<experiment>/experiment_data_folds.ods
    ./experiment_data/<experiment>/experiment_data_test.ods
"""
import code
import sys
import os
TOP = os.path.abspath(os.path.join(__file__,'../../../'))
LIB = os.path.join(TOP,'Library')
sys.path.append(LIB)
import logger
from pyexcel_ods3 import save_data, read_data
from collections import OrderedDict

def log(*args,pre=None):
    logger.log(*args,pre='experiment_data.py' if pre==None else 'experiment_data.py.'+pre)

# constants
SYMLOG_METRICS_DIR = os.path.join(TOP,'Experiment','Analysis','SYMLOG_metrics')
ESEM_path = os.path.join(TOP,'Experiment','Analysis','ESEM - Final.ods')
OUT_DIR = os.path.join(TOP,'Experiment','Analysis','experiment_data')
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

# functions
def aggregateData(exp_name,pre=''):
    pre = pre + '.aggregateData'
    log('started',pre=pre)
    # congregate the SYMLOG metrics
    SYMLOG_metrics_path = os.path.join(SYMLOG_METRICS_DIR,exp_name,'SYMLOG_metrics.ods')
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
    # folds data
    esem_data = read_data(ESEM_path)
    esem_model = esem_data['Model']
    esem_prec = esem_data['Precision']
    esem_rec = esem_data['Recall']

    folds_dict = {}
    testing_dict = {}
    prec_dict = {}
    rec_dict = {}

    headers = esem_model[0]
    for row in esem_model[1:]:
        if len(row) != 0:
            name = row[headers.index('Repository')]
            status = row[headers.index('Archived')]
            active = 'active' if status=='Active' else 'inactive' #False is if it is 'Archived' or 'FSE' (which in the paper the say is inactive)
            folds_dict[name] = {'status':active}
            log('processed',name,pre=pre)
        else:
            log('error, zero length row',row,pre=pre)
    headers = esem_prec[0]
    for row in esem_prec[1:]:
        if len(row) != 0:
            name = row[headers.index('Repository')]
            result = row[headers.index('Result')]
            if result not in ['TP','FP']:
                print('unknown result',result)
            status = 'inactive' if result in ['TP','FP'] else 'active'
            prec_dict[name] = {'status':status}
    headers = esem_rec[0]
    for row in esem_rec[1:]:
        if len(row) != 0:
            name = row[headers.index('Repository')]
            classification = row[headers.index('Classification')]
            if classification not in ['TP','FN']:
                print('unknown classification',classification)
            status = 'inactive' if classification == 'TP' else 'active' if classification == 'FN' else 'unknown?'
            rec_dict[name] = {'status':status}
    testing_dict.update(prec_dict)
    testing_dict.update(rec_dict)

    # output the data
    out_data = OrderedDict()
    out_path = os.path.join(OUT_DIR,exp_name,'experiment_data.ods')
    for label,d in zip(['Folds','Testing','Precision','Recall'],[folds_dict,testing_dict,prec_dict,rec_dict]):
        # get the common repos between this and symlog
        common_repos = []
        symlog_only = []
        for key in d:
            if key in symlog_dict:
                common_repos.append(key)
            else:
                symlog_only.append(key)
        print(len(symlog_only),'repos not in symlog_dict for d',label)
        if not os.path.exists(os.path.join(OUT_DIR,exp_name)):
            os.mkdir(os.path.join(OUT_DIR,exp_name))
        symlog_metrics = list(list(symlog_dict.values())[0].keys())
        header = ['reponame','esem_status',*symlog_metrics]
        data = []
        for reponame in common_repos:
            esem_status = d[reponame]['status']
            symlog_scores = [symlog_dict[reponame][metric] for metric in symlog_metrics]
            lst = [reponame,esem_status]+symlog_scores
            data.append(lst)

        out_data.update({label:[header,*data]})
    # code.interact(local=dict(globals(),**locals()))
    save_data(out_path, out_data)

    print('done',exp_name)

if __name__ == '__main__':
    logger.deleteLogs()
    pre='experiment_data.py'
    # aggregate data for experiments
    aggregateData('old',pre=pre)
    aggregateData('reorient',pre=pre)


