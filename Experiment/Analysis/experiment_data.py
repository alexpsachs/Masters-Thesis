"""
The purpose of this script is to combine the relevant data in tables:
    ./SYMLOG_metrics/<experiment>/SYMLOG_metrics.ods
    ./Big5_metrics/<experiment>/Big5_metrics.ods
    ./ESEM - Final.ods
    ../Data/trainningset.csv
so that there can be one datasource for the analysis into
    ./experiment_data/<experiment>/experiment_data.ods #Folds: the tab with the training data, Testing: ...
"""
import code
import sys
import os
import json
TOP = os.path.abspath(os.path.join(__file__,'../../../'))
LIB = os.path.join(TOP,'Library')
sys.path.append(LIB)
import logger
from pyexcel_ods3 import save_data, read_data
from pyexcel_io import get_data
from collections import OrderedDict

def log(*args,pre=None):
    logger.log(*args,pre='experiment_data.py' if pre==None else 'experiment_data.py.'+pre)

# constants
SYMLOG_METRICS_DIR = os.path.join(TOP,'Experiment','Analysis','SYMLOG_metrics')
BIG5_METRICS_DIR = os.path.join(TOP,'Experiment','Analysis','Big5_metrics')
ESEM_path = os.path.join(TOP,'Experiment','Analysis','ESEM - Final.ods')
ESEM_TRAIN_path = os.path.join(TOP,'Experiment','Data','trainingset.csv')
OUT_DIR = os.path.join(TOP,'Experiment','Analysis','experiment_data')
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
# config
config_include_test_set = False # whether of not to include the repositories in ESEM's "empirical validation" set (i.e. the ones NOT used to train the model)

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
    # congregate the big5 metrics
    Big5_metrics_path = os.path.join(BIG5_METRICS_DIR,exp_name,'Big5_metrics.ods')
    big5_data = read_data(Big5_metrics_path)['Big5']
    headers = big5_data[0]
    metrics = big5_data[0][1:]
    repo_index = big5_data[0].index('repository')
    big5_dict = {}
    for row in big5_data[1:]: #exclude the header
        reponame = row[repo_index].replace('__','/')
        value = {metric:row[headers.index(metric)] for metric in metrics}
        big5_dict[reponame] = value
    # congregate the esem training data
    ESEM_TRAIN_path
    esem_training_data = get_data(ESEM_TRAIN_path)['trainingset.csv']
    headers = esem_training_data[0]
    metrics = esem_training_data[0][1:]
    repo_index = headers.index('Repository')
    esem_train_dict = {}
    for row in esem_training_data[1:]: #exclude the header
        reponame = row[repo_index]
        value = {metric:row[headers.index(metric)] for metric in metrics}
        value['Failed'] = 'inactive' if value['Failed'] == 'Yes' else 'active' if value['Failed'] == 'No' else 'Error' # change to the status of the project to be consistent with ESEM - final.ods spreadsheet
        esem_train_dict[reponame] = value
    # congregate the ESEM labels
    # folds data
    esem_data = read_data(ESEM_path)
    esem_model = esem_data['Model']
    esem_prec = esem_data['Precision']
    esem_rec = esem_data['Recall']
    # remove empty lines
    esem_model = [x for x in esem_model if len(x) != 0]
    esem_prec = [x for x in esem_prec if len(x) != 0]
    esem_rec = [x for x in esem_rec if len(x) != 0]

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
            status = 'inactive' if result == 'TP' else 'active' if result == 'FP' else 'ERROR unknown'
            prec_dict[name] = {'status':status}
    headers = esem_rec[0]
    for row in esem_rec[1:]:
        if len(row) != 0:
            name = row[headers.index('Repository')]
            classification = row[headers.index('Classification')]
            if classification not in ['TP','FN']:
                print('unknown classification',classification)
            status = 'inactive' if classification in ['TP','FN'] else 'ERROR unknown'
            rec_dict[name] = {'status':status}
    testing_dict.update(prec_dict)
    testing_dict.update(rec_dict)

    # do some data quality checks on the data
    print('Performing data quality check')
    train_repos = list(esem_train_dict.keys())
    print(len(train_repos),'repos in trainingset.csv')
    esem_model_repos = list(folds_dict.keys())
    print(len(esem_model_repos),'repos in ESEM - final.ods.model')
    common = set(train_repos + esem_model_repos)
    print(len(common),'repos common to both...','GOOD' if len(common) == len(train_repos) == len(esem_model_repos) else 'BAD')
    agree_repos = [x for x in common if esem_train_dict[x]['Failed'] == folds_dict[x]['status']]
    print(len(agree_repos),'repos have the same status...','GOOD' if len(agree_repos) == len(common) else 'BAD')
    print('Data quality check complete')

    # output the data
    out_data = OrderedDict()
    out_path = os.path.join(OUT_DIR,exp_name,'experiment_data.ods')
    symlog_metrics = list(list(symlog_dict.values())[0].keys())
    big5_metrics = list(list(big5_dict.values())[0].keys())
    esem_train_metrics = list(list(esem_train_dict.values())[0].keys())
    print(len(esem_train_metrics),'metrics in esem_train_metrics')
    esem_train_metrics = [x for x in esem_train_metrics if x != 'Failed'] # exclude the column that has the status as this isn't actually one of the metrics
    print(len(esem_train_metrics),'ACTUAL metrics in esem_train_metrics')
    for label,d in zip(['Folds','Testing','Precision','Recall'],[folds_dict,testing_dict,prec_dict,rec_dict]):
        # get the common repos between this and symlog
        common_repos = []
        symlog_only = []
        for key in d:
            if config_include_test_set:
                if key in symlog_dict: # symlog_dict includes ALL repositories including those used in ESEM's "empirical validation"
                    common_repos.append(key)
                else:
                    symlog_only.append(key)
            else:
                if key in symlog_dict and key in folds_dict: # i.e. exclude the testing sets for now
                    common_repos.append(key)
                else:
                    symlog_only.append(key)
        print(len(symlog_only),'repos not in symlog_dict for d',label)
        if not os.path.exists(os.path.join(OUT_DIR,exp_name)):
            os.mkdir(os.path.join(OUT_DIR,exp_name))
        header = ['reponame','esem_status',*symlog_metrics,*big5_metrics,*esem_train_metrics]
        data = []
        for reponame in common_repos:
            esem_status = d[reponame]['status']
            symlog_scores = [symlog_dict[reponame][metric] for metric in symlog_metrics]
            big5_scores = [big5_dict[reponame][metric] for metric in big5_metrics]
            esem_train_scores = [esem_train_dict[reponame][metric] for metric in esem_train_metrics]
            lst = [reponame, esem_status] + symlog_scores + big5_scores + esem_train_scores
            data.append(lst)
        out_data.update({label:[header,*data]})
    save_data(out_path, out_data)
    # output the metrics for each category of features used
    json.dump(symlog_metrics, open(os.path.join(OUT_DIR,exp_name,'symlog_metrics.json'),'w'),indent=4)
    json.dump(big5_metrics, open(os.path.join(OUT_DIR,exp_name,'big5_metrics.json'),'w'),indent=4)
    json.dump(esem_train_metrics, open(os.path.join(OUT_DIR,exp_name,'esem_train_metrics.json'),'w'),indent=4)
    print('done',exp_name)

if __name__ == '__main__':
    logger.deleteLogs()
    pre='experiment_data.py'
    # aggregate data for experiments
    aggregateData('old',pre=pre)
    aggregateData('reorient',pre=pre)


