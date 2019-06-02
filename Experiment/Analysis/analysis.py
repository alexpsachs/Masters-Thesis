"""
The purpose of this script is to analyze the data in 
./experiment_data.ods
"""
from pyexcel_ods3 import read_data
import pandas
import code
from sklearn import preprocessing
from sklearn import svm
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import sys
import code
import xgboost as xgb
import json

TOP = os.path.abspath(os.path.join(__file__,'../../../'))
LIB = os.path.join(TOP,'Library')
sys.path.append(LIB)
import analyzer
import logger
def log(*args,pre=None):
    logger.log(*args,pre='analysis.py' if pre==None else 'analysis.py'+pre)
debug = False

# constants
INDIR = os.path.join(TOP,'Experiment','Analysis','experiment_data')
OUTDIR = os.path.join(TOP,'Experiment','Analysis','analysis')

# config
config_include_test_set = False # whether of not to include the repositories in ESEM's "empirical validation" set (i.e. the ones NOT used to train the model)

# functions
def analyzeExperiment(exp_config,pre=''):
    exp_name = exp_config['name']
    # make sure the folder is there, if not create it
    directory = os.path.join(OUTDIR,exp_name)
    if not os.path.exists(directory):
        os.mkdir(directory)
    global debug
    pre = pre + '.analyzeExperiment'
    # read in the data
    log('analyzing',exp_name,pre=pre)
    base = os.path.join(INDIR,exp_config['indir'])
    if not os.path.exists(base):
        os.mkdir(base)
        log('made directory',base,pre=pre)
    in_path = os.path.join(base,'experiment_data.ods')

    ods_data = read_data(in_path)['Folds']
    headers = ods_data[0]
    data_dict = {label:[] for label in headers}
    for row in ods_data[1:]:
        for i,label in enumerate(headers):
            data_dict[label].append(row[i])
    df = pandas.DataFrame(data=data_dict)

    prec_df = None
    rec_df = None
    if config_include_test_set:
        ods_data = read_data(in_path)['Precision']
        headers = ods_data[0]
        data_dict = {label:[] for label in headers}
        for row in ods_data[1:]:
            for i,label in enumerate(headers):
                data_dict[label].append(row[i])
        prec_df = pandas.DataFrame(data=data_dict)
        prec_df = pandas.DataFrame(data=data_dict)

        ods_data = read_data(in_path)['Recall']
        headers = ods_data[0]
        data_dict = {label:[] for label in headers}
        for row in ods_data[1:]:
            for i,label in enumerate(headers):
                data_dict[label].append(row[i])
        rec_df = pandas.DataFrame(data=data_dict)

    # transform the data by normalizing and binarizing esem_status
    # PARADIGN SHIFT HERE
    active_num = 0 #1
    inactive_num = 1 #0
    df.loc[df['esem_status'] == 'active', 'esem_status'] = active_num
    df.loc[df['esem_status'] == 'inactive', 'esem_status'] = inactive_num
    df = df.drop(columns=['reponame'])
    columns = df.columns.values
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pandas.DataFrame(x_scaled)
    df.columns = columns

    if config_include_test_set:
        prec_df.loc[prec_df['esem_status'] == 'active', 'esem_status'] = active_num
        prec_df.loc[prec_df['esem_status'] == 'inactive', 'esem_status'] = inactive_num
        prec_df = prec_df.drop(columns=['reponame'])
        columns = prec_df.columns.values
        x = prec_df.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        prec_df = pandas.DataFrame(x_scaled)
        prec_df.columns = columns

        rec_df.loc[rec_df['esem_status'] == 'active', 'esem_status'] = active_num
        rec_df.loc[rec_df['esem_status'] == 'inactive', 'esem_status'] = inactive_num
        rec_df = rec_df.drop(columns=['reponame'])
        columns = rec_df.columns.values
        x = rec_df.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        rec_df = pandas.DataFrame(x_scaled)
        rec_df.columns = columns

    # plot the raw data
    def p(name,pre=''):
        pre = pre + '.p'
        log('p start for',name,pre=pre)
        # plot the points
        x_vals = df[name].values
        if len(x_vals) == 0:
            log('no x_vals in',name,pre=pre)
            return
        y_vals = df['esem_status'].values
        # plt.plot([1, 2, 3, 4],[2,4,6,8])
        plt.plot(range(len(y_vals)),y_vals,'bo',alpha=0.5)
        plt.plot(range(len(x_vals)),x_vals,'ro',alpha=0.5)
        plt.ylabel(name)
        # plt.show()
        base = os.path.join(OUTDIR,exp_name)
        if not os.path.exists(base):
            os.mkdir(base)
        base = os.path.join(base,'raw_data')
        if not os.path.exists(base):
            os.mkdir(base)
        filepath = os.path.join(base,name+'.png')
        plt.savefig(filepath)
        plt.close()
    p('pf_prop',pre=pre)
    p('dissidence',pre=pre)
    p('upf_corr',pre=pre)
    p('rot_regret',pre=pre)
    p('opp',pre=pre)

    # do the anlaysis
    all_precision,all_recall,all_thresholds,all_labels = [],[],[],[]
    def run_model(model_type,x_labels=None,prc_label=None):
        # if x_labels None, then use all, else use only x_labels as features
        log('doing {0} regression for'.format(model_type),exp_name,pre=pre)
        out_dict,data = analyzer.assess(df,'esem_status',model_type=model_type,x_labels=x_labels)
        best_threshold = out_dict['test']['best threshold']
        if config_include_test_set:
            prec_stats = analyzer.test(df,prec_df,'esem_status',model_type,include_only=['stats','precision'],
                    threshold=best_threshold,x_labels=x_labels)
            rec_stats = analyzer.test(df,rec_df,'esem_status',model_type,include_only=['stats','recall'],
                    x_labels=x_labels)
            out_dict['empirical precision'] = prec_stats
            out_dict['empirical recall'] = rec_stats
        name = model_type if prc_label == None else prc_label
        out_txt_path = os.path.join(OUTDIR,exp_name,'analysis_{0}_model.json'.format(name))
        with open(out_txt_path,'w') as f:
            json.dump(out_dict,f,indent=4)
        all_precision.append(data['precision'])
        all_recall.append(data['recall'])
        all_thresholds.append(data['thresholds'])
        all_labels.append(model_type if prc_label == None else prc_label)
    
    for prc_label in exp_config['models']:
        x_labels = exp_config['models'][prc_label]['x_labels']
        model = exp_config['models'][prc_label]['model']
        run_model(model,x_labels=x_labels,prc_label=prc_label)

    # run_model('Linear')
    # run_model('Logistic')
    # run_model('Xgboost')
    # run_model('SVM')
    # # run_model('SVM',x_labels=symlog_metrics,prc_label='SVM: sym only')
    # # run_model('SVM',x_labels=big5_metrics,prc_label='SVM: big5 only')
    # run_model('Neural')
    
    # draw the roc curves
    precision_recall_path = os.path.join(OUTDIR,exp_name,'analysis_prc.png')
    precision_recall_analysis = os.path.join(OUTDIR,exp_name+'_analysis_prc.png')
    analyzer.plotPRC(all_precision,all_recall,all_thresholds,all_labels,precision_recall_path)
    analyzer.plotPRC(all_precision,all_recall,all_thresholds,all_labels,precision_recall_analysis)

if __name__ == '__main__':
    logger.deleteLogs()
    log('starting')
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)
    symlog_metrics = json.load(open(os.path.join(INDIR,'reorient','symlog_metrics.json'),'r'))
    big5_metrics = json.load(open(os.path.join(INDIR,'reorient','big5_metrics.json'),'r'))
    esem_train_metrics = json.load(open(os.path.join(INDIR,'reorient','esem_train_metrics.json'),'r'))
    all_metrics = symlog_metrics + big5_metrics + esem_train_metrics
    all_models = ['Linear','Logistic','Xgboost','SVM','Neural']

    old_config = {
            'name':'old',
            'indir':'old',
            'models':{m:{'x_labels':all_metrics,'model':m} for m in all_models},
            }
    log('old_config',old_config)
    reorient_config = {
            'name':'reorient',
            'indir':'reorient',
            'models':{m:{'x_labels':all_metrics,'model':m} for m in all_models},
            }
    log('reorient_config',reorient_config)
    sym_vs_big5_config = {
            'name':'sym_vs_big5',
            'indir':'reorient',
            'models':{
                'SVM':{'x_labels':all_metrics,'model':'SVM'},
                'SVM: sym only':{'x_labels':symlog_metrics,'model':'SVM'},
                'SVM: sym bales measures only':{'x_labels':[m for m in symlog_metrics if len(m) > 2],'model':'SVM'},
                'SVM: big5 only':{'x_labels':big5_metrics,'model':'SVM'},
                }
            }
    log('sym_vs_big5_config',sym_vs_big5_config)
    big5_config = {
            'name':'big5',
            'indir':'reorient',
            'models':{
                'SVM':{'x_labels':all_metrics,'model':'SVM'},
                'SVM: median only':{'x_labels':[m for m in big5_metrics if m[-1] == '2'],'model':'SVM'},
                'SVM: percentiles only':{'x_labels':big5_metrics,'model':'SVM'},
                }
            }
    log('big5_config',big5_config)
    sym_config = {
            'name':'sym',
            'indir':'reorient',
            'models':{
                'SVM':{'x_labels':all_metrics,'model':'SVM'},
                'SVM: Bales measures only':{'x_labels':[m for m in symlog_metrics if len(m) > 2],'model':'SVM'}, # as percentiles and regions are 2 or less chars
                'SVM: Bales regions only':{'x_labels':[m for m in symlog_metrics if m.isnumeric()],'model':'SVM'}, # as percentiles and regions are 2 or less chars
                'SVM: percentiles only':{'x_labels':[m for m in symlog_metrics if not m.isnumeric() and m[-1].isnumeric()],'model':'SVM'}, # as percentiles are letter + num
                }
            }
    log('sym_config',sym_config)
    personality_config = {
            'name':'personality',
            'indir':'reorient',
            'models':{
                'SVM':{'x_labels':all_metrics,'model':'SVM'},
                'SVM: esem metrics only':{'x_labels':esem_train_metrics,'model':'SVM'},
                'SVM: esem + big5 metrics':{'x_labels':esem_train_metrics + big5_metrics,'model':'SVM'},
                'SVM: esem + sym metrics':{'x_labels':esem_train_metrics + symlog_metrics,'model':'SVM'},
                }
            }
    log('personality_config',personality_config)

    experiments = [old_config, reorient_config, sym_vs_big5_config, big5_config, sym_config, personality_config]
    for experiment in experiments:
        analyzeExperiment(experiment)
    print('analysis complete')
