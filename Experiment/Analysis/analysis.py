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

TOP = os.path.abspath(os.path.join(__file__,'../../../'))
LIB = os.path.join(TOP,'Library')
sys.path.append(LIB)
import logger
def log(*args,pre=None):
    logger.log(*args,pre='analysis.py' if pre==None else 'analysis.py'+pre)
debug = False

INDIR = os.path.join(TOP,'Experiment','Analysis','experiment_data')
OUTDIR = os.path.join(TOP,'Experiment','Analysis','analysis')

# functions
def analyzeExperiment(exp_name,pre=''):
    global debug
    pre = pre + '.analyzeExperiment'
    # read in the data
    log('analyzing',exp_name,pre=pre)
    base = os.path.join(INDIR,exp_name)
    if not os.path.exists(base):
        os.mkdir(base)
        log('made directory',base,pre=pre)
    in_path = os.path.join(base,'experiment_data.ods')
    ods_data = read_data(in_path)['Data']
    headers = ods_data[0]
    data_dict = {label:[] for label in headers}
    for row in ods_data[1:]:
        for i,label in enumerate(headers):
            data_dict[label].append(row[i])
    df = pandas.DataFrame(data=data_dict)

    # transform the data by normalizing and binarizing esem_status
    df.loc[df['esem_status'] == 'active', 'esem_status'] = 1
    df.loc[df['esem_status'] == 'inactive', 'esem_status'] = 0
    df = df.drop(columns=['reponame'])
    columns = df.columns.values
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pandas.DataFrame(x_scaled)
    df.columns = columns

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
    p('opp_prop',pre=pre)

    # do the anlaysis
    # linear regression
    log('doing linear regression for',exp_name,pre=pre)
    out_txt,predictions = doLinearRegression(df,'esem_status')
    out_txt_path = os.path.join(OUTDIR,exp_name,'linear_regression.txt')
    plotBestFit(df['esem_status'],predictions)
    out_plot_path = os.path.join(OUTDIR,exp_name,'linear_regression.png')
    plt.savefig(out_plot_path)
    plt.close()
    with open(out_txt_path,'w') as f:
        f.write(out_txt)
    # logistic regression
    log('doing logistic regression for',exp_name,pre=pre)
    out_txt,predictions = doLogisticRegression(df,'esem_status')
    out_txt_path = os.path.join(OUTDIR,exp_name,'logistic_regression.txt')
    plotBestFit(df['esem_status'],predictions)
    out_plot_path = os.path.join(OUTDIR,exp_name,'logistic_regression.png')
    plt.savefig(out_plot_path)
    plt.close()
    with open(out_txt_path,'w') as f:
        f.write(out_txt)
    # svm
    log('doing an arbitrary svm',exp_name,pre=pre)
    out_txt,predictions = doSVM(df,'esem_status')
    out_txt += '   blah'
    out_txt_path = os.path.join(OUTDIR,exp_name,'svm.txt')
    plotBestFit(df['esem_status'],predictions)
    out_plot_path = os.path.join(OUTDIR,exp_name,'svm.png')
    plt.savefig(out_plot_path)
    plt.close()
    with open(out_txt_path,'w') as f:
        f.write(out_txt)
    # xgboost stuff
    doXgboost(df,'esem_status')

def doLinearRegression(df,y_label):
    # do multiple linear regression
    new_df = df.copy()
    X = new_df.drop(columns=[y_label])
    Y = new_df[y_label]
    X = sm.add_constant(X)
    model = sm.OLS(Y,X).fit()
    return model.summary().__repr__(), model.predict(X)

def doLogisticRegression(df,y_label):
    pre = '.doLogisticRegression'
    log('starts',pre=pre)
    X = df.drop(columns=[y_label])
    # check if any columns have a single value only (if they do, drop them and print out a warning)
    to_drop = []
    for column in X.columns:
        val_lst = X[column].values
        all_same = True
        for i,val in enumerate(val_lst):
            if i == 0:
                continue
            else:
                if val != val_lst[i-1]:
                    all_same = False
                    break
        if all_same:
            print('logistic regression dropping column',column,'because all the values are the same')
            log('drop column',column,pre=pre)
            to_drop.append(column)
    X = X.drop(columns = to_drop)
    Y = df['esem_status']
    log('X and Y allocated',pre=pre)
    logit_model=sm.Logit(Y,X)
    log('logit_model created',pre=pre)
    model=logit_model.fit()
    log('finished',pre=pre)
    ans = model.summary2().__repr__(), model.predict(X)
    log('ans',ans,pre=pre)
    return ans

def doSVM(df,y_label):
    # do multiple linear regression
    X = df.drop(columns=[y_label])
    Y = df[y_label]
    # X = sm.add_constant(X)
    clf = svm.SVC(kernel='rbf') # radial basis function (i.e. gaussian)
    clf.fit(X,Y)
    predictions = clf.predict(X)
    r2 = clf.score(X.values,Y)
    return 'R^2: '+r2.__repr__(), predictions

def doXgboost(df,y_label):
    # # do multiple linear regression
    X = df.drop(columns=[y_label])
    Y = df[y_label]
    # # X = sm.add_constant(X)
    # clf = svm.SVC(kernel='rbf') # radial basis function (i.e. gaussian)
    # clf.fit(X,Y)
    # predictions = clf.predict(X)
    # r2 = clf.score(X.values,Y)
    # return 'R^2: '+r2.__repr__(), predictions

    # read in data
    dtrain = xgb.DMatrix(X.values,Y.values)
    # dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
    # specify parameters via map
    param = {'max_depth':5, 'eta':1, 'silent':0, 'objective':'binary:logistic' }
    num_round = 10
    bst = xgb.train(param, dtrain, num_round)
    # make prediction
    preds = bst.predict(dtrain)
    code.interact(local=dict(globals(),**locals()))


def plotBestFit(actual, predictions):
    pre = 'plotBestFit'
    log('start',pre=pre)
    log('actual',actual.values,pre=pre)
    log('predictions',predictions,pre=pre)
    ordered = [(item[0],item[1]) for item in zip(actual.values,predictions)]
    ordered.sort()
    log('ordered',ordered,pre=pre)
    plt.plot([i for i,item in enumerate(ordered)],[item[0] for i,item in enumerate(ordered)],'b-',alpha=0.5) # actual values
    plt.plot([i for i,item in enumerate(ordered)],[item[1] for i,item in enumerate(ordered)],'r-',alpha=0.5) # predicted values
    log('end',pre=pre)

if __name__ == '__main__':
    logger.deleteLogs()
    log('starting')
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)
    experiments = ['old','reorient']
    for experiment in experiments:
        if experiment == 'reorient':
            debug = True
        analyzeExperiment(experiment)
