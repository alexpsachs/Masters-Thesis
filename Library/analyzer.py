"""
The purpose of this module is to centralze of the the stats and plotting
functionality for reusability
"""
from bisect import bisect_left
from pyexcel_ods3 import read_data
import pandas as pd
import code
import sklearn
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import svm
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import sys
import code
import xgboost as xgb

TOP = os.path.abspath(os.path.join(__file__,'../../'))
LIB = os.path.join(TOP,'Library')
sys.path.append(LIB)
import logger
def log(*args,pre=None):
    logger.log(*args,pre='analyzer.py' if pre==None else 'analyzer.py'+pre)
debug = False
print('TOP',TOP) if debug else None
print('LIB',LIB) if debug else None

# CLASSES
class Xgboost_wrapper:
    """
    Wrapper for the xgboost model
    """
    def __init__(self):
        self.pre = '.Xgboost_wrapper'
        self.model = None

    def fit(self,X,Y):
        """
        """
        pre = self.pre + '.fit'
        dtrain = xgb.DMatrix(X.values,Y.values)
        # specify parameters via map
        param = {'max_depth':5, 'eta':1, 'silent':0, 'objective':'binary:logistic' }
        num_round = 10
        self.model = xgb.train(param, dtrain, num_round)
        log('model is now trained',pre=pre)

    def predict(self,X_test):
        """
        """
        pre = self.pre + '.predict'
        log('start',pre=pre)
        dtest = xgb.DMatrix(X_test.values)
        preds = self.model.predict(dtest)
        ans = [1 if y >= 0.5 else 0 for y in preds]
        self.model.dump_model('/home/alex/test.txt',dump_format='text')
        log('returned',pre=pre)
        return preds

    def output_tree(self):
        self.model.dump_model('/home/alex/test.txt',dump_format='text')

# FUNCTIONS
def assess(df,y_label,model_type='Linear',test_data=None):
    """
    (assess DataFrame str) This function takes in a dataframe of all the data, takes out
    the column addressed as y_label and then does a linear regression to use the x values to predict
    y_label and returns the stats (dict) and the predictions (list)

    | kargs:
    | model_type:   (anyof 'Linear' 'Logistic' 'Xgboost' 'SVM') this determines which model will be used to assess the datapoints
    | test_data: DataFrame:     This determines what data the predictions will be for, if None then the df will be used

    | ex.
    | df_test = pd.DataFrame(data={'x':[1,2,3,4],'y':[2,4,6,7]})
    | doRegression(df_test,'y',model_type='Linear') -> {'R^2': 0.9796610169491525}, [[1, 2.2000000000000006], [2, 3.9000000000000004], [3, 5.6000000000000005], [4, 7.3]]

    Note: Logistic defaults to one vs all classification

    """
    pre='.assess'
    log('start',pre=pre)
    log('regression',model_type,pre=pre)
    X = df.drop(columns=[y_label])
    Y = df[y_label]
    # train the model
    regr = linear_model.LinearRegression() if model_type == 'Linear' else\
            linear_model.LogisticRegression(solver='liblinear',multi_class='ovr') if model_type == 'Logistic' else\
            sklearn.svm.SVC(kernel='rbf') if model_type == 'SVM' else\
            Xgboost_wrapper() if model_type == 'Xgboost' else\
            None
    # documentation:
    # logistic: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    if regr == None:
        print('ERROR unknown model_type',model_type)
        return
    regr.fit(X,Y)
    if model_type == 'Xgboost':
        regr.output_tree()
    # test the model
    categories = list({y for y in Y.values})
    X_test = test_data.drop(columns=[y_label]) if type(test_data) == pd.DataFrame else X
    Y_test = test_data[y_label] if type(test_data) == pd.DataFrame else Y
    predY = regr.predict(X_test)
    # discretize the predictions
    if model_type in ['Linear','Xgboost']:
        for i,y in enumerate(predY):
            dist = sys.maxsize
            closest = None
            for cat in categories:
                diff = abs(y-cat)
                if diff < dist:
                    closest = cat
                    dist = diff
            predY[i] = closest
    # compare the prediction to the expected values
    log('Y_test',Y_test,pre=pre)
    log('predY',predY,pre=pre)
    stats = {
            'confusion':metrics.confusion_matrix(Y_test.values,predY).tolist(),
            'f1':metrics.f1_score(Y_test,predY)
            }
    # prepare the output
    pred = [[*entry[0],entry[1]] for entry in zip(X_test.values,predY)]
    return stats,pred

def plotBestFit(actual, predictions, filename=None):
    """
    (plotBestFit list list) Takes in the acutal datapoints and the predicted datapoints and then
    plots the two against each other. This will display the plot if filename is not provided.

    kargs:
    filename: Saves the fig to this file

    | ex.
    | actual = [[1,2],[2,4],[3,6],[4,7]]
    | predictions = [[1,2.2],[2,3.9],[3,5.6],[4,7.3]]
    | plotBestFit(actual, predictions, filename='./test.png') -> A plt that shows a line of best fit and is saved to ./test.png

    """
    # normalize the data
    pre = '.plotBestFit'
    log(pre=pre)
    log('start',pre=pre)
    log('filename',filename,pre=pre)
    log('actual type',type(actual),pre=pre)
    log('predictions type',type(predictions),pre=pre)
    # normalize the data
    if type(actual) == pd.DataFrame:
        actual = actual.values
    if type(predictions) == pd.DataFrame:
        predictions = predictions.values
    log('actual',actual,pre=pre)
    log('predictions',predictions,pre=pre)
    actual1 = [tuple(x) for x in actual]
    predictions1 = [tuple(x) for x in predictions]
    # if filename == '/home/alex/Thesis/Library/test2.png':
    #     code.interact(local=dict(globals(),**locals()))
    actual1.sort()
    predictions1.sort()
    actualXs = [e[0] for e in actual1]
    actualYs = [e[1] for e in actual1]
    predXs = [e[0] for e in predictions1]
    predYs = [e[1] for e in predictions1]
    log('actualXs,actualYs',actualXs,actualYs,pre=pre)
    log('predXs,predYs',predXs,predYs,pre=pre)
    # plot the values
    plt.plot(actualXs,actualYs,'ro',alpha=0.5)
    plt.plot(predXs,predYs,'b-',alpha=0.5)
    if filename == None:
        plt.show()
        plt.close()
    else:
        plt.savefig(filename)
        plt.close()
    log('end',pre=pre)

if __name__ == '__main__':
    # testing
    logger.deleteLogs()

    # 1. plotBestFit
    actual = [[1,2],[2,4],[3,6],[4,7]]
    predictions = [[1,2.2],[2,3.9],[3,5.6],[4,7.3]]
    plotBestFit(actual,predictions,os.path.join(LIB,'test1.png'))
    print('Processed Test 1')

    # 2. doLinearRegression 
    df_test = pd.DataFrame(data={'x':[1,2,3,4],'y':[1,1,0,0]})
    exp_conf = np.array([[2, 0],[0, 2]])
    exp_f1 = 1.0
    exp_pred = [[1, 1.0], [2, 1.0], [3, 0.0], [4, 0.0]]
    filepath = os.path.join(LIB,'test2.png')
    ans,pred = assess(df_test,'y',model_type='Linear')
    print('Passed 2') if np.array_equal(exp_conf,ans['confusion'])\
            and exp_f1 == ans['f1']\
            and exp_pred == pred\
            else print('Failed 2 got')

    # 3. logistic regression
    df_test = pd.DataFrame(data={'x':[1,2,3,4],'y':[0,0,1,1]})
    ans,pred = assess(df_test,'y',model_type='Logistic')
    exp_conf = np.array([[1,1],[0,2]])
    exp_f1 = 0.8
    exp_pred = [[1, 0], [2, 1], [3, 1], [4, 1]]
    print('Passed 3') if np.array_equal(exp_conf,ans['confusion']) and exp_f1 == ans['f1'] and exp_pred == pred else print('Failed 3 got',(ans,pred),'instead of',exp_ans)

    # 4. xgboost
    # seems to work on the actual dataset so... whatever I guess?

    # 5. svm
    df_test = pd.DataFrame(data={'x':[1,2,3,4],'y':[0,0,1,1]})
    ans,pred = assess(df_test,'y',model_type='SVM')
    exp_conf = np.array([[1,1],[0,2]])
    exp_f1 = 0.8
    exp_pred = [[1, 0], [2, 1], [3, 1], [4, 1]]
    print('Passed 4') if np.array_equal(exp_conf,ans['confusion']) and exp_f1 == ans['f1'] and exp_pred == pred else print('Failed $ got',(ans,pred),'instead of',({'confusion':exp_conf,'f1':exp_f1},pred))


