"""
The purpose of this module is to centralze of the the stats and plotting
functionality for reusability
"""
from pyexcel_ods3 import read_data
import pandas as pd
import code
import sklearn
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

# FUNCTIONS
def doLinearRegression(df,y_label,plot_prediction=False,plot_prediction_filepath=None):
    """
    (doLinearRegression DataFrame str) This function takes in a dataframe of all the data, takes out
    the column addressed as y_label and then does a linear regression to use the x values to predict
    y_label and returns the stats (dict) and the predictions (list)

    | kargs:
    | plot_prediction: Determines if the prediction plot will be drawn
    | plot_prediction_filepath: Determines where this plot will be saved (will show if empty)

    | ex.
    | df_test = pd.DataFrame(data={'x':[1,2,3,4],'y':[2,4,6,7]})
    | doLinearRegression(df_test,'y') -> {'R^2': 0.9796610169491525}
    """
    pre='.doLinearRegression'
    log('start',pre=pre)
    X = df.drop(columns=[y_label])
    Y = df[y_label]
    regr = linear_model.LinearRegression()
    regr.fit(X,Y)
    predY = regr.predict(X)
    if plot_prediction:
        for column in X.columns:
            actual = df[[column,y_label]].values
            predictions = [item for item in zip(X[column].values,predY)]
            filepath = None if plot_prediction_filepath == None \
                    else plot_prediction_filepath[:-4]+'.'+column+'.png'
            log('final filepath',filepath,pre=pre)
            # code.interact(local=dict(globals(),**locals()))
            plotBestFit(actual,predictions,filename=filepath)
    # plotBestFit(df.values,
    stats = {'R^2':regr.score(X,Y)}
    return stats,predY

def doLogisticRegression(df,y_label,plot_prediction=False,plot_prediction_filepath=None):
    """
    (doLogisticRegression DataFrame str) This function takes in a dataframe of all the data, takes out
    the column addressed as y_label and then does a logistic regression to use the x values to predict
    y_label and returns the stats (dict) and the predictions (list)

    | kargs:
    | plot_prediction: Determines if the prediction plot will be drawn
    | plot_prediction_filepath: Determines where this plot will be saved (will show if empty)

    | ex.
    | df_test = pd.DataFrame(data={'x':[1,2,3,4],'y':[2,4,6,7]})
    | doLogisticRegression(df_test,'y') -> {'R^2': 0.9796610169491525}
    """
    pre='.doLogisticRegression'
    log('start',pre=pre)
    X = df.drop(columns=[y_label])
    Y = df[y_label]
    regr = linear_model.LogisticRegression()
    regr.fit(X,Y)
    predY = regr.predict(X)
    code.interact(local=dict(globals(),**locals()))
    if plot_prediction:
        for column in X.columns:
            actual = df[[column,y_label]].values
            predictions = [item for item in zip(X[column].values,predY)]
            filepath = None if plot_prediction_filepath == None \
                    else plot_prediction_filepath[:-4]+'.'+column+'.png'
            log('final filepath',filepath,pre=pre)
            # code.interact(local=dict(globals(),**locals()))
            plotBestFit(actual,predictions,filename=filepath)
    # plotBestFit(df.values,
    stats = {'R^2':regr.score(X,Y)}
    return stats,predY

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
    pre = '.plotBestFit'
    log('start',pre=pre)
    log('filename',filename,pre=pre)
    # actual
    plt.plot([e[0] for e in actual],[e[1] for e in actual],'ro',alpha=0.5)
    plt.plot([e[0] for e in predictions],[e[1] for e in predictions],'b-',alpha=0.5)
    # plt.plot([i for i,item in enumerate(ordered)],[item[1] for i,item in enumerate(ordered)],'r-',alpha=0.5) # predicted values
    # plt.plot([i for i,item in enumerate(ordered)],[item[0] for i,item in enumerate(ordered)],'bo',alpha=0.5) # actual values
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
    # doLinearRegression 
    df_test = pd.DataFrame(data={'x':[1,2,3,4],'y':[2,4,6,7]})
    exp_ans = {'R^2': 0.9796610169491525}
    filepath = os.path.join(LIB,'test1.png')
    ans = doLinearRegression(df_test,'y',plot_prediction=True,plot_prediction_filepath=filepath)
    print('Passed 1') if exp_ans == ans[0] else print('Failed 1')
    print('predictions',ans[1])

    # plotBestFit
    actual = [[1,2],[2,4],[3,6],[4,7]]
    predictions = [[1,2.2],[2,3.9],[3,5.6],[4,7.3]]
    plotBestFit(actual,predictions,os.path.join(LIB,'test2.png'))
    print('Processed Test 2')

    # logistic regression
    df_test = pd.DataFrame(data={'x':[1,2,3,4],'y':[2,4,6,7]})
    # exp_ans = {'R^2': 0.9796610169491525}
    filepath = os.path.join(LIB,'test3.png')
    ans = doLogisticRegression(df_test,'y',plot_prediction=True,plot_prediction_filepath=filepath)
    # print('Passed 3') if exp_ans == ans[0] else print('Failed 3')
    # print('predictions',ans[1])
