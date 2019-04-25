"""
The purpose of this module is to centralze of the the stats and plotting
functionality for reusability
"""
from bisect import bisect_left
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
def doRegression(df,y_label,model_type='Linear',task='classify',test_data=None):
    """
    (doLinearRegression DataFrame str) This function takes in a dataframe of all the data, takes out
    the column addressed as y_label and then does a linear regression to use the x values to predict
    y_label and returns the stats (dict) and the predictions (list)

    | kargs:
    | model_type:   (anyof 'Linear' 'Logistic') this determines which model will be used to assess the datapoints
    | task: (anyof 'classify' 'prob') this determines if the output is a discrete prediction or a smooth curve (probabilistic)
    | test_data: DataFrame:     This determines what data the predictions will be for, if None then the df will be used

    | ex.
    | df_test = pd.DataFrame(data={'x':[1,2,3,4],'y':[2,4,6,7]})
    | doRegression(df_test,'y',model_type='Linear') -> {'R^2': 0.9796610169491525}, [[1, 2.2000000000000006], [2, 3.9000000000000004], [3, 5.6000000000000005], [4, 7.3]]

    """
    pre='.doRegression'
    log('start',pre=pre)
    log('regression',model_type,pre=pre)
    X = df.drop(columns=[y_label])
    Y = df[y_label]
    # train the model
    regr = linear_model.LinearRegression() if model_type == 'Linear' else\
            linear_model.LogisticRegression(solver='lbfgs',multi_class='auto') if model_type == 'Logistic' else\
            None
    if regr == None:
        print('ERROR unknown model_type',model_type)
        return
    regr.fit(X,Y)
    # test the model
    categories = list({y for y in Y.values})
    X_test = test_data.drop(columns=[y_label]) if type(test_data) == pd.DataFrame else X
    Y_test = test_data[y_label] if type(test_data) == pd.DataFrame else Y
    predY = regr.predict(X_test)
    # normalize to prob
    # if model_type == 'Logistic' and task == 'prob':
    if model_type == 'Logistic':
        for i,x in enumerate(X_test.values):
            probs = regr.predict_proba([x])[0]
            smooth_y = sum([categories[i]*prob for i,prob in enumerate(probs)])
            predY[i] = smooth_y
    # discretize the predictions
    if task == 'classify':
        for i,y in enumerate(predY):
            dist = sys.maxsize
            closest = None
            for cat in categories:
                diff = abs(y-cat)
                if diff < dist:
                    closest = cat
                    dist = diff
            predY[i] = closest
    pred = [[*entry[0],entry[1]] for entry in zip(X_test.values,predY)]
    # compare the prediction to the expected values
    stats = {'R^2':sklearn.metrics.r2_score(Y_test,predY)}
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

    # plotBestFit
    actual = [[1,2],[2,4],[3,6],[4,7]]
    predictions = [[1,2.2],[2,3.9],[3,5.6],[4,7.3]]
    plotBestFit(actual,predictions,os.path.join(LIB,'test1.png'))
    print('Processed Test 1')

    # doLinearRegression 
    df_test = pd.DataFrame(data={'x':[1,2,3,4],'y':[2,4,6,7]})
    exp_ans = {'R^2': 0.9796610169491525}, [[1, 2.2000000000000006], [2, 3.9000000000000004], [3, 5.6000000000000005], [4, 7.3]]
    filepath = os.path.join(LIB,'test2a.png')
    ans,pred = doRegression(df_test,'y',model_type='Linear',task='prob')
    print('Passed 2a') if exp_ans == (ans,pred) else print('Failed 2a got',(ans,pred),'instead of',exp_ans)
    plotBestFit(df_test[['x','y']],pred,filename=os.path.join(LIB,'test2a.png'))
    exp_ans = {'R^2': 1}, [[1, 2.0], [2, 4.0], [3, 6.0], [4, 7.0]]
    filepath = os.path.join(LIB,'test2b.png')
    ans,pred = doRegression(df_test,'y',model_type='Linear',task='classify')
    print('Passed 2b') if exp_ans == (ans,pred) else print('Failed 2b got',(ans,pred),'instead of',exp_ans)
    plotBestFit(df_test[['x','y']],pred,filename=os.path.join(LIB,'test2b.png'))

    # logistic regression
    df_test = pd.DataFrame(data={'x':[1,2,3,4],'y':[2,4,6,7]})
    test_data = pd.DataFrame(data={'x':[0.1*e for e in range(100)], 'y':[0.2*e for e in range(100)]})
    ans,pred = doRegression(df_test,'y',model_type='Logistic',task='prob',test_data=test_data)
    exp_ans = ({'R^2': 0.25},[[1, 4], [2, 7], [3, 7], [4, 7]])
    print('Passed 3a') if exp_ans == (ans,pred) else print('Failed 3a got',(ans,pred),'instead of',exp_ans)
    plotBestFit(df_test[['x','y']],pred,filename=os.path.join(LIB,'test3a.png'))
    ans,pred = doRegression(df_test,'y',model_type='Logistic',task='classify')
    exp_ans = None
    print('Passed 3b') if exp_ans == (ans,pred) else print('Failed 3b got',(ans,pred),'instead of',exp_ans)
    plotBestFit(df_test[['x','y']],pred,filename=os.path.join(LIB,'test3b.png'))
