"""
The purpose of this module is to centralze of the the stats and plotting
functionality for reusability
"""
import math
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
import pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import sys
import code
import xgboost as xgb
import torch
import torch.nn as nn

TOP = os.path.abspath(os.path.join(__file__,'../../'))
LIB = os.path.join(TOP,'Library')
sys.path.append(LIB)
import logger
def log(*args,pre=None):
    logger.log(*args,pre='analyzer.py' if pre==None else 'analyzer.py'+pre)
debug = False

# setup constants
RANDOM_SEED = 1 # set to an int for consitency or None for random
torch.manual_seed(RANDOM_SEED)

# CLASSES
class Linear_wrapper:
    """
    Wrapper for the linear regression
    """
    def __init__(self):
        """
        """
        self.pre = '.Linear_wrapper'
        self.model = linear_model.LinearRegression()

    def fit(self,X,Y):
        self.model.fit(X.values,Y.values)

    def predict(self,X_test,threshold=0.5):
        probs = self.predict_proba(X_test)
        ans = [0 if p < threshold else 1 for p in probs]
        return ans

    def predict_proba(self,X_test):
        pred = self.model.predict(X_test.values)
        ans = [p if p <= 1 else 1 for p in pred]
        return ans

class Logistic_wrapper:
    """
    Wrapper for the linear regression
    """
    def __init__(self):
        """
        """
        self.pre = '.Logistic_wrapper'
        self.model = linear_model.LogisticRegression(solver='liblinear',multi_class='ovr')

    def fit(self,X,Y):
        self.model.fit(X.values,Y.values)

    def predict(self,X_test,threshold=0.5):
        probas = self.predict_proba(X_test)
        ans = [0 if p < threshold else 1 for p in probas]
        return ans

    def predict_proba(self,X_test):
        return [e[1] for e in self.model.predict_proba(X_test.values)]

class Neural_wrapper:
    """
    Wrapper for the neural network model
    """
    def __init__(self):
        """
        """
        self.pre = '.Neural_Wrapper'
        self.model = None

    def fit(self,X,Y):
        """
        """
        pre = self.pre + '.fit'
        n_in, n_h, n_out, batch_size = len(X.columns), 5, 1, 4
        x0 = torch.randn(batch_size, n_in)
        # X = X.copy()
        # normalize X
        x = torch.tensor(X.values.tolist())
        y = torch.tensor([[float(y)] for y in Y.values.tolist()])
        # define the structure of the model
        self.model = nn.Sequential(nn.Linear(n_in, n_h),
                     nn.ReLU(),
                     nn.Linear(n_h, n_out),
                     nn.Sigmoid())
        criterion = torch.nn.MSELoss() # define our loss function
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05) # lr = learning rate
        # train the model
        for epoch in range(50):
            log('epoch',epoch,'begun',pre=pre)
            # Forward Propagation
            y_pred = self.model(x)
            log('y_pred',y_pred,pre=pre)
            # Compute and print loss
            loss = criterion(y_pred, y)
            log('epoch: ', epoch,' loss: ', loss.item(),pre=pre)
            # Zero the gradients
            optimizer.zero_grad()
            # perform a backward pass (backpropagation)
            loss.backward()
            # Update the parameters
            optimizer.step()
        log('model is now trained',pre=pre)

    def predict(self,X_test,threshold=0.5):
        """
        """
        pre = self.pre + '.predict'
        log('start',pre=pre)
        log('X_test shape',X_test.shape,pre=pre)
        log(X_test.head(),pre=pre)
        # translate to tensor
        x = torch.tensor(X_test.values.tolist())
        log('new x',x,pre=pre)
        preds = self.model(x)
        log('preds',preds,pre=pre)
        ans = [1 if y >= threshold else 0 for y in preds]
        log('returned',ans,pre=pre)
        return ans

    def predict_proba(self,X_test):
        """
        """
        pre = self.pre + '.predict_proba'
        log('start',pre=pre)
        log('X_test shape',X_test.shape,pre=pre)
        log(X_test.head(),pre=pre)
        # translate to tensor
        x = torch.tensor(X_test.values.tolist())
        log('new x',x,pre=pre)
        preds = self.model(x)
        lsts = preds.tolist()
        ans = [e[0] for e in lsts]
        log('preds',preds,pre=pre)
        log('ans',ans,pre=pre)
        return ans

class SVM_wrapper:
    """
    Wrapper for the linear regression
    """
    def __init__(self):
        """
        """
        self.pre = '.SVM_wrapper'
        self.model = sklearn.svm.SVC(kernel='rbf',gamma='scale',probability=True,random_state=RANDOM_SEED)

    def fit(self,X,Y):
        self.model.fit(X.values,Y.values)

    def predict(self,X_test,threshold=0.5):
        probas = self.predict_proba(X_test)
        ans = [0 if p < threshold else 1 for p in probas]
        return ans

    def predict_proba(self,X_test):
        return [e[1] for e in self.model.predict_proba(X_test)]

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

    def predict(self,X_test,threshold=0.5):
        """
        """
        pre = self.pre + '.predict'
        log('start',pre=pre)
        log('pre X_test type',type(X_test),pre=pre)
        probs = self.predict_proba(X_test)
        ans = [0 if y < threshold else 1 for y in probs]
        self.model.dump_model('/home/alex/test.txt',dump_format='text')
        log('returned',pre=pre)
        return ans

    def predict_proba(self,X_test):
        """
        """
        pre = self.pre + '.predict_proba'
        log('start',pre=pre)
        log('type X_test',type(X_test),pre=pre)
        lst = X_test.values if type(X_test) == pd.DataFrame else X_test
        dtest = xgb.DMatrix(lst)
        probs = self.model.predict(dtest)
        log('returned',pre=pre)
        return probs

    def output_tree(self):
        self.model.dump_model('/home/alex/test.txt',dump_format='text')

# FUNCTIONS
def f1(confusion_matrix):
    """
    (f1 confusion_matrix) The purpose of this function is to calculate the f1 score of a confusion_matrix
    ex.

    data = [[1,2],[3,4]] # C_{i,j} where i is the true value and j is the predicted value
    f1(data) -> 0.6153846153846153
    """
    ans = None
    prec = precision(confusion_matrix)
    rec = recall(confusion_matrix)
    if type(prec) == str or type(rec) == str:
        return 'There are zeroes here'
    ans = 2 * (precision(confusion_matrix) * recall(confusion_matrix))/(precision(confusion_matrix) + recall(confusion_matrix))
    return ans

def get_model(model_type):
    """
    (get_model str) This function creates and returns the model to be used for analysis.

    get_modeL: str -> <a model that fits the sklearn.linear_model.LogisticRegression archetype>
    """
    # documentation:
    # logistic: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    regr = Linear_wrapper() if model_type == 'Linear' else\
            Logistic_wrapper() if model_type == 'Logistic' else\
            SVM_wrapper() if model_type == 'SVM' else\
            Xgboost_wrapper() if model_type == 'Xgboost' else\
            Neural_wrapper() if model_type == 'Neural' else\
            None
    return regr
    
def assess(df,y_label,model_type='Linear',ranges=None):
    """
    (assess DataFrame str) This function takes in a dataframe of all the data, takes out
    the column addressed as y_label and then does a linear regression to use the x values to predict
    y_label and returns the stats (dict) and the predictions (list)

    | kargs:
    | model_type:   (anyof 'Linear' 'Logistic' 'Xgboost' 'SVM' 'Neural') this determines which model will be used to assess the datapoints
    | ranges: (dictof col_name: (lstof min max)): This determines the full range of values that each column to take (used to normalize the data)

    | ex.
    | df_test = pd.DataFrame(data={'x':[1,2,3,4],'y':[2,4,6,7]})
    | doRegression(df_test,'y',model_type='Linear') -> {'R^2': 0.9796610169491525}, [[1, 2.2000000000000006], [2, 3.9000000000000004], [3, 5.6000000000000005], [4, 7.3]]

    Note: Logistic defaults to one vs all classification

    """
    pre='.assess'
    log('start',pre=pre)
    log('regression',model_type,pre=pre)
    # shuffle the DataFrame rows
    df = df.sample(frac=1,random_state=RANDOM_SEED).reset_index(drop=True)
    log('df sampled head',df.head(),pre=pre)
    X = df.drop(columns=[y_label])
    Y = df[y_label]
    # separate into 11 sets (10 for fold validation and 1 for testing the best threshold)
    Xs = []
    Ys = []
    log('X shape',X.shape,pre=pre)
    num_of_groups = 11
    interval = X.shape[0]//num_of_groups
    log('interval is',interval,pre=pre)
    for i in range(num_of_groups):
        log('range for i of',i,'is',i*interval,'to',(i+1)*interval,pre=pre)
        x_set = X[i*interval:(i+1)*interval] if i < (num_of_groups-1) else X[i*interval:]
        y_set = Y[i*interval:(i+1)*interval] if i < (num_of_groups-1) else Y[i*interval:]
        log('x_set shape',x_set.shape,pre=pre)
        log('y_set shape',y_set.shape,pre=pre)
        Xs.append(x_set)
        Ys.append(y_set)
    log('Xs shape',[e.shape for e in Xs],pre=pre)
    log('Ys shape',[e.shape for e in Ys],pre=pre)
    fold_results = []
    # do the 10 fold testing (so exclude the very last set)
    for i,_ in enumerate(Xs[:-1]):
        X_test = Xs[i]
        Y_test = Ys[i]
        X_train,Y_train = None,None
        setup = False
        for j,_ in enumerate(Xs[:-1]):
            if i != j:
                if not setup:
                    X_train = Xs[j]
                    Y_train = Ys[j]
                    setup = True
                else:
                    X_train = X_train.append(Xs[j])
                    Y_train = Y_train.append(Ys[j])
        # train the model
        # svm: radial basis function (i.e. gaussian)
        regr = get_model(model_type)
        if regr == None:
            print('ERROR unknown model_type',model_type)
            return

        entry = test_basic(X_train, Y_train, X_test, Y_test, regr)
        fold_results.append(entry)
    # aggregate the output data
    stats_lst = [e['stats'] for e in fold_results] # aggregate the stats
    final_matrix = [[0,0],[0,0]]
    for stats in stats_lst:
        matrix = stats['confusion']
        for i,row in enumerate(matrix):
            for j,col in enumerate(row):
                final_matrix[i][j] += col
    log('final_matrix',final_matrix,pre=pre)
    final_matrix_prop = [[0,0],[0,0]]
    tot = sum([n for row in final_matrix for n in row])
    log('final_matrix tot',tot,pre=pre)
    for i,row in enumerate(final_matrix_prop):
        for j,_ in enumerate(row):
            final_matrix_prop[i][j] = final_matrix[i][j]/tot
    all_pred,all_proba,all_y_test = [],[],[]
    log('all originals',all_pred,all_proba,all_y_test,pre=pre)
    for entry in fold_results:
        all_pred.extend(entry['pred'])
        all_proba.extend(entry['pred_proba'])
        all_y_test.extend(entry['y_test'])
    log('all_pred',all_pred,pre=pre)
    log('all_proba',all_proba,pre=pre)
    log('all_y_test',all_y_test,pre=pre)
    prec,rec,thresholds = metrics.precision_recall_curve(all_y_test,all_proba) 
    # get the best threshold
    best_threshold = bestThreshold(rec,prec,thresholds,(1.0,1.0))
    log('best threshold',best_threshold,pre=pre)
    # test on the last set with the best threshold
    X_train = Xs[0]
    Y_train = Ys[0]
    for i in range(1,11):
        X_train = X_train.append(Xs[i])
        Y_train = Y_train.append(Ys[i])
    X_test = Xs[-1]
    Y_test = Ys[-1]
    predY = regr.predict(X_test)
    pred_proba = regr.predict_proba(X_test)
    f1_test = metrics.f1_score(Y_test,predY)
    precision_test = metrics.precision_score(Y_test,predY)
    recall_test = metrics.recall_score(Y_test,predY)
    test_stats = {
            'best threshold':best_threshold,
            'confusion':metrics.confusion_matrix(Y_test.values,predY).tolist(),
            'f1':f1_test,
            'precision':precision_test,
            'recall':recall_test
            }
    # format the output
    final_stats = {
        'confusion':final_matrix,
        'confusion prop':final_matrix_prop,
        'f1':f1(final_matrix),
        'precision':precision(final_matrix),
        'recall':recall(final_matrix),
        'test':test_stats
    }
    data = {
        'predictions':all_pred,
        'probabilities':all_proba,
        'y_test':all_y_test,
        'precision':prec,
        'recall':rec,
        'thresholds':thresholds
    }
    return final_stats,data

def bestThreshold(x,y,thresholds,opt):
    """
    (bestThreshold x y thresholds opt) The purpose of this function is to determine the best thresold based on a set of values for the x-axis, the y-axis, and the thresholds used for each as well as an optimal point (opt) to evaluate the distance to (shortest distanct to the optimal is the 'best')

    bestThreshold: (listof num) (listof num) (listof num) (point) -> num <- the 'best' threshold
    ex.

    x = [1.5,2,3,4]
    y = [1.2,3,4,5]
    thresholds = [0.1,0.2,0.3]
    opt = (1.0,1.0)
    bestThreshold(np.array([x]),np.array([y]),np.array([thresholds]),opt) -> 0.1 # because this is the threshold that produces the coordinates that are cloest to opt
    """
    def dist(x,y):
        # return the distance of current point to opt
        return math.sqrt((x-opt[0])**2 + (y-opt[1])**2)
    bestDist = sys.maxsize
    bestIndex = None
    for i,threshold in enumerate(thresholds):
        currX = x[i]
        currY = y[i]
        currDist = dist(currX,currY)
        if currDist < bestDist:
            bestDist = currDist
            bestIndex = i
    return float(thresholds[bestIndex])

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

def plotPRC(prec,rec,thresholds,labels,filepath):
    """
    (plotROC fpr tpr filepath) The purpose of this function is to create a plot depicting the ROC curve given the false positive rates (fpr), the true positive rates (tpr), and saving it to the filepath provided.
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
    >>> fpr
    array([0. , 0. , 0.5, 0.5, 1. ])
    >>> tpr
    array([0. , 0.5, 0.5, 1. , 1. ])
    >>> thresholds
    array([1.8 , 0.8 , 0.4 , 0.35, 0.1 ])
    """
    pre = '.analyzer.plotPRC'
    log('start',pre=pre)
    log('prec',prec,pre=pre)
    log('rec',rec,pre=pre)
    log('type prec[0]',type(prec[0]),pre=pre)
    if type(prec[0]) in [list,np.ndarray]: # then this is a plot that is intended to plot multiple instances
        for i,_ in enumerate(prec):
            tmp_precision = prec[i]
            tmp_recall = rec[i]
            tmp_thresholds = thresholds[i]
            plotPRC_helper(tmp_precision,tmp_recall,tmp_thresholds,labels[i])
    else:
        plotPRC_helper(prec,rec,thresholds,'')
    plt.legend(loc='best')
    plt.savefig(filepath)
    plt.close()

def plotPRC_helper(prec,rec,thresholds,label):
    """
    (plotROC prec rec label) The purpose of this function is to create a plot depicting the ROC curve given the false positive rates (fpr), the true positive rates (tpr), and saving it to the filepath provided.
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
    >>> fpr
    array([0. , 0. , 0.5, 0.5, 1. ])
    >>> tpr
    array([0. , 0.5, 0.5, 1. , 1. ])
    >>> thresholds
    array([1.8 , 0.8 , 0.4 , 0.35, 0.1 ])
    """
    pre = '.analyzer.plotPRC_helper'
    log('start',pre=pre)
    log('prec',prec,pre=pre)
    log('rec',rec,pre=pre)
    # plot no skill
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    # plot the roc curve for the model
    plt.plot(rec, prec, linestyle='-',alpha=0.7,label=label)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    # put the best threshold point
    best_threshold = bestThreshold(rec,prec,thresholds,(1.0,1.0))
    point = [(rec[i],prec[i]) for i,threshold in enumerate(thresholds) if float(threshold) == best_threshold][0]
    plt.annotate('|_',point)

def precision(confusion_matrix):
    """
    (precision confusion_matrix) The purpose of this function is to calculate the precision given a confusion matrix
    
    ex.
    data = [[1,2],[3,4]] # C_{i,j} where i is the true value and j is the predicted value
    precision(data) -> 0.6666666666666666
    """
    true_pos = confusion_matrix[1][1]
    false_pos = confusion_matrix[0][1]
    if true_pos + false_pos == 0:
        return 'Zeroes here'
    return true_pos/(true_pos + false_pos)

def recall(confusion_matrix):
    """
    (recall confusion_matrix) The purpose of this function is to calculate the recall given a confusion matrix

    ex.
    data = [[1,2],[3,4]] # C_{i,j} where i is the true value and j is the predicted value
    recall(data) -> 0.5714285714285714
    """
    true_pos = confusion_matrix[1][1]
    false_neg = confusion_matrix[1][0]
    if true_pos + false_neg == 0:
        return 'Zeroes here'
    return true_pos/(true_pos + false_neg)

def test(train_df, test_df, y_label, model_type, include_only=None, threshold=0.5):
    """
    (test DataFrame DataFrame str str) This function trains the model defined by model_type with the set 
    contained by train_df and tests this trained model on test_df utilizing the y_label to identify the 
    dependent variable to predict and test and returns the stats and predicions.

    | kwargs:
    | include_only - If this is given, this test will return only the metrics provided in the list
    | threshold - below the threshold a 0 is selected, above the threshold a 1 is selected
    """
    pre = '.test'
    log('start',pre=pre)

    # create the model
    regr = get_model(model_type)
    # train the model
    X_train = train_df.drop(columns=[y_label])
    Y_train = train_df[y_label]
    X_test = test_df.drop(columns=[y_label])
    Y_test = test_df[y_label]
    vals = test_basic(X_train, Y_train, X_test, Y_test, regr,threshold)
    ans = vals if include_only == None else {label:vals[label] for label in include_only}
    log('returned',ans,pre=pre)
    return ans


def test_basic(X_train, Y_train, X_test, Y_test, model, threshold=0.5):
    # this function just tests the dataframes with the given model and returns the stats for them
    pre = '.test_basic'
    log('start',pre=pre)
    log('X_train, first 10',X_train[:10],pre=pre)
    log('X_test, first 10',X_test[:10],pre=pre)
    log('Y_train, first 10',Y_train[:10],pre=pre)
    log('Y_test, first 10',Y_test[:10],pre=pre)
    model.fit(X_train,Y_train)
    if type(model) == Xgboost_wrapper:
        model.output_tree()
    # test the model
    log('X_test',X_test,pre=pre)
    predY = model.predict(X_test,threshold=threshold)
    pred_proba = model.predict_proba(X_test)
    # pred_proba = predY.copy().tolist() if model_type in ['Linear','Xgboost'] else [e[1] for e in regr.predict_proba(X_test)] # the predict_proba gives a 2d numpy array of the probability of each classifcation
    log('predY original',predY,pre=pre)
    log('pred_proba',pred_proba,pre=pre)
    # compare the prediction to the expected values
    log('Y_test',Y_test,pre=pre)
    log('predY',predY,pre=pre)
    stats = {
            'confusion':metrics.confusion_matrix(Y_test,predY).tolist()
            }
    # prepare the output
    pred = [[*entry[0],entry[1]] for entry in zip(X_test,predY)]
    entry = {
            'stats':stats,
            'pred':predY,
            'pred_proba':pred_proba, # this is a dataframe, therefore cannot JSON this
            'y_test':Y_test,
            'precision':metrics.precision_score(Y_test,predY),
            'recall':metrics.recall_score(Y_test,predY)
            }
    log('returned',entry,pre=pre)
    log('end',pre=pre)
    return entry

if __name__ == '__main__':
    # testing
    logger.deleteLogs()

    # 1. doLinearRegression 
    df = pd.DataFrame(data={'x':[0]*5+[1]*5,'y':[1]*5+[0]*5})
    regr = Linear_wrapper()
    regr.fit(df[['x']],df['y'])
    x_test = [[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]]
    y_test = [0,0,0,0,0,1,1,1,1,1]
    ans = regr.predict(x_test)
    exp_ans = y_test
    print('Passed 1') if exp_ans == ans else print('Failed 1 got',ans,'instead of',exp_ans)

    # 2. logistic regression
    df = pd.DataFrame(data={'x':[0]*5+[1]*5,'y':[1]*5+[0]*5})
    regr = Logistic_wrapper()
    regr.fit(df[['x']],df['y'])
    x_test = [[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]]
    y_test = [0,0,0,0,0,1,1,1,1,1]
    ans = regr.predict(x_test)
    exp_ans = y_test
    print('Passed 2') if exp_ans == ans else print('Failed 2 got',ans,'instead of',exp_ans)

    # 3. SVM model
    df = pd.DataFrame(data={'x':[0]*5+[1]*5,'y':[1]*5+[0]*5})
    regr = SVM_wrapper()
    regr.fit(df[['x']],df['y'])
    x_test = [[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]]
    y_test = [0,0,0,0,0,1,1,1,1,1]
    ans = regr.predict(x_test)
    exp_ans = y_test
    print('Passed 3') if exp_ans == ans else print('Failed 3 got',ans,'instead of',exp_ans)

    # 4. Xgboost model
    df = pd.DataFrame(data={'x':[0]*5+[1]*5,'y':[1]*5+[0]*5})
    regr = Xgboost_wrapper()
    regr.fit(df[['x']],df['y'])
    x_test = [[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]]
    y_test = [0,0,0,0,0,1,1,1,1,1]
    ans = regr.predict(x_test)
    exp_ans = y_test
    print('Passed 4') if exp_ans == ans else print('Failed 4 got',ans,'instead of',exp_ans)

    # 6. neural network
    df_test = pd.DataFrame(data={'x':[e/100 for e in range(100)],'y':[0 if e/100 < 0.5 else 1 for e in range(100)]})
    ans = assess(df_test,'y',model_type='Neural')
    print('Test 6 processed: neural')

    data = [[1,2],[3,4]] # C_{i,j} where i is the true value and j is the predicted value
    y_true = [0,0,0,1,1,1,1,1,1,1]
    y_pred = [0,1,1,0,0,0,1,1,1,1]
    # 7. precision
    exp_ans = metrics.precision_score(y_true,y_pred)
    ans = precision(data)
    print('Passed 7') if exp_ans == ans else print('Failed 7 got',ans,'instead of',exp_ans)
    # 8. recall
    exp_ans = metrics.recall_score(y_true,y_pred)
    ans = recall(data)
    print('Passed 8') if exp_ans == ans else print('Failed 8 got',ans,'instead of',exp_ans)
    # 9. recall
    exp_ans = metrics.f1_score(y_true,y_pred)
    ans = f1(data)
    print('Passed 9') if exp_ans == ans else print('Failed 9 got',ans,'instead of',exp_ans)
    # 10. bestThreshold
    x = np.array([1.5,2,3,4])
    y = np.array([1.2,3,4,5])
    thresholds = np.array([0.1,0.2,0.3])
    opt = (1.0,1.0)
    exp_ans = 0.1
    ans = bestThreshold(x,y,thresholds,opt)
    print('Passed 10') if exp_ans == ans else print('Failed 10 got',ans,'instead of',exp_ans)


