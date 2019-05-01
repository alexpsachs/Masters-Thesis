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
import torch
import torch.nn as nn

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
class Neural_wrapper:
    """
    Wrapper for the neural network model
    """
    def __init__(self):
        self.pre = '.Neural_Wrapper'
        self.model = None

    def fit(self,X,Y,ranges=None):
        """
        """
        pre = self.pre + '.fit'
        n_in, n_h, n_out, batch_size = len(X.columns), 5, 1, 4
        x0 = torch.randn(batch_size, n_in)
        X = X.copy()
        # normalize X
        if ranges == None:
            # inter the min and max for each dimension
            self.ranges = []
            # TODO
        else:
            self.ranges = ranges
        X_norm = (X - X.min())/(X.max()-X.min()) # X_min is now 0 and X_max is now 1
        x = torch.tensor(X_norm.values.tolist())
        y = torch.tensor([[float(y)] for y in Y.values.tolist()])
        # define the structure of the model
        self.model = nn.Sequential(nn.Linear(n_in, n_h),
                     nn.ReLU(),
                     nn.Linear(n_h, n_out),
                     nn.Sigmoid())
        criterion = torch.nn.MSELoss() # define our loss function
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01) # lr = learning rate
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

    def predict(self,X_test):
        """
        """
        pre = self.pre + '.predict'
        log('start',pre=pre)
        code.interact(local=dict(globals(),**locals()))
        preds = self.model(X_test)
        ans = [1 if y >= 0.5 else 0 for y in preds]
        log('returned',pre=pre)
        return preds

    def output_tree(self):
        self.model.dump_model('/home/alex/test.txt',dump_format='text')

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
def f1(confusion_matrix):
    """
    (f1 confusion_matrix) The purpose of this function is to calculate the f1 score of a confusion_matrix
    ex.

    data = [[1,2],[3,4]] # C_{i,j} where i is the true value and j is the predicted value
    f1(data) -> 0.6153846153846153
    """
    ans = 2 * (precision(confusion_matrix) * recall(confusion_matrix))/(precision(confusion_matrix) + recall(confusion_matrix))
    return ans

    
def assess(df,y_label,model_type='Linear'):
    """
    (assess DataFrame str) This function takes in a dataframe of all the data, takes out
    the column addressed as y_label and then does a linear regression to use the x values to predict
    y_label and returns the stats (dict) and the predictions (list)

    | kargs:
    | model_type:   (anyof 'Linear' 'Logistic' 'Xgboost' 'SVM' 'Neural') this determines which model will be used to assess the datapoints
    | test_data: DataFrame:     This determines what data the predictions will be for, if None then the df will be used

    | ex.
    | df_test = pd.DataFrame(data={'x':[1,2,3,4],'y':[2,4,6,7]})
    | doRegression(df_test,'y',model_type='Linear') -> {'R^2': 0.9796610169491525}, [[1, 2.2000000000000006], [2, 3.9000000000000004], [3, 5.6000000000000005], [4, 7.3]]

    Note: Logistic defaults to one vs all classification

    """
    pre='.assess'
    log('start',pre=pre)
    log('regression',model_type,pre=pre)
    df = df.sample(frac=1).reset_index(drop=True)
    X = df.drop(columns=[y_label])
    Y = df[y_label]
    # separate into 10 sets for 10-fold validation
    Xs = []
    Ys = []
    interval = X.shape[0]//10
    for i in range(10):
        x_set = X[i*interval:(i+1)*interval] if i < 9 else X[i*interval:]
        y_set = Y[i*interval:(i+1)*interval] if i < 9 else Y[i*interval:]
        Xs.append(x_set)
        Ys.append(y_set)
    fold_results = []
    for i,_ in enumerate(Xs):
        X_test = Xs[i]
        Y_test = Ys[i]
        X_train,Y_train = None,None
        setup = False
        for j,_ in enumerate(Xs):
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
        regr = linear_model.LinearRegression() if model_type == 'Linear' else\
                linear_model.LogisticRegression(solver='liblinear',multi_class='ovr') if model_type == 'Logistic' else\
                sklearn.svm.SVC(kernel='rbf',gamma='scale') if model_type == 'SVM' else\
                Xgboost_wrapper() if model_type == 'Xgboost' else\
                Neural_wrapper() if model_type == 'Neural' else\
                None
        # documentation:
        # logistic: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        if regr == None:
            print('ERROR unknown model_type',model_type)
            return
        regr.fit(X_train,Y_train)
        if model_type == 'Xgboost':
            regr.output_tree()
        # test the model
        categories = list({y for y in Y.values})
        # X_test = test_data.drop(columns=[y_label]) if type(test_data) == pd.DataFrame else X
        # Y_test = test_data[y_label] if type(test_data) == pd.DataFrame else Y
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
        fold_results.append((stats,pred))
    stats_lst = [e[0] for e in fold_results] # aggregate the stats
    final_matrix = [[0,0],[0,0]]
    for stats in stats_lst:
        matrix = stats['confusion']
        for i,row in enumerate(matrix):
            for j,col in enumerate(row):
                final_matrix[i][j] += col
    final_stats = {
                'confusion':final_matrix,
                'f1':f1(final_matrix),
                'precision':precision(final_matrix),
                'recall':recall(final_matrix),
                }
    return final_stats

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

def precision(confusion_matrix):
    """
    (precision confusion_matrix) The purpose of this function is to calculate the precision given a confusion matrix
    
    ex.
    data = [[1,2],[3,4]] # C_{i,j} where i is the true value and j is the predicted value
    precision(data) -> 0.6666666666666666
    """
    true_pos = confusion_matrix[1][1]
    false_pos = confusion_matrix[0][1]
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
    return true_pos/(true_pos + false_neg)

if __name__ == '__main__':
    # testing
    logger.deleteLogs()

    # # 1. plotBestFit
    # actual = [[1,2],[2,4],[3,6],[4,7]]
    # predictions = [[1,2.2],[2,3.9],[3,5.6],[4,7.3]]
    # plotBestFit(actual,predictions,os.path.join(LIB,'test1.png'))
    # print('Processed Test 1')

    # # 2. doLinearRegression 
    # df_test = pd.DataFrame(data={'x':[1,2,3,4],'y':[1,1,0,0]})
    # exp_conf = np.array([[2, 0],[0, 2]])
    # exp_f1 = 1.0
    # exp_pred = [[1, 1.0], [2, 1.0], [3, 0.0], [4, 0.0]]
    # filepath = os.path.join(LIB,'test2.png')
    # ans,pred = assess(df_test,'y',model_type='Linear')
    # print('Passed 2') if np.array_equal(exp_conf,ans['confusion'])\
    #         and exp_f1 == ans['f1']\
    #         and exp_pred == pred\
    #         else print('Failed 2 got')

    # # 3. logistic regression
    # df_test = pd.DataFrame(data={'x':[1,2,3,4],'y':[0,0,1,1]})
    # ans,pred = assess(df_test,'y',model_type='Logistic')
    # exp_conf = np.array([[1,1],[0,2]])
    # exp_f1 = 0.8
    # exp_pred = [[1, 0], [2, 1], [3, 1], [4, 1]]
    # print('Passed 3') if np.array_equal(exp_conf,ans['confusion']) and exp_f1 == ans['f1'] and exp_pred == pred else print('Failed 3 got',(ans,pred),'instead of',exp_ans)

    # 4. xgboost
    # seems to work on the actual dataset so... whatever I guess?

    # # 5. svm
    # df_test = pd.DataFrame(data={'x':[1,2,3,4],'y':[0,0,1,1]})
    # ans,pred = assess(df_test,'y',model_type='SVM')
    # exp_conf = np.array([[2,0],[0,2]])
    # exp_f1 = 1.0
    # exp_pred = [[1, 0], [2, 0], [3, 1], [4, 1]]
    # print('Passed 4') if np.array_equal(exp_conf,ans['confusion']) and exp_f1 == ans['f1'] and np.array_equal(exp_pred,pred) else print('Failed $ got',(ans,pred),'instead of',({'confusion':exp_conf,'f1':exp_f1},pred))

    # 6. neural network
    # df_test = pd.DataFrame(data={'x':[1,2,3,4],'y':[0,0,1,1]})
    # ans,pred = assess(df_test,'y',model_type='Neural')

    # exp_conf = np.array([[2,0],[0,2]])
    # exp_f1 = 1.0
    # exp_pred = [[1, 0], [2, 0], [3, 1], [4, 1]]
    # print('Passed 4') if np.array_equal(exp_conf,ans['confusion']) and exp_f1 == ans['f1'] and np.array_equal(exp_pred,pred) else print('Failed 4 got',(ans,pred),'instead of',({'confusion':exp_conf,'f1':exp_f1},pred))

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
    # 8. recall
    exp_ans = metrics.f1_score(y_true,y_pred)
    ans = f1(data)
    print('Passed 9') if exp_ans == ans else print('Failed 9 got',ans,'instead of',exp_ans)
    print(ans)

