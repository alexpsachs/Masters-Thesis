"""
This module is intended to used to calculate various inter-rater agreement measures
"""
import pandas as pd
import os,sys,code
from statsmodels.stats.anova import AnovaRM
TOP = os.path.abspath(os.path.join(__file__,'../..'))
LIB = os.path.join(TOP,'Library')
sys.path.append(LIB)
import logger
from sklearn.metrics import cohen_kappa_score

def log(*args,pre=None):
    logger.log(*args,pre='ICC' if pre == None else 'ICC.'+pre)

def calcICC(judges, data):
    """
    judges: list of judge names
    data: list of rows of data, each row is a different target
    """
    d = pd.DataFrame(data=data)
    d.columns = judges
    # aovrm = AnovaRM(d, 'judge1', 'judge2', 'judge4',within=['judge3'])
    # res = aovrm.fit()
    # print(res)

    n = sum(d.count())
    k = len(judges)
    grand_mean = sum(d.mean()/k)
    print(grand_mean)

    tblHeight = 6
    SST = 4*sum([(d[judge].mean()-grand_mean)**2 for judge in judges])
    df = tblHeight-1
    MS_BT = SST/df
    print(MS_BT)

def ICC0(data, colHeaders=None, rowHeaders=None):
    """
    (ICC data, [colHeaders, rowHeaders]) is responsible for calculating and returning
    summarizing statistics on the dataset provided data with respect to the headers
    provided.

    ICC: (listof (listof (anyof float int))) -> float, (dictof str: float)

    example:
        data = [
            [9, 2, 5, 8],
            [6, 1, 3, 2],
            [8, 4, 6, 8],
            [7, 1, 2, 6],
            [10, 5, 6, 9],
            [6, 2, 4, 7]
        ]

        # where 9,2,5,8 are the votes from judge 0,1,2,3 for target 0

        ICC(data) -> 0.91, {'BMS': 11.241, 'WMS': 6.2638', 'EMS': 1.019}
    """

    ratings = data
    COLHEADERS = ROWHEADERS = None
    tblWidth = len(ratings[0])
    tblHeight = len(ratings)
    if colHeaders == None:
        COLHEADERS = ['J'+str(i) for i in range(tblWidth)]
    else:
        COLHEADERS = colHeaders[:]
    if rowHeaders == None:
        ROWHEADERS = ['Target'+str(i) for i in range(tblHeight)]
    else:
        ROWHEADERS = rowHeaders[:]
        
    # Calculate row means
    rowMeans = []
    for row in ratings:
        rowMean = sum(row)/tblWidth
        rowMeans.append(rowMean)
    COLHEADERS.append('Mean')
    # Add in col means
    colMeans = []
    for i in range(len(ratings[0])):
        colMeans.append(sum([row[i] for row in ratings])/tblHeight)
    GLOBALMEAN = sum([entry for row in ratings for entry in row])/(tblHeight*tblWidth)
    colMeans.append(GLOBALMEAN)
    ROWHEADERS.append('Mean')


    SST = tblWidth*sum([(mean-GLOBALMEAN)**2 for mean in rowMeans])
    df = tblHeight-1
    MS_BT = SST/df

    n = tblHeight*tblWidth
    SWT = sum([(entry - rowMeans[rowNum])**2\
            for rowNum, row in enumerate(ratings) for colNum, entry in enumerate(row)])
    df = n - tblHeight
    MS_WT = SWT/df

    SSI = sum([(entry-rowMeans[rowNum]-colMeans[colNum]+GLOBALMEAN)**2\
            for rowNum, row in enumerate(ratings) for colNum, entry in enumerate(row)])
    df = (tblHeight-1)*(tblWidth-1)
    MSI = SSI/df

    BMS = MS_BT
    WMS = MS_WT
    EMS = MSI

    ans = {'BMS': MS_BT, 'WMS': MS_WT, 'EMS': MSI}

    # calculate ICC(3,k)
    ICC3k = (BMS-EMS)/BMS
    return ICC3k, ans

def ICC(data, min_val, max_val, normalize = True):
    """
    (ICC data, min_val, max_val) is responsible for calculating and returning
    summarizing statistics on the dataset provided data with respect the ICC(3,k) 
    where k is the number of judges and min_val and max_val are used to determine
    the range of possible ranks (used for normalization).
    
    ICC: (listof (listof (anyof float int))), float, float -> float, (dictof str: float)

    example:
        data = [
            [9, 2, 5, 8],
            [6, 1, 3, 2],
            [8, 4, 6, 8],
            [7, 1, 2, 6],
            [10, 5, 6, 9],
            [6, 2, 4, 7]
        ]

        # where 9,2,5,8 are the votes from judge 0,1,2,3 for target 0

        ICC(data, 1, 10) -> 0.91, {'BMS': 11.241, 'EMS': 1.019, 'JMS':32.486}
    """
    pre = 'ICC'
    log('start',pre=pre)
    log('data',data,pre=pre)
    # 0. normalize the data and create the constants
    new_data = [row[:] for row in data]
    for i,row in enumerate(data):
        for j,val in enumerate(row):
            domain_width = max_val - min_val
            new_val = (val - min_val)/domain_width if normalize else val
            new_data[i][j] = new_val
    log('new_data',new_data,pre=pre)
    num_of_judges = len(new_data[0])
    num_of_targets = len(new_data)
    log('num_of_judges',num_of_judges,pre=pre)
    log('num_of_targets',num_of_targets,pre=pre)

    # 1. calculate the means
    # 1.a calculate the row menas
    row_means = [sum(row)/num_of_judges for i,row in enumerate(new_data)]
    log('row_means',row_means,pre=pre)
    # 1.b calculate the column means
    col_means = [sum([row[j] for row in new_data])/num_of_targets for j in range(num_of_judges)]
    log('col_means',col_means,pre=pre)
    # 1.c calculate the total mean
    tot_mean = sum([sum(row) for row in new_data])/(num_of_judges * num_of_targets)
    log('tot_mean',tot_mean,pre=pre)

    # 2. do the sum of squares (SS)
    # 2.a row SS
    row_SS = [(val-tot_mean)**2 for val in row_means]
    log('row_SS',row_SS,pre=pre)
    # 2.b col SS
    col_SS = [(val-tot_mean)**2 for val in col_means]
    log('col_SS',col_SS,pre=pre)
    # 2.c all SS
    all_SS = [[(new_data[i][j]-tot_mean)**2 for j in range(num_of_judges)] for i in range(num_of_targets)]
    log('all_SS', all_SS, pre=pre)

    # 3. do the aggregated sum of squares
    SSB = sum(row_SS)*num_of_judges
    SSC = sum(col_SS)*num_of_targets
    SST = sum([sum(row) for row in all_SS])
    SSE = SST - SSC - SSB
    log('SSB',SSB,pre=pre)
    log('SSC',SSC,pre=pre)
    log('SST',SST,pre=pre)
    log('SSE',SSE,pre=pre)

    # 4. incorporate the degrees of freedom
    JMS = SSC/(num_of_judges-1)
    EMS = SSE/((num_of_judges-1) * (num_of_targets-1))
    BMS = SSB/(num_of_targets-1)
    log('JMS',JMS,pre=pre)
    log('EMS',EMS,pre=pre)
    log('BMS',BMS,pre=pre)

    # 5. do the final calc
    icc = (BMS-EMS)/BMS

    return icc,{'BMS':BMS,'EMS':EMS, 'JMS':JMS}

def Kappa(data):
    """
    (Kappa data) This function does a linear weighted kappa based on the evaluations of two judges (ranksX and ranksY)
    """
    ranksX = [row[0] for row in data]
    ranksY = [row[1] for row in data]
    return float(cohen_kappa_score(ranksX, ranksY, weights='linear'))

if __name__ == '__main__':
    logger.deleteLogs()
    # testing
    tolerance = 0.0000001
    print('Testing ICC...')
    # test the example from the ICC article
    d = [
            [9,2,5,8],
            [6,1,3,2],
            [8,4,6,8],
            [7,1,2,6],
            [10,5,6,9],
            [6,2,4,7]
            ]
    exp_ans = (0.9093155423770697, {'BMS':11.241666666666669,'EMS':1.019444444444442,'JMS':32.486111111111114})
    # ans = ICC(d,1,10,normalize=False)
    # print('Test 1 Passed') if ans == exp_ans else print('Test 1 Failed: Got',ans,'instead of',exp_ans)
    exp_ans = 0.909315542377069
    # ans = ICC(d,1,10,normalize=True)[0]
    # print('Test 2 Passed') if (exp_ans-tolerance) < ans < (exp_ans+tolerance) else print('Test 2 Failed: Got',ans,'instead of',exp_ans)
    # test with the oni data from the ICC_Manusal.ods spreadsheet
    oni_pn = [
        [10.4137931034483, 4.07142857142857, 3.46082944405092],
        [12.2, 6.25641025641026, 3.30217983486781],
        [5, -3, 3.11162597867751],
        [9.33333333333333, 8, 3.85256570052427],
        [12.6666666666667, -1.33333333333333, 3.1574060827354],
        [8, 0, 3.65346148266967],
        [12, 5.33333333333333, 3.05538968552116],
        [10.6666666666667, 1, 3.13644034421027],
        [9, 7, 3.2783762418698],
        [12.6666666666667, 6.66666666666667, 4.19594471745875],
        [12, 2, 2.81019599676291]
    ]
    exp_ans = 0.445236219967768
    # ans = ICC(oni_pn,-18,18)[0]
    # print('Test 3 Passed') if (exp_ans-tolerance) < ans < (exp_ans+tolerance) else print('Test 3 Failed: Got',ans,'instead of',exp_ans)
    oni_fb = [
        [5.65517241379311, 1.57142857142857, -1.15269179885576],
        [2.55, -0.512820512820514, -0.393730884333156],
        [9, -2, -0.480575998593633],
        [9.33333333333333, -0.666666666666667, -0.39840414694463],
        [6, -5.33333333333333, 0.896628852738834],
        [8, 8, -0.461607459795788],
        [6, 1.33333333333333, 0.736289328414241],
        [5.33333333333333, 6, 0.129856194480237],
        [1, 4, -0.175604599459016],
        [4, 4, -0.165356686490746],
        [6, 4, -0.165356686490746]
    ]
    exp_ans = 0.956980565511847
    ans = ICC(oni_fb,-18,18)[0]
    # print('Test 4 Passed') if (exp_ans-tolerance) < ans < (exp_ans+tolerance) else print('Test 4 Failed: Got',ans,'instead of',exp_ans)

    print('Testing Kappa...')
    ranksX0 = [1]*8+[1]*2+[1]+[2]*3+[2]*11+[2]*5+[3]*7+[3]*55+[3]*11+[4]*1+[4]*11+[5]*2
    ranksY0 = [1]*8+[2]*2+[3]+[1]*3+[2]*11+[3]*5+[2]*7+[3]*55+[4]*11+[3]*1+[4]*11+[5]*2
    def translate(matrix):
        Xs = []
        Ys = []
        for i,row in enumerate(matrix):
            for j,val in enumerate(row):
                Xs += [i+1]*val
                Ys += [j+1]*val
        return Xs, Ys
    ranks = [[8, 2, 1, 0, 0],
            [3, 11, 5, 0, 0],
            [0, 7, 55, 11, 0],
            [0, 0, 1, 11, 0],
            [0, 0, 0, 0, 2]]
    exp_ans = ranksX0, ranksY0
    ans = translate(ranks)
    print('Test 5 Passed') if exp_ans == ans else print('Test 5 Failed: Got',ans,'instead of',exp_ans)
    # simple example
    ranks = [[8, 2, 1],
            [3,11,5],
            [0,7,55]]
    tot = sum([val for row in ranks for val in row])
    row_margins = [sum(row)/tot for row in ranks]
    col_margins = [sum([row[j] for row in ranks])/tot for j in range(len(ranks[0]))]
    p0 = sum([ranks[i][j]/tot*(1-abs(i-j)/(3-1)) for i,row in enumerate(ranks) for j,val in enumerate(row)])
    pe = sum([row_margins[i]*col_margins[j]*(1-abs(i-j)/(3-1))for i,row in enumerate(ranks) for j,val in enumerate(row)])
    k = (p0-pe)/(1-pe)
    exp_ans = k
    Xs,Ys = translate(ranks)
    data = [[Xs[i],Ys[i]] for i in range(len(Xs))]
    ans = Kappa(data)
    print('Test 6 Passed') if (exp_ans-tolerance) < ans < (exp_ans+tolerance) else print('Test 6 Failed: Got',ans,'instead of',exp_ans)






























































    # calcICC(['judge1','judge2','judge3','judge4'],d)
    # datafile = "ToothGrowth.csv"
    # d = {'judge1':[9,6,8,7,10,6],
    #         'judge2':[2,1,4,1,5,2],
    #         'judge3':[5,3,6,2,6,4],
    #         'judge4':[8,2,8,6,9,7]}
    # # data = pd.read_csv(datafile)
    # data = pd.DataFrame(data=d)
    # print(data)

    # N = len(data.len)
    # df_a = len(data.supp.unique()) - 1
    # df_b = len(data.dose.unique()) - 1
    # df_axb = df_a*df_b 
    # df_w = N - (len(data.supp.unique())*len(data.dose.unique()))
    # grand_mean = data['len'].mean()
            
    # ssq_a = sum([(data[data.supp ==l].len.mean()-grand_mean)**2 for l in data.supp])
    # ssq_b = sum([(data[data.dose ==l].len.mean()-grand_mean)**2 for l in data.dose])
    # ssq_t = sum((data.len - grand_mean)**2)

    # vc = data[data.supp == 'VC']
    # oj = data[data.supp == 'OJ']
    # vc_dose_means = [vc[vc.dose == d].len.mean() for d in vc.dose]
    # oj_dose_means = [oj[oj.dose == d].len.mean() for d in oj.dose]
    # ssq_w = sum((oj.len - oj_dose_means)**2) +sum((vc.len - vc_dose_means)**2)

    # ssq_axb = ssq_t-ssq_a-ssq_b-ssq_w

    # ms_a = ssq_a/df_a
    # ms_b = ssq_b/df_b
    # ms_axb = ssq_axb/df_axb
    # ms_w = ssq_w/df_w
