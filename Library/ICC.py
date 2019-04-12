"""
This module is intended to used to calculate ICC given a dataset
"""
import pandas as pd
from statsmodels.stats.anova import AnovaRM

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

def ICC(data, colHeaders=None, rowHeaders=None):
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


if __name__ == '__main__':
    # testing
    d = [
            [9,2,5,8],
            [6,1,3,2],
            [8,4,6,8],
            [7,1,2,6],
            [10,5,6,9],
            [6,2,4,7]
            ]
    # calcICC(['judge1','judge2','judge3','judge4'],d)
    print(ICC(d))
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
