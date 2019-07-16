"""
The purpose of this module is to do various inter-rater agreement measures on
/home/a2sachs/Documents/Experiment1/Data/Oni_SYMLOGs.ods
in order to evaluate IBM Watson
"""
import os,sys,math,code
TOP = os.path.abspath(os.path.join(__file__,'../../..'))
LIB = os.path.join(TOP,'Library')
sys.path.append(LIB)
import InterRaterAgreement as IRA
import logger
from pyexcel_ods3 import get_data, save_data
from collections import OrderedDict
def log(*args,pre=None):
    logger.log(*args,pre='IRA_measures' if pre == None else 'IRA_measures.'+pre)

if __name__ == '__main__':
    logger.deleteLogs()
    log('start')
    # 0. Setup the paths
    # input_filepath = '/home/a2sachs/Documents/Experiment1/Data/Oni_SYMLOGs.ods'
    input_filepath = os.path.join(TOP,'Pilot','Data','Oni_SYMLOGs.ods')
    # dest_filepath = '/home/a2sachs/Documents/Experiment1/Analysis/ICCs.ods'
    dest_filepath = os.path.join(TOP,'Pilot','Analysis','IRA_measures.ods')
    dest_filepath_kappa = os.path.join(TOP,'Pilot','Analysis','IRA_measures_kappa.ods')

    # 1. Import the data
    data = {}
    header = get_data(input_filepath)['OniData'][0]

    Person = header.index('Person')
    IBM_P_N = header.index('IBM_P_N')
    IBM_F_B = header.index('IBM_F_B')	
    IBM_U_D = header.index('IBM_U_D')	
    Alex_P_N = header.index('Alex_P_N')	
    Alex_F_B = header.index('Alex_F_B')	
    Alex_U_D = header.index('Alex_U_D')	
    Andrew_P_N = header.index('Andrew_P_N')	
    Andrew_F_B = header.index('Andrew_F_B')
    Andrew_U_D = header.index('Andrew_U_D')

    for row in get_data(input_filepath)['OniData'][1:]:
        data[row[Person]] = {   
                'IBM_P_N':row[IBM_P_N],
                'IBM_F_B':row[IBM_F_B],
                'IBM_U_D':row[IBM_U_D],
                'Alex_P_N':row[Alex_P_N],
                'Alex_F_B':row[Alex_F_B],
                'Alex_U_D':row[Alex_U_D],
                'Andrew_P_N':row[Andrew_P_N],
                'Andrew_F_B':row[Andrew_F_B],
                'Andrew_U_D':row[Andrew_U_D]
            }

    # 2. Format the data for ICC function
    def refine(judgeLst,axis):
        # Given the judges, and an axis (P_N, F_B, or U_D) this function returns a formatted version 
        pre = 'refine'
        log('start',pre=pre)
        d = []
        for person,votes in data.items():
            # eliminate any 'N/A' rows
            if len([val for key,val in votes.items() if val == 'N/A']) == 0:
                row = [val for key,val in votes.items()\
                        if key.find(axis) != -1 and\
                        key[:key.find('_')] in judgeLst 
                        ] # If the key is of the right axis and is of one of the judges, then accept it
                d.append(row)
        log('d for judges',judgeLst,'for axis',axis,'is',d,pre=pre)
        return d

    # 3. Do the calculations
    headers = ['Axis','IBM+Alex','IBM+Andrew','Alex+Andrew','All']
    combinations = [('IBM','Alex'),
            ('IBM','Andrew'),
            ('Alex','Andrew'),
            ('IBM','Andrew','Alex')]
    axes = ['P_N','F_B','U_D']
    sheet = []
    k_sheet = []
    sheet.append(headers)
    k_sheet.append(headers)
    for axis in axes:
        row = [axis]
        k_row = [axis]
        for combo in combinations:
            calcs = IRA.ICC(refine(combo,axis),-18,18)
            row.append(calcs[0])
            if combo != ('IBM','Andrew','Alex'):
                d = [[round((val+18)*1000) for val in row] for row in refine(combo,axis)]
                k = IRA.Kappa(d)
                k_row.append(k)
            else:
                k_row.append('N/A')
        sheet.append(row)
        k_sheet.append(k_row)
    data = OrderedDict()
    data.update({'ICCs':sheet, 'Kappa':k_sheet})
    # code.interact(local=dict(globals(),**locals()))
    log('data',data)
    save_data(dest_filepath,data)

