"""
The purpose of this module is to consolidate the SYMLOG personalities of the Oni project for IBM Watson
(/home/alex/a2sachs/Documents/Experiment1/Data/IBMWatson/personalities.ods), and for Alex and Andrew
(/home/alex/a2sachs/Dropbox/Masters/Thesis/Scripts/ThesisSiteDev/VotingBank/Oni_SYMLOG.ods) into Oni_SYMLOGs.ods
"""
from pyexcel_ods3 import get_data, save_data
from collections import OrderedDict
# 0. Setup the paths
IBMSYMLOG_filepath = '/home/a2sachs/Documents/Experiment1/Data/IBMWatson/personalities.ods'
Oni_SYMLOG_filepath = '/home/a2sachs/Dropbox/Masters/Thesis/Scripts/ThesisSiteDev/VotingBank/Oni_SYMLOG.ods'
dest_filepath = '/home/a2sachs/Documents/Experiment1/Data/Oni_SYMLOGs.ods'

# 1. Consolidate the data
IBMSYMLOGs = get_data(IBMSYMLOG_filepath)['OniData']
IBMSYMDict = {}
for row in IBMSYMLOGs[1:]:
    index = IBMSYMLOGs[0].index('Person')
    person = row[index]
    index = IBMSYMLOGs[0].index('P_N')
    p_n = row[index]
    index = IBMSYMLOGs[0].index('F_B')
    f_b = row[index]
    index = IBMSYMLOGs[0].index('U_D')
    u_d = row[index]
    IBMSYMDict[person] = {'P_N':p_n,
            'F_B':f_b,
            'U_D':u_d
            }
OniSYMLOGs = get_data(Oni_SYMLOG_filepath)
data = OrderedDict()
sheet = []
headers = ['Person','IBM_P_N','IBM_F_B','IBM_U_D','Alex_P_N','Alex_F_B','Alex_U_D','Andrew_P_N','Andrew_F_B','Andrew_U_D']
sheet.append(headers)
for row in OniSYMLOGs['SYMLOG'][1:]:
    index = OniSYMLOGs['SYMLOG'][0].index('Person')
    person = row[index]
    index = OniSYMLOGs['SYMLOG'][0].index('Alex_P_N')
    Alex_P_N = row[index]
    index = OniSYMLOGs['SYMLOG'][0].index('Alex_F_B')
    Alex_F_B = row[index]
    index = OniSYMLOGs['SYMLOG'][0].index('Alex_U_D')
    Alex_U_D = row[index]
    index = OniSYMLOGs['SYMLOG'][0].index('Andrew_P_N')
    Andrew_P_N = row[index]
    index = OniSYMLOGs['SYMLOG'][0].index('Andrew_F_B')
    Andrew_F_B = row[index]
    index = OniSYMLOGs['SYMLOG'][0].index('Andrew_U_D')
    Andrew_U_D = row[index]
    IBM_P_N = IBM_F_B = IBM_U_D = 'N/A'
    if person in IBMSYMDict:
        if IBMSYMDict[person] != 'N/A':
            entry = IBMSYMDict[person]
            IBM_P_N = entry['P_N']
            IBM_F_B = entry['F_B']
            IBM_U_D = entry['U_D']
    newRow = [person,IBM_P_N,IBM_F_B,IBM_U_D,Alex_P_N,Alex_F_B,Alex_U_D,Andrew_P_N,Andrew_F_B,Andrew_U_D]
    sheet.append(newRow)
data.update({"OniData":sheet})
save_data(dest_filepath, data)
