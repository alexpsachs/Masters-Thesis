"""
The purpose of this module is to take the text from allText.json and turn it into personalities (Big 5 AND SYMLOG)
and then throw that information into /home/a2sachs/Documents/Experiment1/Data/IBMWatson/personalities.ods
"""
import json
import sys
from pyexcel_ods3 import save_data
from collections import OrderedDict
LIB = '/home/a2sachs/Documents/Library'
sys.path.append(LIB)
import IBM
import SYMLOG


# 0. setup the paths
text_filepath = '/home/a2sachs/Documents/Experiment1/Data/IBMWatson/allText.json'  
dest_filepath = '/home/a2sachs/Documents/Experiment1/Data/IBMWatson/personalities.ods'

# 1. load up the text from the json
text = None
with open(text_filepath, 'r') as f:
    text = json.load(f)

# 2. Convert that text into Big 5 personas
big5Personas = {} # {person:{big5Attribute:score_from_0_to_1}}
for person,txt in text.items():
    persona = None
    try:
        persona = IBM.getBig5FromText(txt)
    except:
        persona = 'N/A'
        print(person, ' probably does not have enough words')
    big5Personas[person] = persona

# 3. Convert the Big 5 into SYMLOG personalities
SYMLOGPersonas = SYMLOG.convertBig5ToSYMLOG(big5Personas) # {person:{SYMLOGAxis:score_from_-18_to_+18}}

# 4. Write the personalities into a ods file
data = OrderedDict()
# Big 5 sheet
header = ['Person','Openness','Conscientiousness','Extraversion','Agreeableness','Neuroticism','P_N','F_B','U_D']
sheet = []
sheet.append(header)
for person,big5 in big5Personas.items():
    sym = SYMLOGPersonas[person]
    row = [person]
    if big5 != 'N/A':
        row.append(big5['openness'])
        row.append(big5['conscientiousness'])
        row.append(big5['extraversion'])
        row.append(big5['agreeableness'])
        row.append(big5['neuroticism'])
    else:
        for i in range(5):
            row.append('N/A')
    if sym != 'N/A':
        row.append(sym['p_n'])
        row.append(sym['f_b'])
        row.append(sym['u_d'])
    else:
        for i in range(3):
            row.append('N/A')
    sheet.append(row)

data.update({"OniData": sheet})
save_data(dest_filepath, data)
print('sheet',sheet)
