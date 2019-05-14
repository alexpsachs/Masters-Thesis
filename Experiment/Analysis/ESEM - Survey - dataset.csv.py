# the purpose of this script is to explore ESEM -Survey dataset.csv

import os
import pandas as pd
import code
filepath = os.path.abspath(os.path.join(__file__,'../ESEM - Survey - dataset.csv'))
print('filepath',filepath)
df = pd.read_csv(filepath)
print(df['Result'].value_counts())

