"""
The purpose of this script is to analyze the data in 
./experiment_data.ods
"""
from pyexcel_ods3 import read_data
import pandas
import code
from sklearn import preprocessing
import statsmodels.api as sm
import os

INDIR = os.path.abspath('./experiment_data')

# functions
def analyzeExperiment(exp_name,pre=''):
    pass
# read in the data
ods_data = read_data(in_path)['Data']
headers = ods_data[0]
data_dict = {label:[] for label in headers}
for row in ods_data[1:]:
    for i,label in enumerate(headers):
        data_dict[label].append(row[i])
df = pandas.DataFrame(data=data_dict)

# transform the data by normalizing and binarizing esem_status
df.loc[df['esem_status'] == 'active', 'esem_status'] = 1
df.loc[df['esem_status'] == 'inactive', 'esem_status'] = 0
df = df.drop(columns=['reponame'])
columns = df.columns.values
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pandas.DataFrame(x_scaled)
df.columns = columns

# do multiple linear regression
X = df.drop(columns=['esem_status'])
Y = df['esem_status']
X = sm.add_constant(X)
model = sm.OLS(Y,X).fit()
print(model.summary())

def p(name):
    # plot the points
    x_vals = df[name].values
    y_vals = df['esem_status'].values
    import matplotlib.pyplot as plt
    # plt.plot([1, 2, 3, 4],[2,4,6,8])
    plt.plot(range(len(x_vals)),x_vals,'ro')
    plt.plot(range(len(y_vals)),y_vals,'bo')
    plt.ylabel(name)
    # plt.show()
    plt.savefig('/home/a2sachs/'+name+'.png')
    plt.close()
p('pf_prop')
p('dissidence')
p('upf_corr')
p('rot_regret')
p('opp_corr')
