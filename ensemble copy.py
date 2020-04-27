import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import itertools
#Parameter estimation
import optunity
import optunity.metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
#SVR model
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import BaggingRegressor
#Feature scaling (Normalize)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#Noise reduction
from scipy.signal import savgol_filter
#Metrics
from scipy import stats
import sklearn.metrics as metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
#Visualize
import matplotlib.pyplot as plt
#File management
import os.path
import os, glob
import csv
from os import listdir
from os.path import isfile, join

def plot_accuracy(y_true, y_pred, xax_, t_s, train_size, score, test_dates, filename):

    xax=xax_
    ig=plt.figure(figsize=(25, 12), dpi= 80, facecolor='w', edgecolor='k')
    plt.plot(y_true[0::xax], label='y_true')
    plt.plot(y_pred[0::xax], label='y_pred')
    min_y_bound = min(min(y_true), min(y_true))
    max_y_bound = max(max(y_true), max(y_true))
    step = abs(max_y_bound-min_y_bound)/50

    plt.yticks(np.arange(min_y_bound, max_y_bound+step, step))
    plt.xticks(np.arange(0, y_true[0::xax].size+1,1), test_dates[0::xax], rotation=70)

 
    plt.xlabel('Instances')
    plt.ylabel('CO2')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("graphs/"+filename+"_score{score:.3f}_test{test_s:.4f}_train{train:.3f}.png".format(test_s=t_s, score = score, train=train_size))
    fname = "predictions/"+filename+"_score{score:.3f}_test{test_s:.4f}_train{train:.3f}.csv".format(test_s=t_s, score = score, train=train_size)
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    df = pd.DataFrame({"y_true" : y_true, "y_pred" : y_pred})
    df.to_csv(fname, index=False, sep=";")

def regression_results(y_true, y_pred):
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    r2=metrics.r2_score(y_true, y_pred)
    metrics_dic = {'r2':round(r2,4), 'mae':round(mean_absolute_error,4), 'mse':round(mse,4), 'rmse':round(np.sqrt(mse),4)}
    return metrics_dic  

def filter_data(data, wl, po):
    dataset_filtered = data.copy()
    dataset_filtered['CO2']= savgol_filter(dataset_filtered['CO2'], wl, po)
    dataset_filtered['H']= savgol_filter(dataset_filtered['H'], wl, po)
    dataset_filtered['T']= savgol_filter(dataset_filtered['T'], wl, po)
    return dataset_filtered

def normalize_data(data, min_, max_):
    dataset_normalized = data.copy()
    dataset_normalized['CO2'] = ((data['CO2']-min_['CO2'])/(max_['CO2']-min_['CO2']))
    dataset_normalized['H'] = ((data['H']-min_['H'])/(max_['H']-min_['H']))
    dataset_normalized['T'] = ((data['T']-min_['T'])/(max_['T']-min_['T']))
    return dataset_normalized

def inverse_normalize(x, min_, max_):
    for i, ob in enumerate(x):
        x[i]=(x[i]*(max_-min_)+min_)
    return x

def get_percentage(x, total):
    return ((x*100)/total)/100

mypath = 'data/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

data_train = []
data_test = {}

min_max = []
for file in onlyfiles:
    filename = os.path.splitext(file)[0]
    dataset = pd.read_csv('data/'+filename+'.csv', delimiter=";", parse_dates=True)
    dataset = filter_data(dataset, 21, 1)
    min_max.append(list(dataset.drop(['Time'], axis=1).min()))
    min_max.append(list(dataset.drop(['Time'], axis=1).max()))
    train_file, test_file = train_test_split(dataset, train_size=0.7, test_size=0.3, random_state=55, shuffle=False)
    data_train.append(train_file)
    data_test[filename]=test_file

df_minmax = pd.DataFrame(min_max, columns=['CO2', 'H', 'T'])

df_train = pd.concat(data_train)    
    
normalized_train = normalize_data(df_train, df_minmax.min(), df_minmax.max())    
    
#Split train data into X and y
X = normalized_train.drop(labels=['Time', 'CO2'], axis=1)
y = normalized_train.iloc[:,0:2]

#Fit the scale to scale the data later
sc_X = StandardScaler()
sc_X.fit(X)

train_size = 0.7
test_size = 0.3

#Split between train and test
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=55, shuffle=False)
y_train=y_train['CO2'].to_numpy()
test_dates = y_valid['Time']
y_valid = y_valid['CO2'].to_numpy()
#Scale the train and test data
X_train = sc_X.transform(X_train)
X_valid = sc_X.transform(X_valid)

#Create the ranges of the parameters for the estimation
gamma_range = np.logspace(0.0001,9,base=2, num=40)
C_range = np.logspace(0.0001,9,base=2, num=40)
epsilon_range = [0.01, 0.05, 0.1, 0.125, 0.2, 0.5, 0.7, 0.9]
tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma_range, 'C': C_range, 'epsilon':epsilon_range }]

#Ensemble
regressor = BaggingRegressor(base_estimator=SVR(), n_estimators=100, random_state=0)

#Estimate the parameters
#regressor = RandomizedSearchCV(SVR(), tuned_parameters, scoring='r2', cv=3, verbose=1, n_iter=10)

#Train the model with the best parameters
regressor.fit(X_train, y_train)

#Predict the validation data
y_true, y_pred  = y_valid, regressor.predict(X_valid)

#Get the results of the validation
output = regression_results(y_true, y_pred)
print('Validation results: ', output)

y_true = inverse_normalize(y_true, df_minmax['CO2'].min(), df_minmax['CO2'].max())
y_pred = inverse_normalize(y_pred, df_minmax['CO2'].min(), df_minmax['CO2'].max())
y_true = y_true.reshape(-1,1)
y_pred = y_pred.reshape(-1,1)

plot_accuracy(y_true, y_pred, int(y_true.size/50), test_size, train_size, output['r2'], test_dates, 'validation')



for i in data_test:
    normalized_test = normalize_data(data_test[i], df_minmax.min(), df_minmax.max())
    #Get different test sizes (6h and 1 day)
    _, test_1 = train_test_split(normalized_test, test_size=get_percentage(360,normalized_test.shape[0]), shuffle=False)
    _, test_2 = train_test_split(normalized_test, test_size=get_percentage(1440,normalized_test.shape[0]), shuffle=False)

    X_test1 = test_1.drop(['Time', 'CO2'], axis=1)
    y_test1 = test_1['CO2'].to_numpy()
    test_dates1 =test_1['Time']

    X_test2 = test_2.drop(['Time', 'CO2'], axis=1)
    y_test2 = test_2['CO2'].to_numpy()
    test_dates2 =test_2['Time']
    X_test1 = sc_X.transform(X_test1)
    X_test2 = sc_X.transform(X_test2)

    y_true1, y_pred1 = y_test1, regressor.predict(X_test1)
    y_true2, y_pred2 = y_test2, regressor.predict(X_test2)

    y_true1 = inverse_normalize(y_true1, df_minmax['CO2'].min(), df_minmax['CO2'].max())
    y_pred1 = inverse_normalize(y_pred1, df_minmax['CO2'].min(), df_minmax['CO2'].max())
    y_true1 = y_true1.reshape(-1,1)
    y_pred1 = y_pred1.reshape(-1,1)

    y_true2 = inverse_normalize(y_true2, df_minmax['CO2'].min(), df_minmax['CO2'].max())
    y_pred2 = inverse_normalize(y_pred2, df_minmax['CO2'].min(), df_minmax['CO2'].max())
    y_true2 = y_true2.reshape(-1,1)
    y_pred2 = y_pred2.reshape(-1,1)

    output = regression_results(y_true1, y_pred1)
    print('Test 1 results: ', output)

    plot_accuracy(y_true1, y_pred1, int(y_true1.size/50), test_size, train_size, output['r2'], test_dates1, str(i))

    output = regression_results(y_true2, y_pred2)
    print('Test 2 results: ', output)

    plot_accuracy(y_true2, y_pred2, int(y_true2.size/50), test_size, train_size, output['r2'], test_dates2, str(i))