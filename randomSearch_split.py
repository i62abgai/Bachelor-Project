import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.utils import check_random_state
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

def plot_accuracy(y_true, y_pred, xax_, t_s, train_size, score, filename, dates):

    xax=xax_
    ig=plt.figure(figsize=(25, 12), dpi= 80, facecolor='w', edgecolor='k')
    plt.plot(y_true[0::xax], label='y_true')
    plt.plot(y_pred[0::xax], label='y_pred')
    min_y_bound = min(min(y_true), min(y_true))
    max_y_bound = max(max(y_true), max(y_true))
    step = abs(max_y_bound-min_y_bound)/50

    plt.yticks(np.arange(min_y_bound, max_y_bound+step, step))
    plt.xticks(np.arange(0, y_true[0::xax].size+1,1), dates[0::xax], rotation=70)

 
    plt.xlabel('Instances')
    plt.ylabel('CO2')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("graphs/random_split/"+filename+"_score{score:.3f}_test{test_s:.4f}_train{train:.3f}.png".format(test_s=t_s, score = score, train=train_size))
    fname = "predictions/random_split/"+filename+"_score{score:.3f}_test{test_s:.4f}_train{train:.3f}.csv".format(test_s=t_s, score = score, train=train_size)
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

def normalize_data(data):
    dataset_normalized = data.copy()
    dataset_normalized['CO2'] = ((dataset_normalized['CO2']-dataset_normalized['CO2'].min())/(dataset_normalized['CO2'].max()-dataset_normalized['CO2'].min()))
    dataset_normalized['H'] = ((dataset_normalized['H']-dataset_normalized['H'].min())/(dataset_normalized['H'].max()-dataset_normalized['H'].min()))
    dataset_normalized['T'] = ((dataset_normalized['T']-dataset_normalized['T'].min())/(dataset_normalized['T'].max()-dataset_normalized['T'].min()))
    return dataset_normalized

def inverse_normalize(x, min_, max_):
    for i, ob in enumerate(x):
        x[i]=(x[i]*(max_-min_)+min_)
    return x

def remove_outliers(dataset):
    z = np.abs(stats.zscore(dataset.iloc[:,1:4]))
    threshold = 3
    return dataset[(z < threshold).all(axis=1)]



mypath = 'data/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

output = []
for file in onlyfiles:

    #Read file and change date type
    filename = os.path.splitext(file)[0]
    dataset = pd.read_csv('data/'+filename+'.csv', delimiter=";", parse_dates=True)
    dataset['Time'] = pd.to_datetime(dataset.Time)

    dataset = remove_outliers(dataset)

    #Filter dataset
    dataset = filter_data(dataset, 41, 1)

    test_size = [1440, 1440*5]
    for i in test_size:

        #Split between X and y
        X = dataset.drop(['CO2', 'Time'], axis=1)
        y = dataset[['CO2', 'Time']]

        #Split between train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=55, shuffle = True)
        X_train = X_train.copy()
        X_test = X_test.copy()
        date_test = y_test['Time']

        y_test = y_test['CO2'].to_numpy()
        y_train = y_train['CO2'].to_numpy()

        #Scale data
        min_H = X_train['H'].min()
        max_H = X_train['H'].max()

        min_T = X_train['T'].min()
        max_T = X_train['T'].max()

        X_train['H'] = (X_train['H']-min_H)/(max_H-min_H)
        X_train['T'] = (X_train['T']-min_T)/(max_T-min_T)

        X_test['H'] = (X_test['H']-min_H)/(max_H-min_H)
        X_test['T'] = (X_test['T']-min_T)/(max_T-min_T)

        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)

        # Select tuning parameter range
        gamma_range = np.logspace(0.0001,9,base=2, num=40)
        C_range = np.logspace(0.0001,9,base=2, num=40)
        epsilon_range = [0.01, 0.05, 0.1, 0.125, 0.2, 0.5, 0.7, 0.9]
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma_range, 'C': C_range, 'epsilon':epsilon_range }]

        #Regressor parameter estimation
        regressor = RandomizedSearchCV(SVR(), tuned_parameters, scoring='r2', cv=3, verbose=1, n_iter=10)
        regressor.fit(X_train, y_train)

        #Predict test values
        y_true, y_pred = y_test, regressor.predict(X_test)

        #Calculate the score
        results = regression_results(y_true, y_pred)      
        results['train_size'] = i
        results['test_size'] = i
        results['file'] = filename
        output.append(results)
        plot_accuracy(y_true, y_pred, int(y_true.size/50), i, i, results['r2'], filename, sorted(date_test))
    
results_df = pd.DataFrame.from_dict(output)
print(results_df)
results_df.to_csv('predictions/results_shuffle.csv', sep=';')




