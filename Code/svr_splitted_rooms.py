import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.utils import check_random_state
#Wrapper functions
import wrapper
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


data_path = 'data/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

output = []
for file in onlyfiles:

    #Read file and change date type
    filename = os.path.splitext(file)[0]
    dataset = pd.read_csv('data/'+filename+'.csv', delimiter=";", parse_dates=True)
    dataset['Time'] = pd.to_datetime(dataset.Time)

    dataset = wrapper.remove_outliers(dataset, 2)

    #Filter dataset
    dataset = wrapper.filter_data(dataset, 41, 1)

    #Get min and max
    min_CO2 = dataset['CO2'].min()
    max_CO2 = dataset['CO2'].max()

    test_size = [1440, 7200]
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

        #Normalize outputs to get normalized scores
        y_true_norm = [ ((i-min_CO2)/(max_CO2-min_CO2)) for i in y_true ]
        y_pred_norm = [ ((i-min_CO2)/(max_CO2-min_CO2)) for i in y_pred ]

        #Calculate the score
        results = wrapper.regression_results(y_true_norm, y_pred_norm)      
        results['train_size'] = i
        results['test_size'] = i
        results['file'] = filename
        output.append(results)
        folder = 'random_split'
        wrapper.plot_accuracy(y_true, y_pred, int(y_true.size/20), i, i, results['r2'], filename, folder, sorted(date_test))
    
results_df = pd.DataFrame.from_dict(output)
print(results_df)
results_df.to_csv('predictions/results_shuffle.csv', sep=';')




