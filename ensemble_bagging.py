import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.utils import check_random_state
#Wrapper functions
import wrapper
#Parameter estimation
from sklearn.model_selection import ParameterGrid
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

#Read file
filename = 'merged_no_kitchen_bathroom'
dataset = pd.read_csv('merged_data/'+filename+'.csv', delimiter=";", parse_dates=True)
dataset = dataset.drop(['id'], axis=1)

dataset = wrapper.remove_outliers(dataset, 2)

#Filter dataset data
dataset = wrapper.filter_data(dataset, 41, 1)

#Get min and max
min_CO2 = dataset['CO2'].min()
max_CO2 = dataset['CO2'].max()

output = []
tests = [1440, 1440*5]
regressor = []
for i in range(len(tests)):
    #Split between train and test
    data_train, data_test = train_test_split(dataset, test_size=tests[i], random_state=55, shuffle = True)

    #Split between X and y
    X_train = data_train.drop(['CO2', 'Time'], axis=1).copy()
    y_train = data_train['CO2']

    X_test = data_test.drop(['CO2', 'Time'], axis=1).copy()
    y_test = data_test[['CO2', 'Time']]

    date_test = y_test['Time']

    y_test = y_test['CO2'].to_numpy()

    #Normalize data
    min_H = X_train['H'].min()
    max_H = X_train['H'].max()

    min_T = X_train['T'].min()
    max_T = X_train['T'].max()

    X_train['H'] = (X_train['H']-min_H)/(max_H-min_H)
    X_train['T'] = (X_train['T']-min_T)/(max_T-min_T)

    X_test['H'] = (X_test['H']-min_H)/(max_H-min_H)
    X_test['T'] = (X_test['T']-min_T)/(max_T-min_T)

    #Scale data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    #If first iteration, create the model and estimate parameters
    if i ==0:
        rng = check_random_state(0)
        grid = ParameterGrid({"max_samples": [0.5, 1.0],
                            "max_features": [0.5, 1.0],
                            "bootstrap": [True, False]})
        for params in grid:
            regressor= BaggingRegressor(base_estimator=SVR(),
                            verbose=10,
                            random_state=rng,
                            **params).fit(X_train, y_train)

    #Predict the test data with the best values found
    y_true, y_pred = y_test, regressor.predict(X_test)

    #Normalize outputs to get normalized scores
    y_true_norm = [ ((i-min_CO2)/(max_CO2-min_CO2)) for i in y_true ]
    y_pred_norm = [ ((i-min_CO2)/(max_CO2-min_CO2)) for i in y_pred ]

    #Save the results for each test
    results = wrapper.regression_results(y_true_norm, y_pred_norm)
    results['train_size'] = tests[i]
    results['test_size'] = tests[i]
    results['file'] = 'merged'
    output.append(results)
    folder = 'no_kitchen_bathroom_bagging'
    wrapper.plot_accuracy(y_true, y_pred, int(y_pred.size/20), tests[i], tests[i], results['r2'], 'merged', folder, sorted(date_test) )

results_df=pd.DataFrame.from_dict(output)
print(results_df)
results_df.to_csv('predictions/ensemble_bagging_no_kitchen_bathroom.csv', sep=';')


