import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.utils import check_random_state
#Wrapper functions
import wrapper
#Parameter estimation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
#SVR model
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import VotingRegressor
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

test_size = [1440, 1440*5]

for i in test_size:

    dict_test_X = {}
    dict_test_y = {}
    dict_test_dates = {}

    list_train_X = []
    list_train_y = []

    estimators = []   

    for file in onlyfiles:

        #Read file and change date type
        filename = os.path.splitext(file)[0]
        dataset = pd.read_csv('data/'+filename+'.csv', delimiter=";", parse_dates=True)
        dataset['Time'] = pd.to_datetime(dataset.Time)
        print("Working with file: ", filename)

        #Remove outliers
        dataset = wrapper.remove_outliers(dataset, 2)

        #Filter dataset
        dataset = wrapper.filter_data(dataset, 41, 1)

        #Get min and max
        min_CO2=dataset['CO2'].min()
        max_CO2=dataset['CO2'].max()

        #Split between train and test
        data_train, data_test = train_test_split(dataset, test_size=i, random_state=55, shuffle = True)

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

        # Save the train data in order to train the ensemble later
        list_train_X.append(X_train)
        list_train_y.append(y_train)

        # Save the test data in order to test the ensemble later
        dict_test_X[filename] = X_test
        dict_test_y[filename] = y_test
        dict_test_dates[filename] = date_test   

        #Scale data
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

        estimators.append([filename,regressor.best_estimator_])

        #Predict test values
        y_true, y_pred = y_test, regressor.best_estimator_.predict(X_test)

        #Normalize outputs to get normalized scores
        y_true_norm = [ ((i-min_CO2)/(max_CO2-min_CO2)) for i in y_true ]
        y_pred_norm = [ ((i-min_CO2)/(max_CO2-min_CO2)) for i in y_pred ]

        #Calculate the score
        results = wrapper.regression_results(y_true_norm, y_pred_norm)     
        results['train_size'] = i
        results['test_size'] = i
        results['file'] = filename
        print(pd.DataFrame(results, index=[0]))

    #Concat the train data in order to use for training the ensemble
    ens_X_train = pd.concat(list_train_X)
    ens_y_train = pd.concat(list_train_y)

    #Scale train data
    sc_X = StandardScaler()
    ens_X_train = sc_X.fit_transform(ens_X_train)
    
    # Create ensemble model and fit train data in it
    print('Fitting data...')
    ensemble_model = VotingRegressor(estimators=estimators)
    ensemble_model.fit(ens_X_train, ens_y_train)

    #Test data from the different rooms
    for X_index, y_index, date_index in zip(dict_test_X, dict_test_y, dict_test_dates):
        X_test = sc_X.transform(dict_test_X[X_index])
        y_test = dict_test_y[y_index]
        date_test = dict_test_dates[date_index]
        y_true, y_pred = y_test, ensemble_model.predict(X_test)

        #Normalize outputs to get normalized scores
        y_true_norm = [ ((i-min_CO2)/(max_CO2-min_CO2)) for i in y_true ]
        y_pred_norm = [ ((i-min_CO2)/(max_CO2-min_CO2)) for i in y_pred ]

        results = wrapper.regression_results(y_true_norm, y_pred_norm)
        results['train_size'] = i
        results['test_size'] = i
        results['file'] = X_index
        output.append(results)
        folder = 'no_kitchen_bathroom_voting'
        wrapper.plot_accuracy(y_true, y_pred, int(y_true.size/20), i, i, results['r2'], X_index, folder, sorted(date_test))

results_df = pd.DataFrame.from_dict(output)
print(results_df)

results_df.to_csv('predictions/ensemble_voting_no_kitchen_bathroom.csv', sep=';')





