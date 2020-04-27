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

def inverse_scaler(y_true, y_pred):
    y_true = y_true.reshape(-1,1)
    y_pred = y_pred.reshape(-1,1)
    y_true = sc_y.inverse_transform(y_true)
    y_pred = sc_y.inverse_transform(y_pred)
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    return y_true, y_pred

def inverse_normalize(x):
    for i, ob in enumerate(x):
        x[i]=(x[i]*(dataset['CO2'].max()-dataset['CO2'].min())+dataset['CO2'].min())
    return x

def get_percentage(x, total):
    return ((x*100)/total)/100

def plot_accuracy(y_true, y_pred, xax_, step_, t_s, dates, file):
    y_true = inverse_normalize(y_true)
    y_pred = inverse_normalize(y_pred)
    y_true = y_true.reshape(-1,1)
    y_pred = y_pred.reshape(-1,1)
    xax=xax_
    ig=plt.figure(figsize=(25, 12), dpi= 80, facecolor='w', edgecolor='k')
    plt.plot(y_true[0::xax], label='y_true')
    plt.plot(y_pred[0::xax], label='y_pred')
    min_y_bound = min(min(y_true), min(y_true))
    max_x_bound = max(max(y_true), max(y_true))
    step = step_

    plt.yticks(np.arange(min_y_bound, max_x_bound+step, step))
    plt.xticks(np.arange(0, y_true[0::xax].size+1,1), dates[0::xax], rotation=70)
    #plt.gcf().subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.xlabel('Instances')
    plt.ylabel('CO2')
    plt.legend()
    plt.grid()
    plt.savefig("graphs/"+file+"_test{test_s:.4f}_score{score:.3f}.png".format(test_s=t_s, score = test_score))
    fname = "predictions/"+file+"_test{test_s:.4f}_score{score:.3f}.csv".format(test_s=t_s, score = test_score)
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


sc_X = StandardScaler()
sc_y = StandardScaler()

def scale_data(X,y):
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)
    y = np.ravel(y)
    return X,y


def split_data(data, train_size, test_size):
    return train_test_split(data, train_size=train_size, test_size=test_size, random_state=55, shuffle=False)

def split_XY(data):
    X = data.iloc[:,2:4].values.astype(float)
    y = data.iloc[:,1:2].values.astype(float)
    return X,y

def estimate_parameters(X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=10)
    gamma_range = np.logspace(0.0001,9,base=2, num=40)
    C_range = np.logspace(0.0001,9,base=2, num=40)
    epsilon_range = [0.01, 0.05, 0.1, 0.125, 0.2, 0.5, 0.7, 0.9]
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma_range, 'C': C_range, 'epsilon':epsilon_range }]
    #grid_regressor = GridSearchCV(SVR(), tuned_parameters, scoring='r2', cv=3)
    grid_regressor = RandomizedSearchCV(SVR(), tuned_parameters, scoring='r2', cv=tscv, verbose=1, n_iter=10)
    grid_regressor.fit(X_train, y_train)
    return grid_regressor

def create_model(C, gamma, epsilon,X_train, y_train):
    regressor = SVR(kernel='rbf',
                    C=C,
                    gamma=gamma,
                    epsilon=epsilon)
    regressor.fit(X_train,y_train)
    return regressor


def predict_data(regressor, X_, y_):
    y_true, y_pred = y_, regressor.predict(X_)
    return y_true, y_pred

mypath = 'data/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
models = []
test_ensemble= {}

#Iterate through files
for file in onlyfiles:
    filename = os.path.splitext(file)[0]
    dataset = pd.read_csv('data/'+filename+'.csv', delimiter=";", parse_dates=True)
    print('Working with: ', file)
    dataset['Time'] = pd.to_datetime(dataset.Time) 
    dataset['Timestamp'] = pd.to_datetime(dataset.Time,format='%d-%m-%Y %H:%M') 
    dataset.index = dataset.Timestamp
    
    #Data filtering
    print('Filtering data...')
    dataset['CO2']= savgol_filter(dataset['CO2'], 7, 1)
    dataset['H']= savgol_filter(dataset['H'], 7, 1)
    dataset['T']= savgol_filter(dataset['T'], 7, 1)
    
    #Data normalization
    dataset_normalized = dataset.copy()
    dataset_normalized['CO2'] = ((dataset['CO2']-dataset['CO2'].min())/(dataset['CO2'].max()-dataset['CO2'].min()))
    dataset_normalized['H'] = ((dataset['H']-dataset['H'].min())/(dataset['H'].max()-dataset['H'].min()))
    dataset_normalized['T'] = ((dataset['T']-dataset['T'].min())/(dataset['T'].max()-dataset['T'].min()))

    train_size = 0.1
    test_size = 0.4

    #Split between traind and test
    print('Data split (Traind and test)')
    print('\tTest instances: ', round(dataset.shape[0]*test_size,0))
    print('\tTrain instances: ', round(dataset.shape[0]*train_size,0))


    train_data, test_data = split_data(dataset_normalized,train_size, test_size)

    #Save test data for the ensemble
    test_ensemble[filename]=(test_data)

    #Split between X and y (train and test)
    X_train, y_train = split_XY(train_data)

    #Data scale
    print('Data scale')
    X_train,y_train = scale_data(X_train,y_train)

    #Parameter estimation
    print('Parameter tuning')
    grid_regressor=estimate_parameters(X_train, y_train)

    #Save best parameters
    C = grid_regressor.best_params_['C']
    gamma = grid_regressor.best_params_['gamma']
    epsilon = grid_regressor.best_params_['epsilon']
    print('Training model...')

    #Model creation
    regressor=create_model(C, gamma, epsilon, X_train, y_train)
    #Save model for ensemble
    models.append(regressor)


#Ouput dataframe to save the resultds
output = pd.DataFrame(columns=['r2', 'mae', 'mse', 'rmse', 'test_size'])

#Concatenate all the test data
#test_ensemble=pd.concat(test_ensemble)

ensemble_model = VotingRegressor(estimators=[('1', models[0]), ('2', models[1]), ('3', models[2]), ('4', models[3]), ('5', models[4])])

for j in test_ensemble:   

    #Get percentages
    one_hour = get_percentage(60, test_ensemble[j].shape[0])
    three_hour = get_percentage(180, test_ensemble[j].shape[0])
    six_hour = get_percentage(360, test_ensemble[j].shape[0])
    one_day = get_percentage(1440, test_ensemble[j].shape[0])
    three_day = get_percentage(3*1440, test_ensemble[j].shape[0])

    test_s = [one_hour, three_hour, six_hour, one_day, three_day]

   
    for i, o in enumerate(test_s):
        _, test_split = split_data(test_ensemble[j], None, test_s[i])
        #Split between X and y data
        X_test, y_test = split_XY(test_split)
        X_test,y_test = scale_data(X_test,y_test)

        y_true = y_test
        y_pred=ensemble_model.fit(X_test, y_test).predict(X_test)

        #Inverse the data scale
        y_true,y_pred = inverse_scaler(y_true, y_pred)

        print('Saving '' results...\n')
        #Get scores and save them
        results_test=regression_results(y_true,y_pred)
        test_score = results_test['r2']
        results_test['test_size']=round(test_s[i],4)
        results_test['train_size']=round(train_size,4)
        test_dic = pd.DataFrame(results_test, index=[0])
        output = output.append(test_dic, ignore_index=True)
        dates = test_split['Time']
        #Plot the graphs
        plot_accuracy(y_true, y_pred, int(y_true.size/50), 100, test_s[i], dates, str(j))
        
print(output)
output.to_csv('configuration_models/model_score_test'+str(test_s[i])+'_train'+str(train_size)+'.csv', mode='a', header=True)  


