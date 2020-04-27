import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.utils import check_random_state
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

def plot_accuracy(y_true, y_pred, xax_, t_s, train_size, score, filename):

    xax=xax_
    ig=plt.figure(figsize=(25, 12), dpi= 80, facecolor='w', edgecolor='k')
    plt.plot(y_true[0::xax], label='y_true')
    plt.plot(y_pred[0::xax], label='y_pred')
    min_y_bound = min(min(y_true), min(y_true))
    max_y_bound = max(max(y_true), max(y_true))
    step = abs(max_y_bound-min_y_bound)/50

    plt.yticks(np.arange(min_y_bound, max_y_bound+step, step))
    plt.xticks(np.arange(0, y_true[0::xax].size+1,1), np.arange(0, y_true.size, 1)[0::xax], rotation=70)

 
    plt.xlabel('Instances')
    plt.ylabel('CO2')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("graphs/bagging/"+filename+"_score{score:.3f}_test{test_s:.4f}_train{train:.3f}.png".format(test_s=t_s, score = score, train=train_size))
    fname = "predictions/bagging/"+filename+"_score{score:.3f}_test{test_s:.4f}_train{train:.3f}.csv".format(test_s=t_s, score = score, train=train_size)
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

def get_percentage_list(total_size):
    one_hour = get_percentage(60, total_size)
    three_hour = get_percentage(180, total_size)
    six_hour = get_percentage(360, total_size)
    one_day = get_percentage(1440, total_size)
    #three_day = get_percentage(3*1440, total_size)
    return [one_hour, three_hour, six_hour, one_day] #, three_day]

def get_percentage(x, total):
    return ((x*100)/total)/100

def main():

    #Read file
    filename = 'merged_useless'
    dataset = pd.read_csv('merged_data/'+filename+'.csv', delimiter=";", parse_dates=True)
    dataset = dataset.drop(['id'], axis=1)

    dataset = remove_outliers(dataset)

    #Get min and max
    min_CO2 = dataset['CO2'].min()
    max_CO2 = dataset['CO2'].max()
    #Filter dataset data
    dataset = filter_data(dataset, 21, 1)

    #Normalize dataset
    dataset = normalize_data(dataset)

    #Split between X and Y
    X = dataset.drop(['CO2', 'Time'], axis=1)
    y = dataset['CO2'].to_numpy()

    #Set train and test size
    train_size = 0.7
    test_size = 0.3

    #Scale data
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    output = []
    tests = get_percentage_list(dataset.shape[0])
    regressor = []
    for i in range(len(tests)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=tests[i], random_state=55)

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

        y_true, y_pred = y_test, regressor.predict(X_test)
        results = regression_results(y_true, y_pred)
        output.append(results)

        y_true = inverse_normalize(y_true, min_CO2, max_CO2)
        y_pred = inverse_normalize(y_pred, min_CO2, max_CO2)

        plot_accuracy(y_true, y_pred, int(y_pred.size/50), test_size, train_size, results['r2'], 'merged')

    print(pd.DataFrame.from_dict(output))

if __name__ == "__main__":
    main()
