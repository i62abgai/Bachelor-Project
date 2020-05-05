import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.utils import check_random_state
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

def plot_accuracy(y_true, y_pred, xax_, t_s, train_size, score, filename, folder, dates):
    """Function that plots the predicted and true values on the y ax, and the date in the x ax.
    Also saves the true and predicted in a csv file.

    Arguments:
        y_true {list} -- list with true values of CO2
        y_pred {list} -- list with predicted values of CO2
        xax_ {int} -- step of the y ax
        t_s {float} -- test size
        train_size {float} -- train size
        score {float} -- R2 score obtained
        filename {string} -- name of the file which has been tested
        folder {string} -- name of the folder where the plot is going to be saved
        dates {list} -- list of date to print in the x ax
    """
    xax=xax_
    fig=plt.figure(figsize=(12, 12), dpi= 80, facecolor='w', edgecolor='k')
    plt.plot(y_true[0::xax], label='y_true')
    plt.plot(y_pred[0::xax], label='y_pred')
    min_y_bound = min(min(y_true), min(y_true))
    max_y_bound = max(max(y_true), max(y_true))
    step = abs(max_y_bound-min_y_bound)/20

    plt.yticks(np.arange(min_y_bound, max_y_bound+step, step), fontsize=15)
    plt.xticks(np.arange(0, y_true[0::xax].size+1,1), dates[0::xax], rotation=70, fontsize=15)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('CO2 (ppm)', fontsize=16)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    plt.savefig("graphs/"+folder+"/"+filename+"_score{score:.3f}_test{test_s:d}_train{train:d}.png".format(test_s=t_s, score = score, train=train_size))
    fname = "predictions/"+folder+"/"+filename+"_score{score:.3f}_test{test_s:d}_train{train:d}.csv".format(test_s=t_s, score = score, train=train_size)
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    df = pd.DataFrame({"y_true" : y_true, "y_pred" : y_pred})
    df.to_csv(fname, index=False, sep=";")

def regression_results(y_true, y_pred):
    """Function that calculates the different scores used to compare the models.

    Arguments:
        y_true {list} -- list with true values of CO2
        y_pred {list} -- list with predicted values of CO2

    Returns:
        dictionary -- dictionary with all the calculated scores
    """    
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    r2=metrics.r2_score(y_true, y_pred)
    metrics_dic = {'r2':round(r2,4), 'mae':round(mean_absolute_error,4), 'mse':round(mse,4), 'rmse':round(np.sqrt(mse),4)}
    return metrics_dic

def filter_data(data, wl, po):
    """Function that applies Savitzky–Golay filter to the data 

    Arguments:
        data {dataframe} -- Dataframe with all the data
        wl {int} -- Windows size to apply the filter
        po {int} -- Polynomial order to apply the filter

    Returns:
        dataframe -- Dataframe with all the data filtered
    """    
    dataset_filtered = data.copy()
    dataset_filtered['CO2']= savgol_filter(dataset_filtered['CO2'], wl, po)
    dataset_filtered['H']= savgol_filter(dataset_filtered['H'], wl, po)
    dataset_filtered['T']= savgol_filter(dataset_filtered['T'], wl, po)
    return dataset_filtered

def remove_outliers(dataset, threshold):
    """Function that detect the outliers from the dataset and removes them

    Arguments:
        dataset {dataframe} -- Dataframe with all the data
        threshold {float} -- Threshold to consider a data point as outlier

    Returns:
        dataframe -- dataframe without outliers
    """    
    z = np.abs(stats.zscore(dataset.iloc[:,1:4]))
    return dataset[(z < threshold).all(axis=1)]  