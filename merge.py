import os.path
import os, glob
import csv
from os import listdir
from os.path import isfile, join
import pandas as pd

data_path = 'data/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
data = []
for file in onlyfiles:

    #Read file and change date type
    filename = os.path.splitext(file)[0]
    dataset = pd.read_csv('data/'+filename+'.csv', delimiter=";", parse_dates=True)
    data.append(dataset)

data = pd.concat(data)

data.to_csv('merged_Data/merged_no_kitchen_bathroom.csv', sep=';')
