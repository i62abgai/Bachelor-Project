import pandas as pd
import numpy as np
import time
import math
import click
import os

# file libraries
from os import listdir
from os.path import isfile, join

# visual libraries
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D 
plt.style.use('ggplot')

# sklearn libraries
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler 

def readFile(path, file):
    
    if not os.path.isdir(path+"/pre"):
        os.mkdir(path+"/pre/")       	
        
    print("\nPreprocessing file: ", file, "\n")
    fullpath = path + "/" + file
    # Read the data in the CSV file using pandas
    df = pd.read_csv(fullpath, delimiter=";")
    print(df.head(), "\n")
    print("\tShape: ", df.shape)
    print("\tNull values: ", df.isnull().any().sum())
    df = df.drop(["Time"], axis=1)
    
    print("\n\tRescale data:\n")
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    rescaleddf = scaler.fit_transform(df)
    print(rescaleddf)
    
    print("\n\tNormalize data:\n")
    vdf = normalize(rescaleddf, norm='l2', axis=1, copy=True, return_norm=False)
    print(vdf)
    # Get the maximum values of each column i.e. along axis 0
    maxInColumns = np.amax(vdf, axis=0)
    
    print('\n\tMax value of every column: ', maxInColumns)
    
    np.savetxt(path+"/pre/"+file, vdf, delimiter=";")
    
    input()
    
@click.command()
@click.option('--folder', "-f", default=None, required=True,
              help=u'File to preprocess.')
def main(folder):    
    # Get all files in the folder
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    # Open and preprocess them
    for name in onlyfiles:
        readFile(folder,name)
      
if __name__ == "__main__":
    main()