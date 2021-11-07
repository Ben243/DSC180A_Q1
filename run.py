

import sys
import json
import pandas as pd

sys.path.insert(0, 'src')
from etl import featurize, clean_df, clean_label_data, generate_labels #generate_data, save_data
from eda import plot_timeseries, plot_histogram
from utils import convert_notebook
from os import listdir
from os.path import isfile, join, expanduser
from time import time

raw_data_path = "data/raw"
temp_path = "data/temp"
out_path = "data/out"
img_path = "notebooks/figures"

def data_():
    '''data target logic. Generates temporary files that are cleaned.'''

    data_csv_files = [join(raw_data_path, f) for f in listdir(raw_data_path)]
    dataframes = [clean_label_data(file) for file in data_csv_files]
    
    # temp_path = "data/temp"
    for i in range(len(data_csv_files)):
        dataframes[i].to_csv(join(temp_path, listdir(raw_data_path)[i]))
    return

def eda_():
    '''eda target logic. Generates all relevant visualizations used.'''

    # temp_path = "data/temp"
    # out_path = "notebooks/figures"
    csv1 = [join(temp_path, "s2_200-100-iperf.csv"), join(temp_path, "s2_200-500-iperf.csv")]
    csv2 = [join(temp_path, "s2_200-10000-iperf.csv"), join(temp_path, "s2_200-50000-iperf.csv")]
    
    plot_histogram(csv1+csv2, img_path, filename="100-500-10000-50000_hist.png", function="time delta")
    
    plot_timeseries(csv1, img_path, filename="100-500_amp.png", function="amplitude")
    plot_timeseries(csv2, img_path, filename="10000-50000_amp.png", function="amplitude")
    
    plot_timeseries(csv1, img_path, filename="100-500_mean.png", function="mean")
    plot_timeseries(csv2, img_path, filename="10000-50000_mean.png", function="mean")
    return

def features_():

    # temp_path = "data/temp"
    
    df = generate_labels(folderpath=temp_path, features=True)
    
    tm = int(time())
    df.to_csv(f'data/out/features_{tm}.csv')

def test_():
    '''test target logic. Involves simulating entire ML process on sample test data.'''
    return

def clean_():
    '''clean target logic. removes all temporary/output files generated in directory.'''
    for f in os.listdir(temp_path):
        os.remove(os.path.join(temp_path, f))
        
    for f in os.listdir(out_path):
        os.remove(os.path.join(out_path, f))

    for f in os.listdir(img_path):
        os.remove(os.path.join(img_path, f))
    return


def main(targets):

    data_config = json.load(open('config/data-params.json'))
    eda_config = json.load(open('config/eda-params.json'))

    # goes under the assumption that all the datafiles have their latency and loss in the name
    # cleans and adds features for all of the csv in the raw data folder
    if 'data' in targets:
        """
        data = generate_data(**data_config)
        save_data(data, **data_config)
        """
        data_()
        
    if 'eda' in targets: 
        eda_()

    if 'features' in targets:
        features_()
    
    if 'test' in targets:
        test_()
        
    if 'clean' in targets:
        clean_()

    if 'all' in targets:
        data_()
        eda()
        features_()
        # clean()

if __name__ == '__main__':

    targets = sys.argv[1:]
    main(targets)
