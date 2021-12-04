
import os
import sys
import json
import pandas as pd

sys.path.insert(0, 'src')
from etl import featurize, clean_df, clean_label_data, generate_labels #generate_data, save_data
from eda import plot_timeseries, plot_histogram
from train import train_model
from utils import convert_notebook
from os import listdir, remove
from os.path import isfile, join, expanduser
from time import time

#TODO maybe put all these filepaths in a relevant params json
raw_data_path = "data/raw"
temp_path = "data/temp"
out_path = "data/out"

test_path = "test/testdata"
img_path = "notebooks/figures"
figure_data_path = "test/figure_data"
model_path = "models"

def etl_(raw_data_path=raw_data_path, temp_path=temp_path):
    '''etl target logic. Generates temporary files that are cleaned.'''
    ## dump featurized data into temp folder
    data_csv_files = [join(raw_data_path, f) for f in listdir(raw_data_path)]
    dataframes = [clean_label_data(file_, True) for file_ in data_csv_files]
    
    tm = int(time())

    for i in range(len(data_csv_files)):
        if (i == 0):
            # dataframes[i].to_csv(join(temp_path, listdir(raw_data_path)[i]))    
            dataframes[i].to_csv(join(out_path,f'features_{tm}.csv'))
        else:
            # dataframes[i].to_csv(join(temp_path, listdir(raw_data_path)[i]), header=False)
            dataframes[i].to_csv(join(out_path,f'features_{tm}.csv'), header=False, mode='a')

    # features_label = pd.concat([pd.read_csv(csv) for csv in data_csv_files]).drop(columns="group")
    # features_label = pd.concat(map(pd.read_csv, data_csv_files))#.drop(columns="group")
    # 
    # features_label.to_csv(join(out_path,f'features_{tm}.csv'))

def eda_(temp_path=figure_data_path, img_path=img_path):
    '''Generates all relevant visualizations used in early data analysis.'''

    csv1 = [join(temp_path, "s2_200-100-iperf.csv"), join(temp_path, "s2_200-500-iperf.csv")]
    csv2 = [join(temp_path, "s2_200-10000-iperf.csv"), join(temp_path, "s2_200-50000-iperf.csv")]
    
    plot_histogram(csv1+csv2, img_path, filename="100-500-10000-50000_hist.png", function="time delta")
    
    plot_timeseries(csv1, img_path, filename="100-500_amp.png", function="amplitude")
    plot_timeseries(csv2, img_path, filename="10000-50000_amp.png", function="amplitude")
    
    plot_timeseries(csv1, img_path, filename="100-500_mean.png", function="mean")
    plot_timeseries(csv2, img_path, filename="10000-50000_mean.png", function="mean")
    return

def train_(test=False):
    '''trains a model to predict latency and packet loss with the output of etl and features.'''    
    train_model(out_path, model_path, test=test)

def test_():
    '''test target logic. Involves simulating entire ML process on sample test data.'''
    clean_()
    # eda_()
    etl_(raw_data_path=test_path)
    train_(test=True)

def clean_(): # TODO revisit which directories should be scrubbed
    '''clean target logic. removes all temporary/output files generated in directory.'''
    # for dr_ in [temp_path, out_path, model_path, img_path]:
    for dr_ in [temp_path, out_path]:
        for f in listdir(dr_):
            remove(join(dr_, f))
        
    return

def main(targets):

    data_config = json.load(open('config/data-params.json'))
    eda_config = json.load(open('config/eda-params.json'))

    # goes under the assumption that all the datafiles have their latency and loss in the name
    # cleans and adds features for all of the csv in the raw data folder
    if 'etl' in targets or 'data' in targets:
        """
        data = generate_data(**data_config)
        save_data(data, **data_config)
        """
        etl_()
        
    if 'eda' in targets: 
        eda_()

    if 'train' in targets:
        train_()

    if 'test' in targets:
        test_()
        
    if 'clean' in targets:
        clean_()

    if 'all' in targets:
        etl_()
        eda()
        features_()
        # train_()
        # clean_()
    else:
        return

if __name__ == '__main__':

    targets = sys.argv[1:]
    main(targets)
