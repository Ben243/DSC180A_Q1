

import sys
import json
import pandas as pd
from sklearn.impute import SimpleImputer

sys.path.insert(0, 'src')
from etl import featurize, clean_df, clean_label_data, generate_labels #generate_data, save_data
from eda import plot_timeseries, plot_histogram
from utils import convert_notebook
from os import listdir, remove
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
    df.to_csv(join(out_path,f'features_{tm}.csv'))

def train_(latency_=True, pca_=True):
    '''train target logic. Generates model (random forest) and produces output'''
    featurelst = listdir(out_path)
    featurelst.sort()
    df = pd.read_csv(join(out_path,featurelst[-1])) # gets latest feature file from data/out
    
    df = df[df['label_latency'] <= 500]

    cols = [col for col in df.columns if not 'label' in col]

    X = features_label[cols].fillna(0) # removing nulls for model to train
    latency_y = df['label_latency']
    packet_y = df['label_packet_loss']
    
    ## splitting training data ##
    if pca_: 
        pca = PCA(n_components=21)
        X_pca = pca.fit_transform(X) 

        X_train, X_test, latency_y_train, latency_y_test, packet_y_train, packet_y_test = train_test_split(
            X_pca, latency_y, packet_y, train_size=0.75, shuffle=True, random_state=42)
    else: 
        X_train, X_test, latency_y_train, latency_y_test, packet_y_train, packet_y_test = train_test_split(
            X, latency_y, packet_y, train_size=0.75, shuffle=True, random_state=42)
    
    ## Predicting Latency ##
    latency_rf = RandomForestRegressor(n_jobs=-1)
    latency_rf.fit(X_train, latency_y_train)
    r2_latency = latency_rf.score(X_test, latency_y_test)
    df['label_latency_pred'] = latency_rf.predict(X)

    ## Predicting Packet Loss ##
    packet_rf = RandomForestRegressor(n_jobs=-1)
    packet_rf.fit(X_train, packet_y_train)
    r2_packet = packet_rf.score(X_test, packet_y_test)
    df['label_packet_loss_pred'] = packet_rf.predict(X)

    ## Model Output and metrics
    print(f'R2 Score - Latency: {r2_latency}, Packet Loss: {r2_packet}') # feel free to add more metrics
    df.to_csv(join(out_path,f'out_{featurelst[-1]}'))

def test_(): # TODO revisit what counts as simulated data
    '''test target logic. Involves simulating entire ML process on sample test data.'''
    data_()
    features_()
    train_()
    return

def clean_(): # TODO revisit which directories should be scrubbed
    '''clean target logic. removes all temporary/output files generated in directory.'''
    for f in listdir(temp_path):
        remove(join(temp_path, f))
        
    # for f in listdir(out_path):
    #     remove(join(temp_path, f))

    # for f in listdir(img_path):
    #     remove(join(temp_path, f))
        
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
    
    if 'train' in targets:
        train_()

    if 'test' in targets:
        test_()
        
    if 'clean' in targets:
        clean_()

    if 'all' in targets:
        data_()
        eda_()
        features_()
        train_()
        # clean()

if __name__ == '__main__':

    targets = sys.argv[1:]
    main(targets)
