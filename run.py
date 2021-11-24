

import sys
import json
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

sys.path.insert(0, 'src')
from etl import featurize, clean_df, clean_label_data, generate_labels #generate_data, save_data
from eda import plot_timeseries, plot_histogram
from utils import convert_notebook
from os import listdir, remove, makedirs
from os.path import isfile, join, expanduser
from time import time

raw_data_path = "data/raw"
temp_path = "data/temp"
out_path = "data/out"
img_path = "notebooks/figures"

test_data_path = "test/testdata"
test_temp_path = "test/temp"
test_out_path = "test/out"

def data_(test=False):
    '''data target logic. Generates temporary files that are cleaned.'''
    data_path = test_data_path if test else raw_data_path
    temp_ = test_temp_path if test else temp_path
    
    data_csv_files = [join(data_path, f) for f in listdir(data_path)]
    dataframes = [clean_label_data(file) for file in data_csv_files]
    
    # temp_path = "data/temp"
    for i in range(len(data_csv_files)):
        dataframes[i].to_csv(join(temp_, listdir(data_path)[i]))
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

def features_(test=False):

    # temp_path = "data/temp"
    temp_ = test_temp_path if test else temp_path
    out_ = test_out_path if test else out_path

    df = generate_labels(folderpath=temp_, features=True)

    tm = int(time())
    df.to_csv(join(out_,f'features_{tm}.csv'))

def train_(latency_=True, pca_=True):
    '''train target logic. Generates model (random forest) and produces output'''
    out_ = test_out_path if test else out_path

    featurelst = listdir(out_)
    featurelst.sort()
    df = pd.read_csv(join(out_,featurelst[-1])) # gets latest feature file from data/out
    
    df = df[df['label_latency'] <= 500]

    cols = [col for col in df.columns if not 'label' in col]

    X = features_label[cols].fillna(0) # removing nulls for model to train, TODO change to impute?
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

def test_():
    '''test target logic. Involves simulating entire ML process on sample test data.'''
    data_(test=True)
    # eda_() #hard coded values, does not run
    features_(test=True)
    clean_()
    # train_() # unfinished target
    return

def clean_(): # TODO revisit which directories should be scrubbed
    '''clean target logic. removes all temporary/output files generated in directory.'''
    for dr_ in [test_temp_path, test_out_path]: #, out_path, img_path]:
        for f in listdir(dr_):
            remove(join(dr_, f))
        
    return

def main(targets):

    data_config = json.load(open('config/data-params.json'))
    eda_config = json.load(open('config/eda-params.json'))

    for dr_ in [temp_path, out_path, test_temp_path, test_out_path]:
        makedirs(dr_, exist_ok=True)

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
