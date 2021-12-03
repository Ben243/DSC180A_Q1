
from time import time
import pandas as pd
import numpy as np
import json
import pickle
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from os import listdir
from os.path import join

## defining parameters
model_params = json.load(open('config/model-params.json')) 

loss_model = model_params['loss_model']
latency_model = model_params['latency_model']
PCA_COMPONENTS = model_params['PCA_COMPONENTS']
n_jobs = model_params['n_jobs']
n_estimators = model_params['n_estimators']
max_depth = model_params['max_depth'] # @justin let me know if you think this necessary


# def train_model(data_path, pickle_path, filename): #TODO make a decision on 'filename' parameter
def train_model(data_path, pickle_path):

    ## load feature output
    # data_csv_files = [join(data_path, f) for f in listdir(data_path)]
    # data = pd.concat([pd.read_csv(csv) for csv in data_csv_files]).drop(columns="group")
    featurelst = listdir(data_path)
    featurelst.sort()
    data = pd.read_csv(join(data_path,featurelst[-1])) ## gets latest feature file from feature path

    ## feature selection
    loss_cols = [
        "mean_tdelta_min", 'mean_tdelta_max', 'mean_tdelta_mean', 'max_tdelta_var', 
        '1->2Bytes_var', '2->1Bytes_var', '1->2Pkts_var', '2->1Bytes_var', 
        '1->2Pkts_rolling_2s_mean_var', '2->1Pkts_rolling_2s_mean_var', 
        '1->2Pkts_rolling_3s_mean_var', '2->1Pkts_rolling_3s_mean_var']
    # latency_cols = list(set(data.columns) - set(loss_cols)) + ['pred_loss']
    latency_cols = [
        "1->2Bytes_min", "1->2Bytes_max","1->2Bytes_mean","1->2Bytes_median",
        "1->2Bytes_std","1->2Bytes_var","2->1Bytes_min","2->1Bytes_max","2->1Bytes_mean",
        "2->1Bytes_median","2->1Bytes_std","2->1Bytes_var","1->2Pkts_min","1->2Pkts_max",
        "1->2Pkts_mean","1->2Pkts_median","1->2Pkts_std","1->2Pkts_var","1->2Pkts_sum",
        "2->1Pkts_min","2->1Pkts_max","2->1Pkts_mean","2->1Pkts_median","2->1Pkts_std",
        "2->1Pkts_var","2->1Pkts_sum","mean_tdelta_min","mean_tdelta_max","mean_tdelta_mean",
        "mean_tdelta_var","mean_tdelta_std","max_tdelta_max","max_tdelta_mean",
        "max_tdelta_var","max_tdelta_std","1->2Pkts_rolling_2s_mean_min",
        "1->2Pkts_rolling_2s_mean_max","1->2Pkts_rolling_2s_mean_var",
        "1->2Pkts_rolling_2s_mean_std","1->2Pkts_rolling_2s_mean_sum",
        "2->1Pkts_rolling_2s_mean_min","2->1Pkts_rolling_2s_mean_max",
        "2->1Pkts_rolling_2s_mean_var","2->1Pkts_rolling_2s_mean_std",
        "2->1Pkts_rolling_2s_mean_sum","1->2Pkts_rolling_3s_mean_min",
        "1->2Pkts_rolling_3s_mean_max","1->2Pkts_rolling_3s_mean_var",
        "1->2Pkts_rolling_3s_mean_std","1->2Pkts_rolling_3s_mean_sum",
        "2->1Pkts_rolling_3s_mean_min","2->1Pkts_rolling_3s_mean_max",
        "2->1Pkts_rolling_3s_mean_var","2->1Pkts_rolling_3s_mean_std",
        "2->1Pkts_rolling_3s_mean_sum","2->1_interpacket_mean","1->2_interpacket_mean",
        "pred_loss"
    ]
    ## packet loss model training
    loss_X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(data[loss_cols]) # imputation with mean strategy
    # loss_X_norm = (X_pca - X_pca.mean()) / X_pca.std()
    loss_y = np.log(1/data['label_packet_loss']) # log inverted loss
    
    # loss_X_train, loss_X_test, loss_y_train, loss_y_test = train_test_split(
    #     loss_X, loss_y, test_size=0.25, random_state=42)

    loss_forest = RandomForestRegressor(n_jobs=n_jobs, n_estimators=n_estimators, max_depth=max_depth)
    loss_forest.fit(loss_X, loss_y)

    data['pred_loss'] = np.exp(1/loss_forest.predict(loss_X)) # adding prediction loss as a feature for latency
    
    ## latency model training
    latency_X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(data[latency_cols]) # imputation with mean strat
    # latency_X = PCA(n_components=PCA_COMPONENTS).fit_transform(latency_X)
    latency_y = data['label_packet_loss']

    # latency_X_train, latency_X_test, latency_y_train, latency_y_test = train_test_split(
    #     latency_X_pca, latency_y, test_size=0.25, random_state=42)

    latency_forest = RandomForestRegressor(n_jobs=n_jobs, n_estimators=n_estimators, max_depth=max_depth)

    latency_forest.fit(latency_X, latency_y)

    #TODO figure out where to use models or output model predictions. probably a separate notebook.

    ## model saving
    with open(join(pickle_path, loss_model),"wb") as f:
        pickle.dump(loss_forest, f)

    with open(join(pickle_path, latency_model),"wb") as f:
        pickle.dump(latency_forest, f)
