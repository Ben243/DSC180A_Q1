
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
from sklearn.pipeline import Pipeline
from os import listdir
from os.path import join

## defining parameters
model_params = json.load(open('config/model-params.json')) 

loss_model = model_params['loss_model']
latency_model = model_params['latency_model']
test_loss_model = model_params['test_loss_model']
test_latency_model = model_params['test_latency_model']
PCA_COMPONENTS = model_params['PCA_COMPONENTS']
n_jobs = model_params['n_jobs']
loss_n_estimators = model_params['loss_n_estimators']
latency_n_estimators = model_params['latency_n_estimators']
loss_max_depth = model_params['loss_max_depth']
latency_max_depth = model_params['latency_max_depth']
test_size = model_params['test_size']
randomstate = 42

def train_model(data_path, pickle_path, out_path='data/out', test=False):

    ## load feature output
    featurelst = listdir(data_path)
    featurelst.sort()
    data = pd.read_csv(join(data_path,featurelst[-1])).drop(columns='group') ## gets latest feature file from feature path

    if not test:
        train_data, validation_data = train_test_split(data, test_size=test_size, random_state=randomstate)
        train_data.to_csv(join(out_path,'0_train_out.csv'))
        validation_data.to_csv(join(out_path,'0_validation_out.csv'))
    else:
        data.to_csv('test/test_features/test_featureset.csv')
        train_data = data

    ## feature selection
    loss_cols = [
        "mean_tdelta_min", 'mean_tdelta_max', 'mean_tdelta_mean', 'max_tdelta_var', 
        '1->2Bytes_var', '2->1Bytes_var', '1->2Pkts_var', '2->1Bytes_var', 
        '1->2Pkts_rolling_2s_mean_var', '2->1Pkts_rolling_2s_mean_var', 
        '1->2Pkts_rolling_3s_mean_var', '2->1Pkts_rolling_3s_mean_var']
    latency_cols = [
        '1->2Bytes_max', '1->2Bytes_mean', '1->2Bytes_median', '1->2Bytes_min',
        '1->2Pkts_max', '1->2Pkts_mean', '1->2Pkts_median', '1->2Pkts_min',
        '1->2Pkts_rolling_2s_mean_max', '1->2Pkts_rolling_2s_mean_min',
        '1->2Pkts_rolling_3s_mean_max', '1->2Pkts_rolling_3s_mean_min',
        '2->1Bytes_max', '2->1Bytes_mean', '2->1Bytes_median', '2->1Bytes_min',
        '2->1Pkts_max', '2->1Pkts_mean', '2->1Pkts_median', '2->1Pkts_min',
        '2->1Pkts_rolling_2s_mean_max', '2->1Pkts_rolling_2s_mean_min',
        '2->1Pkts_rolling_3s_mean_max', '2->1Pkts_rolling_3s_mean_min',
        '2->1Pkts_var', 'label_latency', 'label_packet_loss', 'max_tdelta_max',
        'max_tdelta_mean', 'mean_tdelta_var', 
        'pred_loss'
    ]

    ## packet loss model training
    loss_X = train_data[loss_cols]    
    loss_y = np.log(train_data['label_packet_loss']) # log loss

    
    loss_pipe = Pipeline(steps=[
        ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('clf', RandomForestRegressor(n_jobs=n_jobs, n_estimators=loss_n_estimators, max_depth=loss_max_depth))])
    loss_pipe.fit(loss_X, loss_y)

    train_data['pred_loss'] = loss_pipe.predict(loss_X) # adding prediction loss as a feature for latency
    
    ## latency model training
    latency_X = train_data[latency_cols]
    latency_y = np.log(train_data['label_latency'])

    latency_pipe = Pipeline(steps=[
        ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('reduce_dim', PCA(PCA_COMPONENTS)), 
        ('clf', RandomForestRegressor(n_jobs=n_jobs, n_estimators=latency_n_estimators, max_depth=latency_max_depth))])

    latency_pipe.fit(latency_X, latency_y)

    ## model saving
    loss_model_path = test_loss_model if test else loss_model
    latency_model_path = test_latency_model if test else latency_model

    with open(join(pickle_path, loss_model_path),"wb") as f:
        pickle.dump(loss_pipe, f)

    with open(join(pickle_path, latency_model_path),"wb") as f:
        pickle.dump(latency_pipe, f)
