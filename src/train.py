
from time import time
import pandas as pd
import numpy as np
import json
import pickle
import sys

from sklearn.compose import ColumnTransformer
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
PCA_COMPONENTS = model_params['PCA_COMPONENTS']
n_jobs = model_params['n_jobs']
n_estimators = model_params['n_estimators']
max_depth = model_params['max_depth']
test_size = model_params['test_size']
randomstate = 2021


# def train_model(data_path, pickle_path, filename): #TODO make a decision on 'filename' parameter
def train_model(data_path, pickle_path, out_path='data/out', model_name=None, test=False):

    ## load feature output
    featurelst = listdir(data_path)
    featurelst = [file for file in featurelst if 'feature' in file]
    featurelst.sort()
    data = pd.read_csv(join(data_path,featurelst[-1])) ## gets latest feature file from feature path

    tm = int(time())
    
    if not test:
        train_data, validation_data = train_test_split(data, test_size=test_size, random_state=randomstate)
        train_data.to_csv(join(out_path,f'train_{tm}.csv'), index=False)
        validation_data.to_csv(join(out_path,f'validation_{tm}.csv'), index=False)
    else:
        tm = 'test'
        data.to_csv('test/test_features/test_featureset.csv', index=False)
        train_data = data
    
    ## feature selection
    loss_cols = [
        "mean_tdelta_min", 'mean_tdelta_max', 'mean_tdelta_mean', 'max_tdelta_var', 
        '1->2Bytes_var', '2->1Bytes_var', '1->2Pkts_var', '2->1Bytes_var', 
        '1->2Pkts_rolling_2s_mean_var', '2->1Pkts_rolling_2s_mean_var', 
        '1->2Pkts_rolling_3s_mean_var', '2->1Pkts_rolling_3s_mean_var']
    latency_cols = list(set(data.columns) - set(loss_cols))# + ['pred_loss']

    loss_X = train_data
    ## packet loss model training
    #loss_X = train_data[loss_cols]    
    loss_y = np.log(train_data['label_packet_loss']) # log loss

    
    loss_pipe = Pipeline(steps=[
        ("selector", ColumnTransformer([("selector", "passthrough", loss_cols)], remainder="drop")),
        ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('reduce_dim', PCA(PCA_COMPONENTS)), 
        ('clf', RandomForestRegressor(n_jobs=n_jobs, n_estimators=n_estimators, max_depth=max_depth))])
    loss_pipe.fit(loss_X, loss_y)

    # data['pred_loss'] = loss_forest.predict(loss_X) # adding prediction loss as a feature for latency
    
    latency_X = train_data
    ## latency model training
    #latency_X = train_data[latency_cols]
    latency_y = np.log(train_data['label_latency'])

    latency_pipe = Pipeline(steps=[
        ("selector", ColumnTransformer([("selector", "passthrough", latency_cols)], remainder="drop")),
        ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('reduce_dim', PCA(PCA_COMPONENTS)), 
        ('clf', RandomForestRegressor(n_jobs=n_jobs, n_estimators=n_estimators, max_depth=max_depth))])

    latency_pipe.fit(latency_X, latency_y)

    ## model saving

    if test == False:
        with open(join(pickle_path, f'loss_{tm}.pyc'),"wb") as f:
            pickle.dump(loss_pipe, f)

        # with open(join(pickle_path, latency_model),"wb") as f:
        #     pickle.dump(latency_forest, f)

        with open(join(pickle_path, f'latency_{tm}.pyc'),"wb") as f:
            pickle.dump(latency_pipe, f)
    else:
        with open(join(pickle_path, 'loss_test_model.pyc'),"wb") as f:
            pickle.dump(loss_pipe, f)

        # with open(join(pickle_path, latency_model),"wb") as f:
        #     pickle.dump(latency_forest, f)

        with open(join(pickle_path, 'latency_test_model.pyc'),"wb") as f:
            pickle.dump(latency_pipe, f)
