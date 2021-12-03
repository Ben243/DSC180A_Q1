
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
PCA_COMPONENTS = model_params['PCA_COMPONENTS']
n_jobs = model_params['n_jobs']
n_estimators = model_params['n_estimators']
max_depth = model_params['max_depth'] # @justin let me know if you think this necessary


# def train_model(data_path, pickle_path, filename): #TODO make a decision on 'filename' parameter
def train_model(data_path, pickle_path):

    ## load feature output
    featurelst = listdir(data_path)
    featurelst.sort()
    data = pd.read_csv(join(data_path,featurelst[-1])) ## gets latest feature file from feature path

    ## feature selection
    loss_cols = [
        "mean_tdelta_min", 'mean_tdelta_max', 'mean_tdelta_mean', 'max_tdelta_var', 
        '1->2Bytes_var', '2->1Bytes_var', '1->2Pkts_var', '2->1Bytes_var', 
        '1->2Pkts_rolling_2s_mean_var', '2->1Pkts_rolling_2s_mean_var', 
        '1->2Pkts_rolling_3s_mean_var', '2->1Pkts_rolling_3s_mean_var']
    latency_cols = list(set(data.columns) - set(loss_cols))# + ['pred_loss']

    ## packet loss model training
    # loss_X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(data[loss_cols]) # imputation with mean strategy
    loss_X = data[loss_cols]    
    loss_y = np.log(data['label_packet_loss']) # log loss

    loss_pipe = Pipeline(steps=[
        ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('clf', RandomForestRegressor(n_jobs=n_jobs, n_estimators=n_estimators, max_depth=max_depth))])
    loss_pipe.fit(loss_X, loss_y)
    
    # loss_forest = RandomForestRegressor(n_jobs=n_jobs, n_estimators=n_estimators, max_depth=max_depth)
    # loss_forest.fit(loss_X, loss_y)

    # data['pred_loss'] = loss_forest.predict(loss_X) # adding prediction loss as a feature for latency
    
    ## latency model training
    # latency_X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(data[latency_cols]) # imputation with mean strat
    # pca_ = PCA(n_components=PCA_COMPONENTS)
    # latency_X = pca_.fit_transform(latency_X)
    latency_X = data[latency_cols]
    latency_y = np.log(data['label_latency'])

    latency_pipe = Pipeline(steps=[
        ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('reduce_dim', PCA(PCA_COMPONENTS)), 
        ('clf', RandomForestRegressor(n_jobs=n_jobs, n_estimators=n_estimators, max_depth=max_depth))])

    latency_pipe.fit(latency_X, data['label_latency'])
    # latency_X_train, latency_X_test, latency_y_train, latency_y_test = train_test_split(
    #     latency_X_pca, latency_y, test_size=0.25, random_state=42)

    # latency_forest = RandomForestRegressor(n_jobs=n_jobs, n_estimators=n_estimators, max_depth=max_depth)

    # latency_forest.fit(latency_X, latency_y)


    ## model saving

    with open(join(pickle_path, loss_model),"wb") as f:
        pickle.dump(loss_pipe, f)

    # with open(join(pickle_path, latency_model),"wb") as f:
    #     pickle.dump(latency_forest, f)

    with open(join(pickle_path, latency_model),"wb") as f:
        pickle.dump(latency_pipe, f)
