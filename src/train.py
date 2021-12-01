
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from os import listdir
from os.path import isfile, join, expanduser

PCA_COMPONENTS = 10

def train_model(data_path, pickle_path, filename):

    data_csv_files = [join(data_path, f) for f in listdir(data_path)]
    data = pd.concat([pd.read_csv(csv) for csv in data_csv_files]).drop(columns="group")
    
    cols = [x for x in data.columns if not 'label' in x]
    X = data[cols].fillna(0)
    pca = PCA(n_components=PCA_COMPONENTS)
    X_pca = pca.fit_transform(X)
    
    y_log = pd.DataFrame({"latency": list(np.log(data['label_latency'])), "loss": list(np.log(data['label_packet_loss']))})
    
    forest = RandomForestRegressor(n_jobs=-1, n_estimators=500, max_depth=15)
    
    X_norm = (X_pca - X_pca.mean()) / X_pca.std()

    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_log, test_size=0.25)
    
    forest.fit(X_train, y_train)
    
    with open(join(pickle_path, filename),"wb") as f:
        pickle.dump(forest, f)
