

import os
import numpy as np
import pandas as pd

GROUP_INTERVAL = 10

# def generate_data(size=(1000, 3), **kwargs):

#     n, k = size
#     means = np.random.uniform(-3, 3, size=k)
#     stds = np.random.uniform(size=k)
    
#     data = np.random.normal(means, stds, size)
#     data = pd.DataFrame(data, columns=['x_%d' % i for i in range(k)])    

#     return data


# def save_data(data, data_fp, **kwargs):

#     os.makedirs(os.path.split(data_fp)[0], exist_ok=True)

#     data.to_csv(data_fp, index=False)

#     return 


'''
Extraction/Cleaning
'''
def clean_df(df):
    '''
    removes unnecessary connections from data and
    removes initial peak from dataset
    '''
    df_cleaned = df[df['Proto'] == df['Proto'].mode()[0]]
    # df_cleaned = df[df['IP1'] == df['IP1'].mode()[0]] 

    df_cleaned['group'] = df_cleaned['Time']//GROUP_INTERVAL # generates 10 second groupings of the data.
    
    timefilter = np.sort(df_cleaned['group'].unique())[3:] # takes out first thirty seconds from dataset
    df_cleaned = df_cleaned[df_cleaned['group'].isin(timefilter)] # comment out to include initial peak
    df_cleaned.drop(columns='group', inplace=True)

    return df_cleaned



'''
Transformations
'''

def transform(df):
    '''
    generates new columns/metrics for features in data generation process
    '''
    # df['byte_ratio'] = df.apply(byte_ratio, axis=1) # probably not needed
    # df['pkt_ratio'] = df.apply(pkt_ratio, axis=1) # consider removing too
    
    df['mean_tdelta'] = df['packet_times'].str.split(';').apply(mean_diff) # basically latency


    return df


def featurize(df):
    '''
    generates metrics and features for the data generation process
    '''
    df = transform(df)
    
    # reduce metrics to salient features for the model to use
    features = df.groupby(['group']).agg({
        '1->2Bytes': [min, max, np.mean, np.median, np.var],
        '2->1Bytes': [min, max, np.mean, np.median, np.var],
        '1->2Pkts': [min, max, np.mean, np.median, np.var],
        '2->1Pkts': [min, max, np.mean, np.median, np.var],
        'mean_tdelta': [min, max, np.mean, np.var]
    })
    features.columns = ["_".join(a) for a in features.columns.to_flat_index()] # flattens MultiIndex
    return features

def clean_label_data(filepath, features=False):
    '''
    takes filepath data, cleans it and possibly converts it to features
    dumps it to the output directory

    keyword args:
    filepath -- string file path for a csv file
    features -- boolean for whether to convert data using featurize()
    '''
    if not filepath.endswith('.csv'):
        raise Exception('Not csv format')

    df = pd.read_csv(filepath)
    df = clean_df(df)

    df['group'] = df['Time']//GROUP_INTERVAL # generates 10 second group intervals to groupby on

    if features:
        df = featurize(df) # convert 10 second groups into feature space (1 row)
    else:
        df = transform(df) # only adds columns and does not flatten into feature space
    
    df['label_latency'] = filepath.split('_')[1].split('-')[0] # add labels
    df['label_packet_loss'] = filepath.split('_')[1].split('-')[1] 

    # filenm = pth.split('/')[-1].split('.')[0]
    # df_feat.to_csv(f'{out}{filenm}_features.csv')

    return df

def generate_labels(fileslist=[], folderpath='data', features=False):
    '''
    generates labeled data with either a list of files or a specified directory
    returns dataframe
    '''
    
    temp = 0

    if len(fileslist) > 0:
        for item in fileslist: # TODO maybe merge for loops somehow
            if not isinstance(temp, pd.DataFrame): # init temp as dataframe
                temp = clean_label_data(filepath=pth, features=features)
            else:
                temp = temp.append(clean_label_data(filepath=pth, features=features))

        return temp

    for datafile in os.scandir(folderpath):
        pth = datafile.path
            
        if not isinstance(temp, pd.DataFrame): # init temp as dataframe
            temp = clean_label_data(filepath=pth, features=features)
        else:
            temp = temp.append(clean_label_data(filepath=pth, features=features))
    
    return temp
'''
Feature Generation
'''

def byte_ratio(x):
    '''
    gets byte transfer ratio 
    based on untransformed '1->2Bytes' and '2->1Bytes' columns. 
    >>> df.apply(transfer_ratio, axis=1)
    '''
    if not x['2->1Bytes']:
        return 0
    return x['1->2Bytes'] / x['2->1Bytes']

def pkt_ratio(x):
    '''
    gets packet transfer ratio 
    based on untransformed '1->2Bytes' and '2->1Bytes' columns. 
    >>> df.apply(pkt_ratio, axis=1)
    '''
    if not x['2->1Pkts']:
        return 0
    return x['1->2Pkts'] / x['2->1Pkts']

def mean_diff(lst):
    '''
    returns mean difference in a column, 
    meant to be used on transformed 'packet_times' column
    >>> df['packet_times'].str.split(';').apply(mean_diff)
    '''
    lst = np.array(list(filter(None, lst))) # takes out empty strings
    mn = np.mean([int(t) - int(s) for s, t in zip(lst, lst[1:])]) #TODO use numpy diff if needed
    return 0 if np.isnan(mn) else mn

# rolling average (packets per second)

# distance between peaks and troughs

# period and amplitude