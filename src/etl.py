

import os
import numpy as np
import pandas as pd


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

    df_cleaned['group'] = df_group['Time']//10 # generates 10 second groupings of the data.
    df_cleaned = df_cleaned[df_cleaned['group'].isin()]
    timefilter = np.sort(df_cleaned['group'].unique())[3:] # takes out first thirty seconds from dataset
    df_cleaned = df_cleaned[df_cleaned['group'].isin(timefilter)] # comment out to include initial peak
    df_cleaned.drop(columns='group', inplace=True)

    return df_cleaned



'''
Transformation
'''

def featurize(df):
    '''
    generates metrics and features for the data generation process
    '''
    ## generate new columns/metrics for features
    # df['byte_ratio'] = df.apply(byte_ratio, axis=1) # probably not needed
    # df['pkt_ratio'] = df.apply(pkt_ratio, axis=1) # consider removing too
    
    df['mean_tdelta'] = df['packet_times'].str.split(';').apply(mean_diff) # basically latency
    
    # reduce metrics to salient features for the model to use
    group = df.groupby(['group']).agg({
        '1->2Bytes': [min, max, np.mean, np.median, np.var],
        '2->1Bytes': [min, max, np.mean, np.median, np.var],
        '1->2Pkts': [min, max, np.mean, np.median, np.var],
        '2->1Pkts': [min, max, np.mean, np.median, np.var],
        'mean_tdelta': [min, max, np.mean, np.var]
    })
    
    return group

def data_generation(filepath, out='data/temp/'):
    '''
    takes filepath data, cleans it and dumps it to temp
    '''
    pth = datafile.path
    if not pth.endswith('.csv'):
        raise Exception('not csv format')
        # print(pth.split('_')[1].split('-')[:-1])
    # print(pd.read_csv(pth).head())

    df = pd.read_csv(filepath)
    df = clean_df(df)

    df = df[df['Proto'] == df['Proto'].mode()[0]] # cleaning from any IPV6
    df['group'] = df['Time']//10 # generates 10 second group intervals to groupby on
    df_feat = featurize(df) # make groups into feature space

    filenm = pth.split('/')[-1].split('.')[0]
    df_feat.to_csv(f'{out}{filenm}_features.csv')

    return df_feat

def generate_using_all():
    # for datafile in os.scandir('data'):
    #     pth = datafile.path
    #     if pth.endswith('.csv'):
    #         # print(pth.split('_')[1].split('-')[:-1])
    #         print(pd.read_csv(pth).head())

    #         df = pd.read_csv(pth)
    #         df = clean_df(df)

    #         df = df[df['Proto'] == df['Proto'].mode()[0]] # cleaning from any IPV6
    #         df['group'] = df['Time']//10 # generates 10 second group intervals to groupby on
    #         df_feat = featurize(df) # make groups into feature space
    #         #TODO return a combined temp table with all different packet ratios
    return
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