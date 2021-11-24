

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
    df['max_tdelta'] = df['packet_times'].str.split(';').apply(max_diff) # basically latency
    df['1->2Pkts_rolling_2s_mean'] = df['1->2Pkts'].rolling(2).mean()
    df['2->1Pkts_rolling_2s_mean'] = df['2->1Pkts'].rolling(2).mean()
    df['1->2Pkts_rolling_3s_mean'] = df['1->2Pkts'].rolling(3).mean()
    df['2->1Pkts_rolling_3s_mean'] = df['2->1Pkts'].rolling(3).mean()
    df['2->1_interpacket'] = df['packet_dirs'].str.split(';').apply(cleanlist).apply(lambda x: np.diff(np.where(x == 1)[0]).mean())
    df['1->2_interpacket'] = df['packet_dirs'].str.split(';').apply(cleanlist).apply(lambda x: np.diff(np.where(x == 2)[0]).mean())

    
    # df['1->2Bytes_rolling_2s_mean'] = df['1->2Bytes'].rolling(2).mean() # further analysis needed for use
    # df['2->1Bytes_rolling_2s_mean'] = df['1->2Bytes'].rolling(2).mean()
    return df


def featurize(df):
    '''
    generates metrics and features for the data generation process
    '''
    df = transform(df)
    
    df[["packet_sizes", 'packet_dirs']] = df[["packet_sizes", 'packet_dirs']].apply(lambda x: x.str.split(';').apply(cleanlist))
    df['1->2Mean_Bytes'] = df.apply(lambda x: get_packet_dir_sizes_mean(x.packet_sizes, x.packet_dirs, 2), axis=1).mean()
    df['1->2Ct_Pkts'] = df.apply(lambda x: get_packet_dir_sizes_ct(x.packet_sizes, x.packet_dirs, 2), axis=1).mean()

    # reduce metrics to salient features for the model to use
    features = df.groupby(['group']).agg({
        '1->2Bytes': [min, max, np.mean, np.median, np.std, np.var],
        '2->1Bytes': [min, max, np.mean, np.median, np.std, np.var],
        '1->2Pkts': [min, max, np.mean, np.median, np.std, np.var, sum],
        '2->1Pkts': [min, max, np.mean, np.median, np.std, np.var, sum],
        'mean_tdelta': [min, max, np.mean, np.var, np.std],
        'max_tdelta': [max, np.mean, np.var, np.std],
        '1->2Pkts_rolling_2s_mean': [min, max, np.var, np.std, sum],
        '2->1Pkts_rolling_2s_mean': [min, max, np.var, np.std, sum],
        '1->2Pkts_rolling_3s_mean': [min, max, np.var, np.std, sum],
        '2->1Pkts_rolling_3s_mean': [min, max, np.var, np.std, sum],
        '2->1_interpacket':[np.mean],
        '1->2_interpacket':[np.mean]
        
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

    df['group'] = df['Time']//GROUP_INTERVAL  # generates 10 second group intervals to groupby on

    if features:
        df = featurize(df) # convert 10 second groups into feature space (1 row)
    else:
        df = transform(df) # only adds columns and does not flatten into feature space
    
    df['label_latency'] = filepath.split('_')[1].split('-')[0] # add labels
    df['label_packet_loss'] = filepath.split('_')[1].split('-')[1] 

    # df['group'] = int(str(df['label_packet_loss']) + str(df['label_latency']) + \
    #     str(df['group'])) #TODO utilize if unique groups are necessary

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
            #TODO change group id method, they overlap if done concurrently
        else:
            temp = temp.append(clean_label_data(filepath=pth, features=features))
    
    return temp
'''
Feature Generation
'''

def cleanlist(lst):
    '''
    helper function to help clean splitted semicolon separated value columns
    removes empty string values and attempts to cast to int
    '''
    return np.array(list(filter(None, [x for x in lst if x not in ["[",']', ' ', '\n']])), dtype=int)

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

def max_diff(lst):
    '''
    returns max difference in a column, 
    meant to be used on transformed 'packet_times' column
    >>> df['packet_times'].str.split(';').apply(max_diff)
    '''
    lst = np.array(list(filter(None, lst))).astype(np.int64)
    # mn = max([int(t) - int(s) for s, t in zip(lst, lst[1:])]) if len(lst) > 0 else np.nan
    diffs = np.diff(lst)
    mn = max(diffs) if len(diffs) > 0 else np.nan # length of diffs might be zero
    return 0 if np.isnan(mn) else mn

def get_packet_dir_sizes(sizes, dir_, value):
    """gets directional filtered output of one direction of traffic"""
    sizes = cleanlist(sizes)
    dir_ = cleanlist(dir_)
    mask_ = dir_ == value
    return sizes[mask_]

def get_packet_dir_sizes_mean(sizes, dir_, value):
    """gets mean of directional filtered output of one direction of traffic"""
    sizes = cleanlist(sizes)
    dir_ = cleanlist(dir_)
    mask_ = dir_ == value
    return sizes[mask_].mean()

def get_packet_dir_sizes_var(sizes, dir_, value):
    """gets variance of directional filtered output of one direction of traffic"""
    sizes = cleanlist(sizes)
    dir_ = cleanlist(dir_)
    mask_ = dir_ == value
    return sizes[mask_].var()

def get_packet_dir_sizes_ct(sizes, dir_, value):
    """gets packet counts of directional filtered output of one direction of traffic"""
    sizes = cleanlist(sizes)
    dir_ = cleanlist(dir_)
    mask_ = dir_ == value
    return sizes[mask_].shape[0]

# distance between peaks and troughs

# period and amplitude