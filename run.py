

import sys
import json
import pandas as pd

sys.path.insert(0, 'src')
from etl import featurize, clean_df, clean_label_data #generate_data, save_data
from eda import plot_timeseries, plot_histogram
from utils import convert_notebook
from os import listdir
from os.path import isfile, join, expanduser


def main(targets):

    data_config = json.load(open('config/data-params.json'))
    eda_config = json.load(open('config/eda-params.json'))

    # goes under the assumption that all the datafiles have their latency and loss in the name
    # cleans and adds features for all of the csv in the raw data folder
    if 'data' in targets:
        raw_data_path = "data/raw"
        data_csv_files = [join(raw_data_path, f) for f in listdir(raw_data_path)]
        dataframes = [clean_label_data(file) for file in data_csv_files]

        # dataframes = [pd.read_csv(file) for file in data_csv_files]
        
        # dataframes = [clean_df(df) for df in dataframes]
        
        # dataframes = [featurize(df) for df in dataframes]
        
        temp_path = "data/temp"
        for i in range(len(data_csv_files)):
            dataframes[i].to_csv(join(temp_path, listdir(raw_data_path)[i]))
        
        
        """
        data = generate_data(**data_config)
        save_data(data, **data_config)
        """
    if 'eda' in targets: #TODO do later

        temp_path = "data/temp"
        out_path = "notebooks/figures"
        csv1 = [join(temp_path, "s2_200-100-iperf.csv"), join(temp_path, "s2_200-500-iperf.csv")]
        csv2 = [join(temp_path, "s2_200-10000-iperf.csv"), join(temp_path, "s2_200-50000-iperf.csv")]
        
        plot_histogram(csv1+csv2, out_path, filename="100-500-10000-50000_hist.png", function="time delta")
        
        plot_timeseries(csv1, out_path, filename="100-500_amp.png", function="amplitude")
        plot_timeseries(csv2, out_path, filename="10000-50000_amp.png", function="amplitude")
        
        plot_timeseries(csv1, out_path, filename="100-500_mean.png", function="mean")
        plot_timeseries(csv2, out_path, filename="10000-50000_mean.png", function="mean")
        
        """
        try:
            data
        except NameError:
            data = pd.read_csv(data_config['data_fp'])

        generate_stats(data, **eda_config)
        
        # execute notebook / convert to html
        convert_notebook(**eda_config)
        """

if __name__ == '__main__':

    targets = sys.argv[1:]
    main(targets)
