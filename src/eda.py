
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np


def plot_timeseries(data_filename, outdir, filename="timeseries.png", function=None, window_size=20):

    if isinstance(data_filename, str):
        data_filename = [data_filename]
        data_list = [pd.read_csv(data_filename)]
    if isinstance(data_filename, list):
        data_list = [pd.read_csv(file) for file in data_filename]

    plt.figure(figsize=(12, 8), dpi=80)

    if function == "amplitude":
        for i in range(len(data_list)):
            data = data_list[i]["1->2Pkts"]
            distance = []
            for i in range(len(data)-window_size):
                distance += [max(data[i:i+window_size]) - min(data[i:i+window_size])]
            plt.plot(distance, label=data_filename[i])
    elif function == "mean":
        for i in range(len(data_list)):
            data = data_list[i]["1->2Pkts"]
            avge = []
            for i in range(len(data)-window_size):
                avge += [sum(data[i:i+window_size]) / window_size]
            plt.plot(avge, label=data_filename[i])
    else:
        for i in range(len(data_list)):
            data = data_list[i]
            plt.plot(data["Time"], data["1->2Pkts"], label=data_filename[i])
    
    if function == None:
        plt.subtitle("1->2Pkts vs seconds")
        plt.xlabel("Time (s)")
        plt.ylabel("Packets")
    else:
        plt.suptitle(function + " of 1->2Pkts vs seconds. Window size: " + window_size)
        plt.xlabel("Time (s)")
        plt.ylabel(function + " (packets)")

    plt.legend()

    plt.savefig(os.path.join(outdir, filename))


def plot_histogram(data_filename, outdir, filename="histogram.png", function=None, bin_size=10):

    if isinstance(data_filename, str):
        data_filename = [data_filename]
        data_list = [pd.read_csv(data_filename)]
    if isinstance(data_filename, list):
        data_list = [pd.read_csv(file) for file in data_filename]

    if function == "time delta":
        for i in range(len(data_list)):
            data = data_list[i]["packet_times"]
            avg_time_delta = data.str.split(';').apply(mean_delta)
            plt.plot(avg_time_delta)
    else:
        for i in range(len(data_list)):
            data = data_list[i]
            plt.plot(data["Time"], data["1->2Pkts"], label=data_filename[i])
    
    if function == None:
        plt.subtitle("Frequency of number of packets sent per second")
        plt.xlabel("Packets/sec")
    else:
        plt.suptitle("Average time between packets sent")
        plt.xlabel("Time (ms)")

    plt.legend()

    plt.savefig(os.path.join(outdir, filename))

def mean_delta(tl):
    time_list = np.array(list(filter(None, tl)))
    delta_list = [int(t2) - int(t1) for t1, t2 in zip(time_list, time_list[1:])]
    return 0 if np.isnan(np.mean(delta_list)) else np.mean(delta_list)

