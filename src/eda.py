
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

"""
Creates a time plot of the number of packets being sent per second

data_filename: filename of a dataset of a single conversation. can take a list of filenames and overlap graphs
outdir: directry to put the graph
filename: name of the graph image to save
function: takes in a string to create a graph using a function to apply to the data
    amplitude: takes the difference between the max and min of the data inside a time window
    mean: takes the average of all the data in a moving time window
    None (default): Plots data as is, no window size needed
window_size: the amount of data points to apply the function to
"""
def plot_timeseries(data_filename, outdir, filename="timeseries.png", function=None, window_size=20):
    # turn filename to coresponding dataframe
    if isinstance(data_filename, str):
        data_filename = [data_filename]
        data_list = [pd.read_csv(data_filename)]
    if isinstance(data_filename, list):
        data_list = [pd.read_csv(file) for file in data_filename]

    plt.figure(figsize=(12, 8), dpi=80)

    # for each time get the difference between the max and min for the next window_size data points
    if function == "amplitude":
        for i in range(len(data_list)):
            data = data_list[i]["1->2Pkts"]
            distance = []
            for i in range(len(data)-window_size):
                distance += [max(data[i:i+window_size]) - min(data[i:i+window_size])]
            plt.plot(distance, label=data_filename[i])
    # for each time get the mean for the next window_size of data points
    elif function == "mean":
        for i in range(len(data_list)):
            data = data_list[i]["1->2Pkts"]
            avge = []
            for i in range(len(data)-window_size):
                avge += [sum(data[i:i+window_size]) / window_size]
            plt.plot(avge, label=data_filename[i])
    # plots the packest sent per second as is
    else:
        for i in range(len(data_list)):
            data = data_list[i]
            plt.plot(data["Time"], data["1->2Pkts"], label=data_filename[i])
    
    # labels the graph
    if function == None:
        plt.subtitle("1->2Pkts vs seconds")
        plt.xlabel("Time (s)")
        plt.ylabel("Packets")
    else:
        plt.suptitle(function + " of 1->2Pkts vs seconds. Window size: " + window_size)
        plt.xlabel("Time (s)")
        plt.ylabel(function + " (packets)")

    plt.legend()
    # saves graph to directory
    plt.savefig(os.path.join(outdir, filename))

"""
Creates a histogram of the number of packets

data_filename: filename of a dataset of a single conversation. can take a list of filenames and overlap graphs
outdir: directry to put the graph
filename: name of the graph image to save
function: takes in a string to create a graph using a function to apply to the data
    time delta: graphs the average time between packets being sent for every second
    None (default): Plots data as is
"""
def plot_histogram(data_filename, outdir, filename="histogram.png", function=None):
    # turn filename to coresponding dataframe
    if isinstance(data_filename, str):
        data_filename = [data_filename]
        data_list = [pd.read_csv(data_filename)]
    if isinstance(data_filename, list):
        data_list = [pd.read_csv(file) for file in data_filename]

    # graphs the average time between packets being sent for every second
    if function == "time delta":
        for i in range(len(data_list)):
            data = data_list[i]["packet_times"]
            avg_time_delta = data.str.split(';').apply(mean_delta)
            plt.plot(avg_time_delta)
    # graphs data as is
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

# returns the average time between packets sent
def mean_delta(tl):
    time_list = np.array(list(filter(None, tl)))
    delta_list = [int(t2) - int(t1) for t1, t2 in zip(time_list, time_list[1:])]
    return 0 if np.isnan(np.mean(delta_list)) else np.mean(delta_list)

