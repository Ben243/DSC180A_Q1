import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import join

upper = .15
lower = 0
step = .0025
thresh_l = [i*step for i in range(int((upper-lower) / step))]

def unpickle(model_name, model_path):
    return pickle.load(open(join(model_path, model_name), 'rb'))

def get_acc(target, pred, thresh):
    percent_diff = (target - pred) / target
    return sum(abs(percent_diff) <= thresh) / len(target)

def generate_metrics(model, time, label, metric_path, data_path, test_data_path, test):
    
    if test:
        test_acc = [0 for i in thresh_l]
        test_data = pd.read_csv(join(test_data_path, listdir(test_data_path)[-1]))
        test_pred = model.predict(test_data)
        if label == 'loss':
            y_test = np.log(test_data['label_packet_loss'])
            other_label_data = np.log(test_data['label_latency'])
        if label == 'latency':
            y_test = np.log(test_data['label_latency'])
            other_label_data = np.log(test_data['label_packet_loss'])
        
        for i in range(len(thresh_l)):
            test_acc[i] = get_acc(y_test, test_pred, thresh_l[i])
        
        plt.figure(figsize=(10,6))

        plt.plot(thresh_l, test_acc, label=f'test {label}')
        plt.title(f'{label} sensitivity curve')
        plt.legend()

        figure_name = f'test_{label}_sensitivity_curve.png'
        plt.savefig(join(metric_path, figure_name))

        plt.figure(figsize=(10,6))
        plt.scatter(y_test, test_pred - y_test, c=other_label_data)
        cb = plt.colorbar()
        cb.set_label(f'Color Scale latency')

        figure_name = f'test_residual_{label}_plot.png'
        plt.savefig(join(metric_path, figure_name))
        
    else:
        featurelst = listdir(data_path)
        data_file = []

        for filename in featurelst:
            if time in filename:
                if 'train' in filename:
                    train_file = filename
                if 'validation' in filename:
                    v_file = filename

        train_data = pd.read_csv(join(data_path, train_file))
        vdata = pd.read_csv(join(data_path, v_file))
        test_data = pd.read_csv(join(test_data_path, listdir(test_data_path)[-1]))

        train_pred = model.predict(train_data)
        valid_pred = model.predict(vdata)
        test_pred = model.predict(test_data)

        acc = [0 for i in thresh_l]
        valid_acc = [0 for i in thresh_l]
        test_acc = [0 for i in thresh_l]
        
        if label == 'loss':
            other = 'log latency'
            y_train = np.log(train_data['label_packet_loss']) #log loss
            y_valid = np.log(vdata['label_packet_loss'])
            y_test = np.log(test_data['label_packet_loss'])
            other_label_data = np.log(vdata['label_latency'])

        if label == 'latency':
            other = 'log loss'
            y_train = np.log(train_data['label_latency']) #log loss
            y_valid = np.log(vdata['label_latency'])
            y_test = np.log(test_data['label_latency'])
            other_label_data = np.log(vdata['label_packet_loss'])

        for i in range(len(thresh_l)):
            acc[i] += get_acc(y_train, train_pred, thresh_l[i])
            valid_acc[i] = get_acc(y_valid, valid_pred, thresh_l[i])
            test_acc[i] = get_acc(y_test, test_pred, thresh_l[i])

        plt.figure(figsize=(10,6))

        plt.plot(thresh_l, acc, label=f'training {label}')
        plt.plot(thresh_l, valid_acc, label=f'validation {label}')
        plt.plot(thresh_l, test_acc, label=f'test {label}')
        plt.title(f'{label} sensitivity curve')
        plt.legend()

        figure_name = f'{time}_{label}_sensitivity_curve.png'
        plt.savefig(join(metric_path, figure_name))

        plt.figure(figsize=(10,6))
        plt.scatter(y_valid, valid_pred - y_valid, c=other_label_data)
        cb = plt.colorbar()
        cb.set_label(f'Color Scale {other}')

        figure_name = f'{time}_{label}_residual_plot.png'
        plt.savefig(join(metric_path, figure_name))
    
    
    return