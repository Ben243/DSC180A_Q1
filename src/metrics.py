import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    
    def format_func1(value, tick_number):
        # find number of multiples of pi/2
        N = np.exp(value).astype(int)
        return f"1/{N}"

    def format_func2(value, tick_number):
        # find number of multiples of pi/2
        N = np.exp(value).astype(int)
        return N
    
    if test:
        test_acc = [0 for i in thresh_l]
        test_data = pd.read_csv(join(test_data_path, listdir(test_data_path)[-1]))
        test_pred = model.predict(test_data)
        if label == 'loss':
            other='latency'
            y_test = np.log(test_data['label_packet_loss'])
            other_label_data = np.log(test_data['label_latency'])
        if label == 'latency':
            other = 'loss'
            y_test = np.log(test_data['label_latency'])
            other_label_data = np.log(test_data['label_packet_loss'])
        
        index = 0
        for i in range(len(thresh_l)):
            if thresh_l[i] == 0.1:
                index = i
            test_acc[i] = get_acc(y_test, test_pred, thresh_l[i])
        
        
        
        plt.figure(figsize=(10,6))

        plt.plot(thresh_l, test_acc, label=f'test {label}')
        plt.title(f'Accuracy at 10% tolerance: test_acc:{test_acc[index]:.3f}')
        plt.xlabel('threshold')
        plt.ylabel('accuracy')
        plt.axvline(x=0.10, color='r', linestyle='--', alpha=0.5)
        plt.suptitle(f'{label} sensitivity curve', fontsize=18)

        plt.legend()

        figure_name = figure_name = f'test_{label}_sensitivity_curve.png'
        plt.savefig(join(metric_path, figure_name))

        plt.figure(figsize=(10,6))
        scatter = plt.scatter(y_test, test_pred - y_test, c=other_label_data)
        plt.title(f'{label} Residual plot')
        plt.xlabel(label)
        plt.ylabel('Residual')
        
        plt.axhline(y=0.0, color='r', linestyle='-', alpha=0.5)
        
        if label == 'loss':
            plt.axes().xaxis.set_major_formatter(plt.FuncFormatter(format_func1))
            plt.colorbar(scatter, format=ticker.FuncFormatter(format_func2)).set_label(f'Color Scale of {other}')
        if label == 'latency':
            plt.axes().xaxis.set_major_formatter(plt.FuncFormatter(format_func2))
            plt.colorbar(scatter, format=ticker.FuncFormatter(format_func1)).set_label(f'Color Scale of {other}')


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

        index = 0
        for i in range(len(thresh_l)):
            if thresh_l[i] == .1:
                index = i
            acc[i] += get_acc(y_train, train_pred, thresh_l[i])
            valid_acc[i] = get_acc(y_valid, valid_pred, thresh_l[i])
            test_acc[i] = get_acc(y_test, test_pred, thresh_l[i])

            
        plt.figure(figsize=(10,6))

        plt.plot(thresh_l, acc, label=f'training {label}')
        plt.plot(thresh_l, valid_acc, label=f'validation {label}')
        plt.plot(thresh_l, test_acc, label=f'test {label}')
        plt.title(f'Accuracy at 10% tolerance: training:{acc[index]:.3f}, validation:{valid_acc[index]:.3f}, test_acc:{test_acc[index]:.3f}')
        plt.xlabel('threshold')
        plt.ylabel('accuracy')
        plt.axvline(x=0.10, color='r', linestyle='--', alpha=0.5)
        plt.suptitle(f'{label} sensitivity curve', fontsize=18)

        plt.legend()

        figure_name = f'{time}_{label}_sensitivity_curve.png'
        plt.savefig(join(metric_path, figure_name))

        plt.figure(figsize=(10,6))
        scatter = plt.scatter(y_valid, valid_pred - y_valid, c=other_label_data)
        plt.title(f'{label} Residual plot')
        plt.xlabel(label)
        plt.ylabel('Residual')
        
        plt.axhline(y=0.0, color='r', linestyle='-', alpha=0.5)
        
        if label == 'loss':
            plt.axes().xaxis.set_major_formatter(plt.FuncFormatter(format_func1))
            plt.colorbar(scatter, format=ticker.FuncFormatter(format_func2)).set_label(f'Color Scale of {other}')
        if label == 'latency':
            plt.axes().xaxis.set_major_formatter(plt.FuncFormatter(format_func2))
            plt.colorbar(scatter, format=ticker.FuncFormatter(format_func1)).set_label(f'Color Scale of {other}')


        figure_name = f'{time}_{label}_residual_plot.png'
        plt.savefig(join(metric_path, figure_name))
    
    
    return