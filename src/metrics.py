import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from os.path import join
import seaborn as sns
import numpy as np

loss_features = [
    "mean_tdelta_min",
    "mean_tdelta_max",
    "mean_tdelta_mean",
    "max_tdelta_var",
    "1->2Pkts_max",
    "1->2Bytes_var",
    "2->1Bytes_var",
    "1->2Pkts_var",
    "2->1Bytes_var",
    "1->2Bytes_max",
    "1->2Pkts_rolling_2s_mean_max",
    "1->2Pkts_rolling_3s_mean_max",
    "1->2Pkts_rolling_2s_mean_var",
    "2->1Pkts_rolling_2s_mean_var",
    "1->2Pkts_rolling_3s_mean_var",
    "2->1Pkts_rolling_3s_mean_var",
]
latency_features = [
    "1->2Bytes_mean",
    "1->2Bytes_median",
    "1->2Bytes_min",
    "1->2Pkts_mean",
    "1->2Pkts_median",
    "1->2Pkts_min",
    "1->2Pkts_rolling_2s_mean_min",
    "1->2Pkts_rolling_3s_mean_min",
    "2->1Bytes_max",
    "2->1Bytes_mean",
    "2->1Bytes_median",
    "2->1Bytes_min",
    "2->1Pkts_max",
    "2->1Pkts_mean",
    "2->1Pkts_median",
    "2->1Pkts_min",
    "2->1Pkts_rolling_2s_mean_max",
    "2->1Pkts_rolling_2s_mean_min",
    "2->1Pkts_rolling_3s_mean_max",
    "2->1Pkts_rolling_3s_mean_min",
    "2->1Pkts_var",
    "label_latency",
    "label_packet_loss",
    "max_tdelta_max",
    "max_tdelta_mean",
    "mean_tdelta_var",
    "pred_loss",
]

# this assumes you have run the etl/data, features, and train targets
loss_model = pickle.load(open("models/loss_model.pyc", "rb"))
latency_model = pickle.load(open("models/latency_model.pyc", "rb"))


def plot_metrics(data_path="data/out", outdir=None, filename='modelmetrics.png', function=None):
    """Plots sensitivity curve and residual curve"""
    # load in data
    data = pd.read_csv(join(data_path, "0_train_out.csv"))
    vdata = pd.read_csv(join(data_path, "0_validation_out.csv"))
    test = pd.read_csv("test/test_features/test_featureset.csv")

    ## training data
    loss_X = data[loss_features]
    loss_y = np.log(data["label_packet_loss"])  # log loss
    loss_yhat = loss_model.predict(loss_X)

    data["pred_loss"] = loss_yhat  # feed prediction into latency as feature
    latency_X = data[latency_features]
    latency_y = np.log(data["label_latency"])
    latency_yhat = latency_model.predict(latency_X)

    ## validation data
    loss_Xv = vdata[loss_features]
    loss_yv = np.log(vdata["label_packet_loss"])  # log loss
    loss_valid_yhat = loss_model.predict(loss_Xv)

    vdata["pred_loss"] = loss_valid_yhat  # feed prediction into latency as feature
    latency_Xv = vdata[latency_features]
    latency_yv = np.log(vdata["label_latency"])
    latency_valid_yhat = latency_model.predict(latency_Xv)

    ## test data
    loss_test_X = test[loss_features]
    loss_test_y = np.log(test["label_packet_loss"])  # log loss
    loss_test_yhat = loss_model.predict(loss_test_X)

    test["pred_loss"] = loss_test_yhat  # feed prediction into latency as feature
    latency_test_X = test[latency_features]
    latency_test_y = np.log(test["label_latency"])
    latency_test_yhat = latency_model.predict(latency_test_X)

    if function == "sensitivity":
        # set parameters
        upper = 0.15
        lower = 0
        step = 0.0025
        thresh_l = [i * step for i in range(int((upper - lower) / step))]

        def get_acc(target, pred, thresh):
            """helper function for sensitivity curve generation"""
            latency_diff = (target - pred) / target
            return sum(abs(latency_diff) <= thresh) / len(target)

        loss_acc = [0 for i in range(int((upper - lower) / step))]
        loss_valid_acc = [0 for i in range(int((upper - lower) / step))]
        loss_test_acc = [0 for i in range(int((upper - lower) / step))]

        latency_acc = [0 for i in range(int((upper - lower) / step))]
        latency_valid_acc = [0 for i in range(int((upper - lower) / step))]
        latency_test_acc = [0 for i in range(int((upper - lower) / step))]

        for i in range(len(thresh_l)):
            loss_acc[i] += get_acc(loss_y, loss_yhat, thresh_l[i])
            loss_valid_acc[i] = get_acc(loss_yv, loss_valid_yhat, thresh_l[i])
            loss_test_acc[i] = get_acc(loss_test_y, loss_test_yhat, thresh_l[i])

            latency_acc[i] += get_acc(latency_y, latency_yhat, thresh_l[i])
            latency_valid_acc[i] = get_acc(latency_yv, latency_valid_yhat, thresh_l[i])
            latency_test_acc[i] = get_acc(
                latency_test_y, latency_test_yhat, thresh_l[i]
            )

        fig, (ax1, ax2) = plt.subplots(2, figsize=(13, 10))

        ax1.plot(thresh_l, loss_acc, label="training packet loss")
        ax1.plot(thresh_l, loss_valid_acc, label="validation packet loss")
        ax1.plot(thresh_l, loss_test_acc, label="test packet loss")
        ax1.title.set_text("Packet Loss Model Performance over Accuracy Thresholds")
        ax1.set_xlabel("Accuracy Threshold")
        ax1.set_ylabel("Percent Accurately Predicted")
        ax1.axvline(x=0.10, color="r", linestyle="--", alpha=0.5)
        ax1.legend()

        ax2.plot(thresh_l, latency_acc, label="training latency")
        ax2.plot(thresh_l, latency_valid_acc, label="validation latency")
        ax2.plot(thresh_l, latency_test_acc, label="test latency")
        ax2.title.set_text("Latency Model Performance over Accuracy Thresholds")
        ax2.set_xlabel("Accuracy Threshold")
        ax2.set_ylabel("Percent Accurately Predicted")
        ax2.axvline(x=0.10, color="r", linestyle="--", alpha=0.5)
        ax2.legend(loc="lower right")
        plt.savefig(os.path.join(outdir, filename))

    if function == "residual":

        def frac_formatter(value, tick_number):
            """format helper function for residual plot of packet loss"""
            N = np.exp(value).astype(int)
            return f"1/{N}"

        def exp_formatter(value, tick_number):
            # find number of multiples of pi/2
            N = np.exp(value).astype(int)
            return N

        fig, ax = plt.subplots(2, 2, figsize=(16, 10))
        ax[0, 0].scatter(loss_y, loss_y - loss_yhat, c=latency_y)
        ax[0, 0].title.set_text("Packet Loss Training Residuals")
        ax[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(frac_formatter))
        ax[0, 0].axhline(y=0.0, color="r", linestyle="-", alpha=0.5)

        ax[0, 1].scatter(latency_y, latency_y - latency_yhat, c=loss_y)
        ax[0, 1].title.set_text("Latency Training Residuals")
        ax[0, 1].xaxis.set_major_formatter(plt.FuncFormatter(exp_formatter))
        ax[0, 1].axhline(y=0.0, color="r", linestyle="-", alpha=0.5)

        ax[1, 0].scatter(loss_test_y, loss_test_y - loss_test_yhat, c=latency_test_y)
        ax[1, 0].title.set_text("Packet Loss Test Residuals")
        ax[1, 0].xaxis.set_major_formatter(plt.FuncFormatter(frac_formatter))
        ax[1, 0].axhline(y=0.0, color="r", linestyle="-", alpha=0.5)

        ax[1, 1].scatter(
            latency_test_y, latency_test_y - latency_test_yhat, c=loss_test_y
        )
        ax[1, 1].title.set_text("Latency Test Residuals")
        ax[1, 1].xaxis.set_major_formatter(plt.FuncFormatter(exp_formatter))
        ax[1, 1].axhline(y=0.0, color="r", linestyle="-", alpha=0.5)

        plt.savefig(os.path.join(outdir, filename))

        # TODO do the savefig stuff properly and then make it emulate eda.py a bit more
