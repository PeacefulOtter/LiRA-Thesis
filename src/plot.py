
import matplotlib.pyplot as plt
import numpy as np

from src.gpr import test_model

def new_plot(title='Figure', w=9, h=2):
    fig = plt.figure(title)
    wi, hi = fig.get_size_inches()
    fig.set_size_inches(w, h)

def new_axis(n=1):
    fig, ax = plt.subplots(n, 1)
    wi, hi = fig.get_size_inches()
    fig.set_size_inches(hi*2, hi * n) 
    ax.set_xlabel('distance (meters)')
    ax.set_ylabel('IRI')
    return ax

def plot_simple_data(X, y, ax=None):
    axis = ax 
    if ax == None:
        axis = new_axis()
    axis.plot(X, y, linestyle="-")
    return axis

def plot_data(X, y, X_train, y_train, X_test, y_test, ax=None):
    axis = plot_simple_data(X, y, ax)
    axis.scatter(X_train, y_train, linewidths=1)
    axis.scatter(X_test, y_test, c='red', linewidths=1)
    axis.legend()
    axis.xlabel("$dist (m)$")
    axis.ylabel("$IRI$")
    return axis

def plot_simple_datasets(datasets):
    for dataset in datasets:
        (X, y) = dataset.data
        ax = new_axis()
        ax.scatter(X, y, c='green')
        # X_1 = np.array(X)[X<=1]
        # X_2 = np.array(X)[X>1]
        # ax.scatter(X_1, y[:X_1.shape[0]], c='green')
        # ax.scatter(X_2, y[X_1.shape[0]:], c='orange')

def plot_dataset(dataset):
    (X, y, X_train, y_train, X_test, y_test) = dataset.data
    return plot_data(X, y, X_train, y_train, X_test, y_test)

def plot_datasets(datasets):
    # colors = ['blue', 'green', 'orange', 'purple', 'yellow']
    for dataset in datasets:
        plot_dataset(dataset)

def plot_gpr(gpr, maxX, X_train, y_train, ax, with_std=True):
    print(maxX)
    pred_X = np.array(list(range(0, int(maxX)))).reshape(-1, 1)
    mean_pred, std_pred = gpr.predict(pred_X, return_std=True)
    _pred_X, _mean_pred, _std_pred = pred_X.flatten(), mean_pred.flatten(), std_pred.flatten()

    if with_std:
        ax.fill_between(
            _pred_X,
            _mean_pred - 1.96 * _std_pred,
            _mean_pred + 1.96 * _std_pred,
            alpha=0.3,
            label=r"95% confidence interval",
        )

    ax.scatter(X_train, y_train, label="Observations")
    ax.plot(pred_X, _mean_pred, label="Mean prediction")
    

def plot_gpr_pred(gpr, dataset, with_std=True):
    (X, y, X_train, y_train, X_test, y_test) = dataset.data
    # ax = plot_data(X, y, X_train, y_train, X_test, y_test)
    ax = new_axis()
    plot_gpr(gpr, dataset.way.length, X_train, y_train, ax, with_std)

def plot_simple_gprs(gprs, datasets):
    for i, dataset in enumerate(datasets):
        (X, y) = dataset.data
        gpr = gprs[i]
        ax = new_axis()
        plot_gpr(gpr, dataset.way.length, X, y, ax, with_std=True)
        mse, score, likelihood, string = test_model(gpr, X, y)
        title = str(gpr.kernel_) + '\n' + string
        ax.title.set_text(title)
        
def plot_gprs(gprs, datasets):
    for i, dataset in enumerate(datasets):
        plot_gpr_pred(gprs[i], dataset, with_std=True)