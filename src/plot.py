
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

def plot_data(X, y, ax=None):
    axis = ax 
    if ax == None:
        axis = new_axis()
    axis.plot(X, y, linestyle="-")
    return axis


def plot_gpr(gpr, dataset, way_length, ax, with_std=True):
    pred_X = np.array(list(range(0, int(way_length)))).reshape(-1, 1)
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

    ax.scatter(dataset.X, dataset.y, label="Observations")
    ax.plot(pred_X, _mean_pred, label="Mean prediction")
    
def plot_gprs(gprs, datasets, way_lengths):
    for i, (way_id, dataset) in enumerate(datasets.items()):
        gpr = gprs[i]
        way_length = way_lengths[way_id]
        ax = new_axis()
        plot_gpr(gpr, dataset, way_length, ax, with_std=True)
        _, _, _, string = test_model(gpr, dataset)
        title = str(gpr.kernel_) + '\n' + string
        ax.title.set_text(title)