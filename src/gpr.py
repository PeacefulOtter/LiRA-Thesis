

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error

import numpy as np


def train_with_kernel(kernel, dataset, n_restarts_optimizer=5):
    X_train, y_train = dataset.get()
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
    gpr = gpr.fit(X_train, y_train)
    return gpr

def train_dataset(dataset, kernel, i):
    gpr = train_with_kernel(kernel(), dataset)
    print(gpr.kernel_)
    mse, score, likelihood, string = test_model(gpr, dataset)
    print(i, '\t', string)
    return gpr, mse, score, likelihood

def train_datasets(datasets, kernel):
    gprs = []
    mses = []
    scores = []
    likelihoods = []
    for i, (_, dataset) in enumerate(datasets.items()):
        gpr, mse, score, likelihood = train_dataset(dataset, kernel, i)
        gprs.append(gpr)
        mses.append(mse)
        scores.append(score)
        likelihoods.append(likelihood)
        
    print(f"\n\t ==> MSE \t Mean: {round(np.mean(mses), 4)}\tVariance: {round(np.var(mses), 4)}")
    print(f"\t ==> Score \t Mean: {round(np.mean(scores), 4)}\tVariance: {round(np.var(scores), 4)}")
    print(f"\t ==> MLL \t Mean: {round(np.mean(likelihoods), 4)}\tVariance: {round(np.var(likelihoods), 4)}")
    return gprs


def test_model(gpr, dataset):
    X, y = dataset.get()
    likelihood = gpr.log_marginal_likelihood(gpr.kernel_.theta)
    mean_pred = gpr.predict(X)
    score = gpr.score(X, y)

    mse = mean_squared_error(y, mean_pred)
    string = f"MSE: {round(mse, 5)}\tScore: {round(score, 3)}\tLog-likelihood: {round(likelihood, 3)}"

    return mse, score, likelihood, string