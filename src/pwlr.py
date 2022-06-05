

import pwlf
import numpy as np

from math import floor, ceil

from src.plot import new_axis
from src.types import WayIRI

breakpoints = np.array([3, 6, 12, 20, 35, 50])

def get_breakpoints(X, pred, bp_num):
    length = X.shape[0]
    print(length, bp_num)

    if bp_num > 20 or (bp_num > 5 and length > 250):
        print('too much, splitting')
        low_bp, high_bp = floor(bp_num / 2), ceil(bp_num / 2)
        low_X, high_X = np.array_split(X, 2)
        low_pred, high_pred = np.array_split(pred, 2)
        bp1, yHat1 = get_breakpoints(low_X,  low_pred,  low_bp)
        bp2, yHat2 = get_breakpoints(high_X, high_pred, high_bp)
        return np.append(bp1, bp2), np.append(yHat1, yHat2)

    my_pwlf = pwlf.PiecewiseLinFit(X, pred)

    breakpoints = my_pwlf.fitfast(bp_num, 10) # 5 restarts
    breakpoints = np.unique(breakpoints)
    yHat = my_pwlf.predict(breakpoints)

    return breakpoints, yHat
    
    

def find_breakpoints(dataset, gpr, plotting=False):
    way_length = dataset.way.length
    way_id = dataset.way.way_id
    
    X = np.array(list(range(0, int(way_length)))).reshape(-1, 1)
    pred = gpr.predict(X).reshape(-1, )
    X = X.reshape(-1, )

    factor = max(0.5, way_length / 500) 
    bp_nums = np.round(breakpoints * factor)  # [3, 8, 15]

    acc = []
    mses = []

    for bp_num in bp_nums:

        breakpoints, yHat = get_breakpoints(X, pred, bp_num)

        bp_iris = [
            WayIRI(way_id, dist / way_length, val) 
            for (dist, val) in zip(breakpoints, yHat)
        ]

        interp = np.interp(X, breakpoints, yHat)
        mse = ((pred - interp)**2).mean(axis=0)
        mses.append(mse)

        if plotting:
            ax = new_axis()
            ax.plot(X, pred, c='blue')
            ax.plot(breakpoints, yHat, c='green')
            ax.scatter(breakpoints, yHat, c='green')

        acc.append(bp_iris)

    return acc, mses