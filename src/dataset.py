
import numpy as np

def get_datasets(datas):
    datasets = []
    for way_id, data in datas.items():
        X, y = list(map(np.array, zip(*[(pos.dist, pos.val) for pos in data])))
        dataset = Dataset(ways[i], (_reshape(X), _reshape(y)))
        datasets.append(dataset)
    return datasets