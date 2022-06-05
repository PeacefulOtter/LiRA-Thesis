
import numpy as np 

class LatLonIRI:
    def __init__(self, lat, lon, value, ts):
        self.lat = lat
        self.lon = lon
        self.value = value
        self.ts = ts

    def __repr__(self):
        return str(self.__dict__)


class WayIRI:
    def __init__(self, way_dist, value):
        self.way_dist = way_dist
        self.value = value

    def __repr__(self):
        return str(self.__dict__)


def get_dataset(iris):
    return list(map(
        np.array, 
        zip(*[
            (pos.way_dist, pos.value) 
            for pos in iris
        ])
    ))

class Dataset:
    def __init__(self):
        self.X = np.array([])
        self.y = np.array([])

    def append(self, way_dist, value):
        self.X = np.append(self.X, way_dist)
        self.y = np.append(self.y, value)

    def sort(self):
        indices = np.argsort(self.X)
        self.X = self.X[indices]
        self.y = self.y[indices]

    def get(self):
        return self.X.reshape(-1, 1), self.y.reshape(-1, 1)

    def __repr__(self):
        return str(self.__dict__)