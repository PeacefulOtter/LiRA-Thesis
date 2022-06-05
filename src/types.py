

class LatLonIRI:
    def __init__(self, lat, lon, value, ts):
        self.lat = lat
        self.lon = lon
        self.value = value
        self.ts = ts

    def __repr__(self):
        return str(self.__dict__)


class WayIRI:
    def __init__(self, way_id, way_dist, value):
        self.way_id = way_id
        self.way_dist = way_dist
        self.value = value

    def __repr__(self):
        return str(self.__dict__)


class Dataset:
    def __init__(self, way_id, way_length, X, y):
        self.way_id = way_id
        self.way_length = way_length
        self.X = X
        self.y = y