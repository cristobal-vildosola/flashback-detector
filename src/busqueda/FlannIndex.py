import time
from typing import List

import numpy
import pyflann


class FlannIndex:

    def __init__(self, data: numpy.ndarray, labels: numpy.ndarray, **kwargs):
        self.flann = pyflann.FLANN()

        t0 = time.time()
        self.flann.build_index(data, **kwargs)
        t1 = time.time()

        self.build_time = t1 - t0
        self.labels = labels

    def search(self, busquedas, k=50, checks=10) -> List:
        [neighbours], dist = self.flann.nn_index(busquedas, num_neighbors=k, checks=checks, cores=1)
        results = [self.labels[int(index)] for index in neighbours]
        return results


class Linear(FlannIndex):
    def __init__(self, datos, etiquetas):
        super().__init__(datos, etiquetas, algorithm="linear")


class KDTree(FlannIndex):
    def __init__(self, datos, etiquetas, trees):
        super().__init__(datos, etiquetas, algorithm="kdtree", trees=trees)


class KMeansTree(FlannIndex):
    def __init__(self, datos, etiquetas, branching):
        super().__init__(datos, etiquetas, algorithm='kmeans', branching=branching, iterations=-1)
