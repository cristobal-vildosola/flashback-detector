import time
from abc import ABC
from typing import List

import numpy as np
import pyflann

from indexes.SearchIndex import SearchIndex


class FlannIndex(SearchIndex, ABC):

    def __init__(
            self,
            data: np.ndarray = None,
            labels: np.ndarray = None,
            k: int = 100,
            dummy=False,
            **kwargs
    ):
        self.flann = pyflann.FLANN()

        self.k = k
        self.checks = -1

        if not dummy:
            t0 = time.time()
            self.flann.build_index(data.astype('int32'), **kwargs)
            t1 = time.time()

            self.build_time = t1 - t0
            self.labels = labels

    def search(self, vector) -> List:
        [neighbours], dist = self.flann.nn_index(
            vector.astype('int32'), num_neighbors=self.k, checks=self.checks, cores=1)
        results = [self.labels[int(index)] for index in neighbours]
        return results

    def neighbours_retrieved(self) -> int:
        return self.k

    def candidate_count(self, vector):
        return 0


class LinearIndex(FlannIndex):
    def __init__(
            self,
            data: np.ndarray = None,
            labels: np.ndarray = None,
            k: int = 100,
            dummy=False,
    ):
        super().__init__(data=data, labels=labels, k=k, algorithm="linear", dummy=dummy)

    def name(self) -> str:
        return f'linear_{self.k}'


class KDTreeIndex(FlannIndex):
    def __init__(
            self,
            data: np.ndarray = None,
            labels: np.ndarray = None,
            k: int = 100,
            trees: int = 2,
            checks: int = 10,
            dummy=False,
    ):
        super().__init__(data=data, labels=labels, k=k, algorithm="kdtree", trees=trees, dummy=dummy)
        self.trees = trees
        self.checks = checks

    def name(self) -> str:
        return f'kdTree_{self.trees}'


class KMeansTree(FlannIndex):
    def __init__(
            self,
            data: np.ndarray = None,
            labels: np.ndarray = None,
            k: int = 100,
            branching: int = 4,
            checks: int = 10,
            dummy=False,
    ):
        super().__init__(
            data=data, labels=labels, k=k, algorithm='kmeans', branching=branching, iterations=-1, dummy=dummy)
        self.branching = branching
        self.checks = checks

    def name(self) -> str:
        return f'kmeansTree_{self.branching}'
