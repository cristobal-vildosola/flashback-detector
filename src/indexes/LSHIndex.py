import time

import numpy as np
from nearpy.filters.nearestfilter import NearestFilter
from nearpy.hashes import RandomBinaryProjections

from indexes.HashEngine.HashEngine import HashEngine
from indexes.SearchIndex import SearchIndex


class LSHIndex(SearchIndex):

    def __init__(
            self,
            data: np.ndarray = None,
            labels: np.ndarray = None,
            k: int = 100,
            projections: int = 20,
            tables: int = 3,
            dummy: bool = False,
    ):
        self.k = k
        self.projections = projections
        self.tables = tables

        if not dummy:
            if data is None and labels is None:
                raise Exception('data and labels must be numpy.ndarray when not using dummy indexer')
            t0 = time.time()

            # center data in 0 so that all projections are useful
            self.means = data.mean(axis=0)

            self.engine = HashEngine(
                vectors=data - self.means,
                labels=labels,
                lshashes=[RandomBinaryProjections(f'rbp_{i}', projections)
                          for i in range(tables)],
                vector_filters=[NearestFilter(k)],
                verbose=True,
            )
            self.build_time = time.time() - t0

    def neighbours_retrieved(self) -> int:
        return self.k

    def candidate_count(self, vector) -> int:
        return self.engine.candidate_count(vector - self.means)

    def search(self, vector):
        neighbours = self.engine.neighbours(vector - self.means)
        # vectors = [neighbour[0] for neighbour in neighbours]
        labels = [neighbour[1] for neighbour in neighbours]
        # distances = [neighbour[2] for neighbour in neighbours]
        return labels

    def name(self) -> str:
        return f'LSH_{self.tables}_{self.projections}_{self.k}'
