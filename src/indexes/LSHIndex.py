import time

import numpy as np
from nearpy.hashes import RandomDiscretizedProjections

from indexes.HashEngine.HashEngine import HashEngine
from indexes.SearchIndex import SearchIndex


class LSHIndex(SearchIndex):

    def __init__(
            self,
            data: np.ndarray = None,
            labels: np.ndarray = None,
            k: int = 100,
            projections: int = 3,
            bin_width: int = 10,
            tables: int = 3,
            verbose: bool = True,
            dummy: bool = False,
    ):
        self.k = k
        self.projections = projections
        self.tables = tables

        if not dummy:
            if data is None and labels is None:
                raise Exception('data and labels must be numpy.ndarray when not using dummy indexer')
            t0 = time.time()

            self.engine = HashEngine(
                vectors=data,
                labels=labels,
                lshashes=[RandomDiscretizedProjections(f'rbp_{i}', projections, bin_width=bin_width)
                          for i in range(tables)],
                k=k,
                verbose=verbose,
            )
            self.build_time = time.time() - t0

    def neighbours_retrieved(self) -> int:
        return self.k

    def candidate_count(self, vector) -> int:
        return self.engine.candidate_count(vector)

    def search(self, vector):
        point, labels = self.engine.neighbours(vector)
        return labels

    def name(self) -> str:
        return f'LSH_{self.tables}_{self.projections}'
