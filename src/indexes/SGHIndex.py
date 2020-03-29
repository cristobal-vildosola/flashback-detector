import time

import numpy as np
from nearpy.filters.nearestfilter import NearestFilter

from indexes.HashEngine.HashEngine import HashEngine
from indexes.HashEngine.SGHash import SGHash
from indexes.SearchIndex import SearchIndex


class SGHIndex(SearchIndex):

    def __init__(
            self,
            data: np.ndarray = None,
            labels: np.ndarray = None,
            k: int = 100,
            projections: int = 16,
            training_split: float = 0.1,
            num_bases: int = 300,
            dummy: bool = False,
    ):
        self.k = k
        self.projections = projections

        if not dummy:
            if data is None and labels is None:
                raise Exception('data and labels must be numpy.ndarray when not using dummy indexer')
            t0 = time.time()

            training_data = np.random.permutation(data)[:int(data.shape[0] * training_split)]
            self.engine = HashEngine(
                vectors=data,
                labels=labels,
                lshashes=[SGHash('sgh', training_data=training_data, projections=projections, num_bases=num_bases)],
                vector_filters=[NearestFilter(k)],
                verbose=True,
            )
            self.build_time = time.time() - t0

    def neighbours_retrieved(self) -> int:
        return self.k

    def candidate_count(self, vector) -> int:
        return self.engine.candidate_count(vector)

    def search(self, vector):
        neighbours = self.engine.neighbours(vector)
        # vectors = [neighbour[0] for neighbour in neighbours]
        labels = [neighbour[1] for neighbour in neighbours]
        # distances = [neighbour[2] for neighbour in neighbours]
        return labels

    def name(self) -> str:
        return f'SGH_{self.projections}_{self.k}'
