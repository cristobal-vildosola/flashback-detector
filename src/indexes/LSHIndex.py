import time

from nearpy.distances.euclidean import EuclideanDistance
from nearpy.filters.nearestfilter import NearestFilter
from nearpy.hashes import RandomBinaryProjections

from indexes.OptimizedEngine import OptimizedEngine
from indexes.SearchIndex import SearchIndex


class BynaryLSHIndex(SearchIndex):

    def __init__(self, data=None, labels=None, k=20, projections=20, tables=3, dummy=False):
        if not dummy:
            if data is None and labels is None:
                raise Exception('data and labels must be numpy.ndarray when not using dummy indexer')
            t0 = time.time()

            self.engine = OptimizedEngine(
                vectors=data,
                labels=labels,
                lshashes=[RandomBinaryProjections(f'rbp_{i}', projections)
                          for i in range(tables)],
                distance=EuclideanDistance(),
                vector_filters=[NearestFilter(k)],
                verbose=True,
            )
            self.build_time = time.time() - t0

        self.k = k

        self.projections = projections
        self.tables = tables

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
        return f'LSH_BP_{self.tables}_{self.projections}'

# np.divide(projection, np.abs(projection), out=np.zeros_like(projection, dtype=int), where=projection!=0) + 1
