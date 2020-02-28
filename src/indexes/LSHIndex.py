import time

from nearpy.distances.euclidean import EuclideanDistance
from nearpy.filters.nearestfilter import NearestFilter
from nearpy.hashes import RandomDiscretizedProjections

from indexes.OptimizedEngine import OptimizedEngine


class LSHIndex:

    def __init__(self, data, labels, k=20, projections=3, bin_width=100, tables=3):
        t0 = time.time()
        self.engine = OptimizedEngine(
            vectors=data,
            lshashes=[RandomDiscretizedProjections(f'rdp_{i}', projections, bin_width=bin_width)
                      for i in range(tables)],
            distance=EuclideanDistance(),
            vector_filters=[NearestFilter(k)],
            verbose=True,
        )
        self.k = k

        self.labels = labels
        self.build_time = time.time() - t0

    def search(self, vector):
        neighbours = self.engine.neighbours(vector)
        results = [self.labels[int(neighbour[1])] for neighbour in neighbours]  # labels
        # distances = [neighbour[2] for neighbour in neighbours]  # distance
        return results

    def candidate_count(self, vector):
        return self.engine.candidate_count(vector)
