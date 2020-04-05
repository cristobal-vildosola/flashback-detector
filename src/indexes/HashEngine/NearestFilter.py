from pyflann import FLANN


class NearestFilter:
    def __init__(self, k: int):
        self.k = k
        self.flann = FLANN()

    def filter(self, v, points, labels):
        [neighbours_i], _ = self.flann.nn(
            points,
            v.astype('float32'),
            num_neighbors=min(self.k, len(points)),
            algorithm='linear'
        )
        return points[neighbours_i], labels[neighbours_i]
