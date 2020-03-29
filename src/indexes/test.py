import numpy
from nearpy.distances.manhattan import ManhattanDistance
from nearpy.filters.nearestfilter import NearestFilter
from nearpy.hashes import RandomDiscretizedProjections

from indexes.HashEngine.HashEngine import HashEngine

if __name__ == '__main__':
    dimension = 192
    vectors = numpy.random.rand(1000000, dimension)

    engine = HashEngine(
        vectors,
        lshashes=[
            RandomDiscretizedProjections('rdp_1', 2, 50),
            RandomDiscretizedProjections('rdp_2', 2, 50),
            RandomDiscretizedProjections('rdp_3', 2, 50)
        ],
        distance=ManhattanDistance(),
        vector_filters=[NearestFilter(10)],
        verbose=True
    )

    query = numpy.random.randn(dimension)
    results = engine.neighbours(query)

    for result in results:
        print(result[1], result[2])
