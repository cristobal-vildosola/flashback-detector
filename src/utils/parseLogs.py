import numpy
from features import AutoEncoderFE, ColorLayoutFE
from indexes import LSHIndex, SGHIndex, LinearIndex, KDTreeIndex
from keyframes import MaxHistDiffKS, FPSReductionKS
from utils.files import get_features_dir, get_neighbours_dir, get_results_dir

selectors = [
    FPSReductionKS(n=3),
    MaxHistDiffKS(frames_per_window=1),
]
extractors = [
    ColorLayoutFE(),
    # AutoEncoderFE(dummy=True, model_name='model'),
]

indexes = [
    LinearIndex(dummy=True),
    KDTreeIndex(dummy=True, trees=5),
    SGHIndex(dummy=True, projections=10),
    LSHIndex(dummy=True, projections=5, tables=10),  # color layout
    # LSHIndex(dummy=True, projections=3, tables=10),  # auto encoder
]


def read_selection_log(selector, extractor):
    features_dir = get_features_dir(selector, extractor)


if __name__ == '__main__':
    times = []
    seconds = 0
    with open('log.txt') as log:
        for line in log:
            seconds_chapter, time = line.split(' ')
            times.append(int(time))
            seconds += int(seconds_chapter)

    print(seconds, numpy.sum(times))
    print(seconds * 6 / numpy.sum(times))
