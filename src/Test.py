import time
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from features import AutoEncoderFE, ColorLayoutFE, FeatureExtractor
from indexes import LSHIndex, SGHIndex, LinearIndex, KDTreeIndex, SearchIndex
from keyframes import KeyframeSelector, MaxHistDiffKS, FPSReductionKS
from utils.files import get_neighbours_dir, get_results_dir


class Frame:
    def __init__(self, video: str, timestamp: float, index: int):
        self.video: str = video
        self.timestamp: float = timestamp
        self.index: int = index


class Neighbours:
    def __init__(self, timestamp: float, frames: List[Frame]):
        self.timestamp: float = timestamp
        self.frames: List[Frame] = frames


def read_neighbours(neighbours_file: str) -> List[Neighbours]:
    """
    reads a neighbours file.

    :param neighbours_file: full path to the neighbours file
    """
    neighbours_list = list()

    with open(neighbours_file, 'r') as log:
        for line in log:
            # split time from frames and parse
            timestamp, neighbours = line.split(' $ ')
            timestamp = float(timestamp)
            neighbours = neighbours.split(' | ')

            # parse frames
            frames = []
            for neighbour in neighbours:
                # split neighbour data
                neighbours_file, tiempo_frame, indice = neighbour.split(' # ')
                frames.append(Frame(video=neighbours_file, timestamp=float(tiempo_frame), index=int(indice)))

            # add frame neighbours to the list
            neighbours_list.append(Neighbours(timestamp=timestamp, frames=frames))

    return neighbours_list


def find_copies(
        video_name: str,
        selector: KeyframeSelector,
        extractor: FeatureExtractor,
        index: SearchIndex,
):
    print(f'searching for copies in {video_name}')
    t0 = time.time()

    # read neghbours
    neighbours_path = get_neighbours_dir(selector=selector, extractor=extractor, index=index)
    neighbours_list = read_neighbours(f'{neighbours_path}/{video_name}.txt')

    videos = {}

    for neighbours in neighbours_list:
        for frame in neighbours.frames:

            video = frame.video
            if video not in videos:
                videos[video] = []
            videos[video].append((neighbours.timestamp, frame.timestamp))

    for video in videos.keys():
        points = np.array(videos[video])
        plt.plot(points[:, 0], points[:, 1], '.')
        plt.title(f'{video}')
        plt.show()

    return


def main():
    selectors = [
        FPSReductionKS(n=3),
        # MaxHistDiffKS(frames_per_window=1),
    ]
    extractors = [
        ColorLayoutFE(),
        AutoEncoderFE(dummy=True, model_name='model'),
    ]
    indexes = [
        # LinearIndex(dummy=True, k=100),
        KDTreeIndex(dummy=True, trees=5, k=100),
        # SGHIndex(dummy=True, projections=14, tables=2, k=100),
        # LSHIndex(dummy=True, projections=14, k=100),
    ]

    for selector in selectors:
        for extractor in extractors:
            for index in indexes:
                find_copies(
                    video_name='385',
                    selector=selector,
                    extractor=extractor,
                    index=index,
                )
    return


if __name__ == '__main__':
    main()
