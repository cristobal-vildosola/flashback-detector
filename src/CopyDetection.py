import os
import time
from typing import List

import numpy as np

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

    :param neighbours_file: full path to the neighbours file.
    """
    neighbours_list = list()

    # stats
    videos = dict()
    num_neighbours = []

    with open(neighbours_file, 'r') as log:
        for line in log:
            # split time from frames and parse
            timestamp, neighbours = line.split(' $ ')
            timestamp = float(timestamp)
            neighbours = neighbours.split(' | ')

            num_neighbours.append(len(neighbours))

            # parse frames
            frames = []
            for neighbour in neighbours:
                # split neighbour data
                neighbours_file, tiempo_frame, indice = neighbour.split(' # ')

                frames.append(Frame(video=neighbours_file, timestamp=float(tiempo_frame), index=int(indice)))

                # count video ocurrences
                videos[neighbours_file] = videos.get(neighbours_file, 0) + 1

            # add frame neighbours to the list
            neighbours_list.append(Neighbours(timestamp=timestamp, frames=frames))

    # print video ocurrences sorted by number
    # print(sorted(videos.items(), key=lambda kv: kv[1], reverse=True)[:50])

    # print neighbours stats
    print(f'min 10 num_neighbours: {sorted(num_neighbours)[:10]}')

    return neighbours_list


class Candidate:

    def __init__(self, video: str, index: int, start_time: float, original_start_time: float):
        self.video: str = video
        self.index: int = index

        self.start_time: float = start_time
        self.orig_start_time: float = original_start_time
        self.duration: float = 0

        self.missing: int = 0
        self.found: int = 0
        self.missing_streak: int = 0

    def orig_end_time(self) -> float:
        return self.orig_start_time + self.duration

    def end_time(self) -> float:
        return self.start_time + self.duration

    def find_next(self, neighbours: Neighbours, threshold: int = 0):
        self.index += 1

        low = self.index - threshold
        high = self.index + threshold
        for frame in neighbours.frames:

            # find index in range
            if self.video == frame.video and low <= frame.index <= high:
                self.duration = neighbours.timestamp - self.start_time

                self.missing_streak = 0
                self.found += 1
                return

        self.missing_streak += 1
        self.missing += 1
        return

    def offset(self) -> float:
        return self.orig_start_time - self.start_time

    def offset_diff(self, other: 'Candidate') -> float:
        return abs(self.offset() - other.offset())

    def contains(self, other: 'Candidate'):
        """ checks is the other copy is completely contained in the actual one """
        orig_contained = self.orig_start_time <= other.orig_start_time and other.orig_end_time() <= self.orig_end_time()
        copy_contained = self.start_time <= other.start_time and other.end_time() <= self.end_time()
        return orig_contained and copy_contained

    def distance(self, other: 'Candidate'):
        if self.video != other.video:
            return 1000000

        closest_start = np.clip(self.start_time, other.start_time, other.end_time())
        closest_end = np.clip(self.end_time(), other.start_time, other.end_time())

        return min(abs(self.start_time - closest_start), abs(self.end_time() - closest_end))

    def combine(self, other: 'Candidate'):
        if self.contains(other) or other.contains(self):
            return

        offset_self = self.offset()
        offset_other = other.offset()

        self.duration = max(self.end_time(), other.end_time()) - min(self.start_time, other.start_time)
        self.start_time = min(self.start_time, other.start_time)
        other.duration = self.duration
        other.start_time = self.start_time

        # each copy keeps it's own offset
        self.orig_start_time = self.start_time + offset_self
        other.orig_start_time = other.start_time + offset_other

        return

    def score(self) -> float:
        if self.found < 3:
            return 0
        return self.found / max(1, self.missing - self.missing_streak)

    def __str__(self):
        return f'{self.start_time:.1f} {self.video} {self.orig_start_time:.1f} {self.duration:.1f} {self.score():.1f}'

    def __repr__(self):
        return str(self)


def find_copies(
        video_name: str,
        selector: KeyframeSelector,
        extractor: FeatureExtractor,
        index: SearchIndex,
        max_missing_streak: int = 7,
        min_duration: float = 1,
        max_offset: float = 0,
        force: bool = False,
):
    """
    Searches for copies in the given video.

    :param video_name: the name of the target video.
    :param selector: the keyframe selector used during feature extraction.
    :param extractor: the feature extractor used during feature extraction.
    :param index: the index used during similarity search.

    :param max_missing_streak: .
    :param min_duration: .
    :param max_offset: .

    :param force: whether to re-do the detection or not
    """

    results_dir = get_results_dir(selector=selector, extractor=extractor, index=index)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    results_path = f'{results_dir}/{video_name}.txt'
    if os.path.isfile(results_path) and not force:
        print(f'skipping video {video_name}')
        return

    # read neghbours
    neighbours_path = get_neighbours_dir(selector=selector, extractor=extractor, index=index)
    neighbours_list = read_neighbours(f'{neighbours_path}/{video_name}.txt')

    print(f'searching for copies in {video_name} using {neighbours_path.split("/")[-1]}')
    t0 = time.time()

    # copies candidates
    candidates = []
    copies = []

    threshold = 1

    for neighbours in neighbours_list:
        closed_copies = []

        # search sequences
        for cand in candidates:
            cand.find_next(neighbours, threshold=threshold)

            # determinar fin de clip.
            if cand.missing_streak >= max_missing_streak:
                closed_copies.append(cand)

        # delete closed copies
        for closed in closed_copies:
            candidates.remove(closed)

            # if the candidate is good, add to the list
            if closed.score() >= 1 and closed.duration > min_duration / 2:
                copies.append(closed)

        # add copies candidates
        for frame in neighbours.frames:

            # skip frames from the video being analized
            if frame.video == video_name:
                continue

            # check that it's not the current frame for any of the existing candidates
            low = frame.index - threshold
            high = frame.index + threshold
            for cand in candidates:
                if cand.video == frame.video and low <= cand.index <= high:
                    continue

            candidates.append(
                Candidate(
                    video=frame.video,
                    index=frame.index,
                    start_time=neighbours.timestamp,
                    original_start_time=frame.timestamp,
                )
            )

    # check reamining candidates
    for cand in candidates:
        if cand.score() >= 1 and cand.duration > min_duration / 2:
            copies.append(cand)

    print(f'\t{len(copies)} candidates detected in {int(time.time() - t0)} seconds')

    # separate copies by video
    video_copies = {}

    for copy in copies:
        video = copy.video

        if video not in video_copies:
            video_copies[video] = []
        video_copies[video].append(copy)

    # first remove repeated copies and sort by start
    for video in video_copies:
        current_copies = video_copies[video]
        filtered = []

        for i in range(len(current_copies)):
            copy_a = current_copies[i]

            add = True
            for j in range(len(current_copies)):
                if j == i:
                    continue

                if current_copies[j].contains(copy_a):
                    add = False
                    break

            if add:
                filtered.append(copy_a)

        video_copies[video] = sorted(filtered, key=lambda c: c.start_time)

    print(f'\t{num_copies(video_copies)} copies kept after removing contained videos')

    # combine copies when possible
    for video in video_copies:
        current_copies = video_copies[video]
        for i in range(len(current_copies)):
            copy_a = current_copies[i]

            for j in range(i + 1, len(current_copies)):
                copy_b = current_copies[j]

                if copy_a.distance(copy_b) < 3 and copy_a.offset_diff(copy_b) < max_offset:
                    copy_a.combine(copy_b)

    # detect overlapped copies
    for video in video_copies:
        current_copies = video_copies[video]
        repeated = set()

        for i in range(len(current_copies)):
            copy_a = current_copies[i]

            for j in range(i + 1, len(current_copies)):
                copy_b = current_copies[j]

                # if there's overlapping, filter video
                if copy_a.distance(copy_b) <= 0.1 and copy_a.offset_diff(copy_b) < 10:
                    if copy_a.duration >= copy_b.duration:
                        repeated.add(copy_b)
                    else:
                        repeated.add(copy_a)

        for copy in repeated:
            current_copies.remove(copy)

    print(f'\t{num_copies(video_copies)} copies kept after combining and removing overlapped videos')

    # delete short copies
    for video in video_copies:

        filtered_copies = []
        for copy in video_copies[video]:
            if copy.duration > min_duration:
                filtered_copies.append(copy)

        video_copies[video] = filtered_copies

    print(f'\t{num_copies(video_copies)} copies kept after removing short videos')

    # write to file
    log = open(results_path, 'w')
    for video in sorted(video_copies.keys()):
        for copy in video_copies[video]:
            log.write(f'{copy}\n')

    log.close()
    print(f'found {num_copies(video_copies)} copies in {int(time.time() - t0)} seconds\n')
    return


def num_copies(video_copies: dict):
    n = 0
    for video in video_copies:
        n += len(video_copies[video])
    return n


def main():
    max_missing_streak = 6
    min_duration = 5
    max_offset = 2

    videos = ['119-120', '143', '178', '215', '385', '417', ]

    selectors = [
        FPSReductionKS(n=3),
        MaxHistDiffKS(frames_per_window=1),
    ]
    extractors = [
        ColorLayoutFE(),
        # AutoEncoderFE(dummy=True),
    ]

    indexes = [
        LinearIndex(dummy=True),
        KDTreeIndex(dummy=True, trees=5),
        SGHIndex(dummy=True, projections=10),
        LSHIndex(dummy=True, projections=5, tables=10),  # color layout
        # LSHIndex(dummy=True, projections=3, tables=10),  # auto encoder
    ]

    for selector in selectors:
        for extractor in extractors:
            for index in indexes:
                for video_name in videos:
                    find_copies(
                        video_name=video_name,
                        selector=selector,
                        extractor=extractor,
                        index=index,
                        max_missing_streak=max_missing_streak,
                        min_duration=min_duration,
                        max_offset=max_offset,
                        force=True,
                    )
    return


if __name__ == '__main__':
    main()
