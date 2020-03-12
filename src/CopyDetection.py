import os
import time
from typing import List

import keyframes.KeyframeSelector as Keyframes
from features.ColorLayout import ColorLayoutFE
from features.FeatureExtractor import FeatureExtractor
from utils.files import get_neighbours_path, get_results_path


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
    videos = dict()

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

                # count video ocurrences
                videos[neighbours_file] = videos.get(neighbours_file, 0) + 1

            # add frame neighbours to the list
            neighbours_list.append(Neighbours(timestamp=timestamp, frames=frames))

    # print video ocurrences sorted by number
    print(sorted(videos.items(), key=lambda kv: kv[1], reverse=True))

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

    def find_next(self, neighbours: Neighbours, threshold: int = 0):
        self.index += 1

        for frame in neighbours.frames:

            # find index in range
            if self.video == frame.video and (frame.index - threshold) <= self.index <= (frame.index + threshold):
                self.duration = neighbours.timestamp - self.orig_start_time

                self.missing_streak = 0
                self.found += 1
                return

        self.missing_streak += 1
        self.missing += 1
        return

    def orig_end_time(self) -> float:
        return self.orig_start_time + self.duration

    def end_time(self) -> float:
        return self.start_time + self.duration

    def overlapped(self, cand: 'Candidate'):
        if self.orig_start_time <= cand.orig_start_time < self.orig_end_time() or \
                cand.orig_start_time <= self.orig_start_time < cand.orig_end_time():
            return True

        return False

    def combine(self, cand: 'Candidate', max_offset: float):
        if self.orig_start_time <= cand.orig_start_time and \
                self.orig_end_time() >= cand.orig_end_time():
            return

        if cand.orig_start_time <= self.orig_start_time and \
                cand.orig_end_time() >= self.orig_end_time():
            return

        offset = abs((self.orig_start_time - cand.orig_start_time) -
                     (self.start_time - cand.start_time))

        if offset > max_offset:
            return

        self.duration = max(self.end_time(), cand.end_time()) - min(self.start_time, cand.start_time)
        self.orig_start_time = min(self.orig_start_time, cand.orig_start_time)
        self.start_time = min(self.start_time, cand.start_time)
        return

    def score(self) -> float:
        if self.found < 3:
            return 0
        return self.found / max(1, self.missing - self.missing_streak)

    def __str__(self):
        return f'{self.orig_start_time:.2f} {self.duration:.2f} {self.video} {self.start_time:.2f} ({self.score():.2f})'


def find_copies(
        video_name: str,
        videos_folder: str,
        selector: Keyframes.KeyframeSelector,
        extractor: FeatureExtractor,
        max_missing_streak: int = 7,
        minimun_duration: float = 1,
        max_offset: float = 0):
    """
    Searches for copies in the given video.

    :param video_name: the name of the target video.
    :param videos_folder: the directory containing the videos.
    :param selector: the keyframe selector used during feature extraction.
    :param extractor: the feature extractor used during feature extraction.

    :param max_missing_streak: .
    :param minimun_duration: .
    :param max_offset: .
    """

    print(f'searching for copies in {video_name}')
    t0 = time.time()

    # read neghbours
    neighbours_path = get_neighbours_path(videos_folder=videos_folder, selector=selector, extractor=extractor)
    neighbours_list = read_neighbours(f'{neighbours_path}/{video_name}.txt')

    # copies candidates
    candidates = []
    copies = []

    for neighbours in neighbours_list:
        closed_copies = []

        # search sequences
        for cand in candidates:
            cand.find_next(neighbours, threshold=1)

            # determinar fin de clip.
            if cand.missing_streak >= max_missing_streak:
                closed_copies.append(cand)

        # delete closed copies
        for closed in closed_copies:
            candidates.remove(closed)

            # if the candidate is good, add to the list
            if closed.score() >= 1 and closed.duration > minimun_duration / 2:
                copies.append(closed)

        # add copies candidates
        for frame in neighbours.frames:

            # skip frames from the video being analized
            if frame.video == video_name:
                continue

            # check that it's not the current frame for any of the existing candidates
            for copy_a in candidates:
                if copy_a.video == frame.video and copy_a.index == frame.index:
                    continue

            candidates.append(
                Candidate(
                    video=frame.video,
                    index=frame.index,
                    start_time=frame.timestamp,
                    original_start_time=neighbours.timestamp))

    # check reamining candidates
    for cand in candidates:
        if cand.score() >= 1 and closed.duration > minimun_duration / 2:
            copies.append(cand)

    print(f'{len(copies)} copies detected')

    # combine copies when possible
    for i in range(len(copies)):
        copy_a = copies[i]

        for j in range(i + 1, len(copies)):
            copy_b = copies[j]

            if copy_a.video == copy_b.video and copy_a.overlapped(copy_b):
                copy_a.combine(copy_b, max_offset)

    # detect overlapped copies
    overlapped = set()
    for i in range(len(copies)):
        copy_a = copies[i]
        for j in range(i + 1, len(copies)):
            copy_b = copies[j]

            # if there's overlapping keep only the longest copy
            if copy_a.video == copy_b.video and copy_a.overlapped(copy_b):
                if copy_a.duration > copy_b.duration:
                    overlapped.add(copy_b)
                else:
                    overlapped.add(copy_a)

    # delete overlapped copies
    for copy in overlapped:
        copies.remove(copy)

    print(f'{len(copies)} copies kept after combining and deleting overlapped videos')

    # delete short copies
    too_short = []
    for copy in copies:
        if copy.duration < minimun_duration:
            too_short.append(copy)
    for copy in too_short:
        copies.remove(copy)

    print(f'{len(copies)} copies kept after deleting short videos')

    # open log
    results_path = get_results_path(videos_folder=videos_folder, selector=selector, extractor=extractor)
    if not os.path.isdir(results_path):
        os.makedirs(results_path)
    log = open(f'{results_path}/{video_name}.txt', 'w')

    # sort by chapter and write to file
    copies = sorted(copies, key=lambda x: x.video)
    for copy in copies:
        log.write(f'{copy}\n')

    log.close()
    print(f'found {len(copies)} copies in {int(time.time() - t0)} seconds')
    return


def main():
    find_copies(
        video_name='417',
        videos_folder='Shippuden_low',
        selector=Keyframes.SimpleKS(),
        extractor=ColorLayoutFE(),
        max_missing_streak=6,
        minimun_duration=5,
        max_offset=1)
    return


if __name__ == '__main__':
    main()
