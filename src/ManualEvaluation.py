import os

import cv2
import numpy as np

from features import AutoEncoderFE, ColorLayoutFE, FeatureExtractor
from indexes import LSHIndex, SGHIndex, LinearIndex, KDTreeIndex, SearchIndex
from keyframes import KeyframeSelector, MaxHistDiffKS, FPSReductionKS
from utils.files import get_results_dir, get_orig_videos_dir, get_ground_truth_dir, log_persistent


def timestamp(time):
    return f'{int(time / 60):02}:{int(time % 60):02}'


class Duplicate:
    def __init__(
            self,
            copy_video: str,
            copy_start: float,
            orig_video: str,
            orig_start: float,
            duration: float,
            score: float = 0,
    ):
        self.copy_video = copy_video
        self.copy_start = copy_start

        self.orig_video = orig_video
        self.orig_start = orig_start

        self.duration = duration
        self.score = score

        self.correct = False

    def video_timestamp(self) -> str:
        start = self.copy_start
        end = self.copy_start + self.duration
        return f'{self.copy_video} {timestamp(start)}-{timestamp(end)}'

    def original_timestamp(self) -> str:
        start = self.orig_start
        end = self.orig_start + self.duration
        return f'{self.orig_video} {timestamp(start)}-{timestamp(end)}'

    def orig_end_time(self) -> float:
        return self.orig_start + self.duration

    def end_time(self) -> float:
        return self.copy_start + self.duration

    def offset(self) -> float:
        return self.orig_start - self.copy_start

    def offset_diff(self, other: 'Duplicate') -> float:
        return abs(self.offset() - other.offset())

    def contains(self, other: 'Duplicate'):
        """ checks is the other copy is completely contained in the actual one """
        orig_contained = self.orig_start <= other.orig_start and other.orig_end_time() <= self.orig_end_time()
        copy_contained = self.copy_start <= other.copy_start and other.end_time() <= self.end_time()
        return orig_contained and copy_contained

    def distance(self, other: 'Duplicate'):
        if self.orig_video != other.orig_video:
            return 1000000

        closest_start = np.clip(self.copy_start, other.copy_start, other.end_time())
        closest_end = np.clip(self.end_time(), other.copy_start, other.end_time())

        return min(abs(self.copy_start - closest_start), abs(self.end_time() - closest_end))

    def intersection(self, other: 'Duplicate'):
        if self.orig_video != other.orig_video:
            return -1

        inter = min(self.end_time(), other.end_time()) - max(self.copy_start, other.copy_start)
        union = max(self.end_time(), other.end_time()) - min(self.copy_start, other.copy_start)

        if union == 0:
            return 0

        return inter / union

    def combine(self, other: 'Duplicate'):
        if self.contains(other) or other.contains(self):
            return

        offset_self = self.offset()
        offset_other = other.offset()

        self.duration = max(self.end_time(), other.end_time()) - min(self.copy_start, other.copy_start)
        self.copy_start = min(self.copy_start, other.copy_start)
        other.duration = self.duration
        other.copy_start = self.copy_start

        # each copy keeps it's own offset
        self.orig_start = self.copy_start + offset_self
        other.orig_start = other.copy_start + offset_other

        return

    def __str__(self) -> str:
        return f'{self.copy_video} {self.copy_start:.2f} {self.orig_video} {self.orig_start:.2f} {self.duration:.2f}'

    def __repr__(self) -> str:
        return str(self)


def compare_videos(duplicate: Duplicate, folder: str = ''):
    # open videos
    videos_path = get_orig_videos_dir()
    copy_video = cv2.VideoCapture(f'{videos_path}/{duplicate.copy_video}.mp4')
    orig_video = cv2.VideoCapture(f'{videos_path}/{duplicate.orig_video}.mp4')

    # add duplicate information
    text = f'{duplicate.video_timestamp()} -> {duplicate.original_timestamp()} ({duplicate.score:.1f})'
    font = cv2.FONT_HERSHEY_COMPLEX
    scale = 1
    thickness = 3
    width, heigth = cv2.getTextSize(text, font, scale, thickness)[0]

    # move pointers to start of videos
    copy_video.set(cv2.CAP_PROP_POS_MSEC, duplicate.copy_start * 1000)
    orig_video.set(cv2.CAP_PROP_POS_MSEC, duplicate.orig_start * 1000)

    # reproduction variables
    img = None
    time = 0
    fps = copy_video.get(cv2.CAP_PROP_FPS)
    frame_duration = 1 / fps
    frame_duration_milis = int(frame_duration * 1000 / 2)  # double speed
    pause = False

    while True:
        finished = not (time < duplicate.duration and copy_video.isOpened() and orig_video.isOpened())

        # retrieve next frames
        if not pause and not finished:
            _, frame1 = copy_video.read()
            _, frame2 = orig_video.read()

            frame1 = cv2.resize(frame1, (640, 358))
            frame2 = cv2.resize(frame2, (640, 358))

            # concatenate frames and add info
            img = cv2.hconcat([frame1, frame2])
            cv2.putText(img, text, (int(640 - width / 2) - 1, heigth), font, scale, (0, 0, 0), thickness=thickness + 4)
            cv2.putText(img, text, (int(640 - width / 2), heigth), font, scale, (255, 255, 255), thickness=thickness)

            time += frame_duration

        # time step and user input
        cv2.imshow(f'resultados {folder}', img)
        res = cv2.waitKey(frame_duration_milis)

        # return value
        if res & 0xff == ord('y'):
            return True
        elif res & 0xff == ord('n'):
            return False

        # reset video
        elif res & 0xff == ord('r'):
            time = 0
            copy_video.set(cv2.CAP_PROP_POS_MSEC, duplicate.copy_start * 1000)
            orig_video.set(cv2.CAP_PROP_POS_MSEC, duplicate.orig_start * 1000)

        # pause video (p or space)
        elif res & 0xff == ord('p') or res == 32:
            pause = not pause


def read_results(results_path, video_name):
    duplicates = []
    with open(f'{results_path}/{video_name}.txt', 'r') as results:
        for line in results:
            video_start, orig_video, orig_start, duration, score = line.split(' ')

            duplicates.append(
                Duplicate(
                    copy_video=video_name,
                    copy_start=float(video_start),
                    orig_video=orig_video,
                    orig_start=float(orig_start),
                    duration=float(duration),
                    score=float(score),
                )
            )

    return duplicates


def evaluate_duplicates(
        video_name: str,
        selector: KeyframeSelector,
        extractor: FeatureExtractor,
        index: SearchIndex,
):
    results_path = get_results_dir(selector=selector, extractor=extractor, index=index)
    folder = results_path.split('/')[-1]

    # check if it was already evaluated before
    if os.path.isfile(f'{results_path}/results.txt'):
        with open(f'{results_path}/results.txt', 'r') as evaluated:
            for line in evaluated:
                video, prec = line.strip().split('\t')

                if video == video_name:
                    res = input(f'video {video} already evaluated in {folder} with precision {prec}. skip? ')
                    if res.lower().strip() == 'y':
                        return

                    break

    # read duplicates
    duplicates = read_results(results_path, video_name)
    correct = 0
    total = len(duplicates)

    print(f'evaluating {total} duplicates')

    ground_truth_path = get_ground_truth_dir()
    if not os.path.isdir(ground_truth_path):
        os.makedirs(ground_truth_path)

    for duplicate in duplicates:
        if compare_videos(duplicate, folder):
            correct += 1
            log_persistent(f'{duplicate}\n', f'{ground_truth_path}/{video_name}.txt')

    cv2.destroyAllWindows()
    precision = correct / total * 100
    print(f'precision {video_name} ({folder}): {precision:.1f}%')
    log_persistent(f'{video_name}\t{precision:.1f}\n', f'{results_path}/results.txt')
    return


def main():
    videos = ['178', '119-120', ]  # '417', '143', '215', '385',

    selectors = [
        FPSReductionKS(n=3),
        MaxHistDiffKS(frames_per_window=1),
    ]
    extractors = [
        ColorLayoutFE(),
        AutoEncoderFE(dummy=True),
    ]

    indexes = [
        LinearIndex(dummy=True),
        KDTreeIndex(dummy=True, trees=5),
        SGHIndex(dummy=True, projections=10),
        LSHIndex(dummy=True, projections=5, tables=10),  # color layout
        LSHIndex(dummy=True, projections=3, tables=10),  # auto encoder
    ]

    datasets = [
        (selectors[0], extractors[0], indexes[0],),  # linear
        # (selectors[0], extractors[0], indexes[1],),  # kdtree
        # (selectors[0], extractors[0], indexes[2],),  # sgh
        # (selectors[0], extractors[0], indexes[3],),  # lsh

        (selectors[0], extractors[1], indexes[0],),  # linear
        (selectors[0], extractors[1], indexes[1],),  # kdtree
        # (selectors[0], extractors[1], indexes[2],),  # sgh
        # (selectors[0], extractors[1], indexes[4],),  # lsh

        (selectors[1], extractors[0], indexes[1],),  # kdtree
        (selectors[1], extractors[0], indexes[2],),  # sgh
        (selectors[1], extractors[0], indexes[3],),  # lsh

        (selectors[1], extractors[1], indexes[1],),  # kdtree
        # (selectors[1], extractors[1], indexes[2],),  # sgh
        # (selectors[1], extractors[1], indexes[4],),  # lsh
    ]

    for video in videos:
        for selector, extractor, index in datasets:
            evaluate_duplicates(
                video_name=video,
                selector=selector,
                extractor=extractor,
                index=index,
            )
    return


if __name__ == '__main__':
    main()
