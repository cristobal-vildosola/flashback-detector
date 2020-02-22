import time
from typing import Tuple
from abc import ABC, abstractmethod

import cv2
import numpy as np


def calc_hist(image, bins=(5, 5, 5), grid_size=3, norm=True):
    res = []
    size_x = len(image)
    size_y = len(image[0])

    for i in range(grid_size):
        for j in range(grid_size):
            hist = cv2.calcHist(
                images=[image[
                        size_x * i // grid_size: size_x * (i + 1) // grid_size,
                        size_y * i // grid_size: size_y * (i + 1) // grid_size]],
                channels=[0, 1, 2], mask=None, histSize=bins, ranges=[0, 256, 0, 256, 0, 256])

            res.append(hist.flatten())

    res = np.array(res).flatten()
    if norm:
        res = normalize(res)

    return res


def cityblock_dist(hist1, hist2):
    return np.linalg.norm(hist1 - hist2, ord=1)


def euclidean_dist(hist1, hist2):
    return np.linalg.norm(hist1 - hist2)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class KeyframeSelector(ABC):
    @abstractmethod
    def select_keyframes(self, filename) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class SimpleKS(KeyframeSelector):
    def __init__(self, n: int = 6):
        self.n = n

    def select_keyframes(self, filename) -> Tuple[np.ndarray, np.ndarray]:
        video = cv2.VideoCapture(filename)

        fps = video.get(cv2.CAP_PROP_FPS)
        frames_step = round(fps / self.n)

        frames = []
        timestamps = []

        processed = 0  # actual frame index
        t0 = time.time()

        while video.grab():
            processed += 1

            # skip step frames to obtain n frames per second
            if processed % frames_step != 0:
                continue

            retval, frame = video.retrieve()
            if not retval:
                continue

            frames.append(frame)
            timestamps.append(processed / fps)

        video.release()
        print(f'selected {len(frames)} of {processed} frames in {time.time() - t0:.1f} secs')

        return np.array(frames), np.array(timestamps)

    def name(self) -> str:
        return f'simple_{self.n}'


class MaxHistDiffKS(KeyframeSelector):
    def __init__(self, frames_per_window: int = 2):
        self.frames_per_window = frames_per_window

    def select_keyframes(self, filename) -> Tuple[np.ndarray, np.ndarray]:
        video = cv2.VideoCapture(filename)
        fps = video.get(cv2.CAP_PROP_FPS)

        window_size = round(fps)
        step_size = round(window_size * 0.5)

        frames = []
        hists = []
        hist_diffs = []
        keyframes_index = []
        indexes = set()

        t0 = time.time()
        processed = 0  # actual frame index

        run = True
        while run:
            processed += 1

            ret, frame = video.read()
            if not ret:
                run = False
            else:
                frames.append(frame)

            # extract keyframes when window is full
            current_size = len(frames)
            if current_size == window_size or not run:

                calculated = len(hists)

                # calculate histograms
                for i in range(calculated, current_size):
                    hists.append(calc_hist(image=frames[i]))

                # calculate histogram differences
                for i in range(calculated, current_size):
                    hist_diffs.append(cityblock_dist(hists[i], hists[i - 1]))

                # sort indexes by diff in descending order
                ordered_indexes = sorted(
                    range(len(hist_diffs)),
                    key=lambda x: hist_diffs[x],
                    reverse=True
                )

                # save top k keyframes
                for i in ordered_indexes[:self.frames_per_window]:
                    index = processed - window_size + i

                    if index not in indexes:
                        indexes.add(index)
                        keyframes_index.append([frames[i], index])

                # advance one step
                frames = frames[step_size:]
                hists = hists[step_size:]
                hist_diffs = hist_diffs[step_size:]

        video.release()

        keyframes_index = sorted(keyframes_index, key=lambda x: x[1])
        keyframes = [x[0] for x in keyframes_index]
        timestamps = [x[1] / fps for x in keyframes_index]

        print(f'selected {len(keyframes)} of {processed} frames in {time.time() - t0:.1f} secs')

        return np.array(keyframes), np.array(timestamps)

    def name(self) -> str:
        return f'window_{self.frames_per_window}'


class ThresholdHistDiffKS(KeyframeSelector):
    def __init__(self, threshold: float = 1.3):
        self.threshold = threshold

    def select_keyframes(self, filename) -> Tuple[np.ndarray, np.ndarray]:
        video = cv2.VideoCapture(filename)
        fps = video.get(cv2.CAP_PROP_FPS)

        t0 = time.time()
        processed = 0  # actual frame index

        _, frame = video.read()
        last_hist = calc_hist(image=frame, norm=True)

        keyframes = [frame]
        timestamps = [0]

        while video.grab():
            processed += 1

            # retrieve frame
            ret, frame = video.retrieve()
            if not ret:
                continue

            # check difference with last frame
            hist = calc_hist(image=frame)
            if cityblock_dist(hist, last_hist) > self.threshold:
                # append keyframe
                keyframes.append(frame)
                timestamps.append(processed / fps)
                last_hist = hist

                # skip next frame to reduce number of moving frames
                video.read()
                processed += 1

        video.release()
        print(f'selected {len(keyframes)} of {processed} frames in {time.time() - t0:.1f} secs')
        return np.array(keyframes), np.array(timestamps)

    def name(self) -> str:
        return f'thresh_{self.threshold}'.replace('.', ',')


def resize_frames(frames, size=(593, 336)):
    resized_frames = np.zeros((len(frames), size[1], size[0], 3), dtype=frames.dtype)

    for i in range(len(frames)):
        resized_frames[i] = cv2.resize(frames[i], size)
    return resized_frames


def add_timestamp(frames, timestamps):
    for i in range(len(frames)):
        cv2.putText(frames[i], f'{timestamps[i]:.2f}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0),
                    thickness=5)
        cv2.putText(frames[i], f'{timestamps[i]:.2f}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),
                    thickness=2)


if __name__ == '__main__':
    chapter = '../../videos/Shippuden_low/178.mp4'

    selector1 = ThresholdHistDiffKS()
    keyframes1, timestamps1 = selector1.select_keyframes(chapter)

    selector2 = MaxHistDiffKS()
    keyframes2, timestamps2 = selector2.select_keyframes(chapter)

    print(len(keyframes1), len(keyframes2))

    keyframes1 = resize_frames(keyframes1)
    add_timestamp(keyframes1, timestamps1)

    keyframes2 = resize_frames(keyframes2)
    add_timestamp(keyframes2, timestamps2)

    # show keyframes
    i1 = 0
    i2 = 0
    while True:
        cv2.imshow(f'Keyframes', cv2.vconcat([keyframes1[i1], keyframes2[i2]]))

        key = cv2.waitKey(0)
        if key & 0xff == ord('a'):
            i1 = max(i1 - 1, 0)
        elif key & 0xff == ord('d'):
            i1 = min(i1 + 1, len(keyframes1) - 1)
        elif key & 0xff == ord('j'):
            i2 = max(i2 - 1, 0)
        elif key & 0xff == ord('l'):
            i2 = min(i2 + 1, len(keyframes2) - 1)
        elif key == 27 or key == -1:  # esc
            break
