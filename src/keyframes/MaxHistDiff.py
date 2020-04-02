from typing import Tuple

import cv2
import numpy as np

from keyframes import KeyframeSelector


class MaxHistDiffKS(KeyframeSelector):
    def __init__(self, frames_per_window: int = 2):
        self.frames_per_window = frames_per_window

    def select_keyframes(self, filename) -> Tuple[np.ndarray, np.ndarray, int]:
        video = cv2.VideoCapture(filename)
        fps = video.get(cv2.CAP_PROP_FPS)

        window_size = round(fps)
        step_size = round(window_size * 0.5)

        frames = []
        hists = []
        hist_diffs = []
        keyframes_index = []
        indexes = set()

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
                for i in range(calculated + 1, current_size):
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

        return np.array(keyframes), np.array(timestamps), processed

    def name(self) -> str:
        return f'window_{self.frames_per_window}'


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


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
