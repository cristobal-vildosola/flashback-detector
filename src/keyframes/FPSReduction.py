from typing import Tuple

import cv2
import numpy as np

from keyframes import KeyframeSelector


class FPSReductionKS(KeyframeSelector):
    def __init__(self, n: int = 3):
        self.n = n

    def select_keyframes(self, filename) -> Tuple[np.ndarray, np.ndarray, int]:
        video = cv2.VideoCapture(filename)

        fps = video.get(cv2.CAP_PROP_FPS)
        frames_step = round(fps / self.n)

        frames = []
        timestamps = []

        processed = 0  # actual frame index

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

        return np.array(frames), np.array(timestamps), processed

    def name(self) -> str:
        return f'reduction_{self.n}'
