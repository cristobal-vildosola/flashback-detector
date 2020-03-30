from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np


class KeyframeSelector(ABC):
    @abstractmethod
    def select_keyframes(self, filename) -> Tuple[np.ndarray, np.ndarray, int]:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


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
    from keyframes import FPSReductionKS, MaxHistDiffKS

    chapter = '../../videos/Shippuden_low/178.mp4'

    selector1 = FPSReductionKS()
    keyframes1, timestamps1, _ = selector1.select_keyframes(chapter)

    selector2 = MaxHistDiffKS(frames_per_window=1)
    keyframes2, timestamps2, _ = selector2.select_keyframes(chapter)

    print(len(keyframes1), len(keyframes2), int(timestamps1[-1]))

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
