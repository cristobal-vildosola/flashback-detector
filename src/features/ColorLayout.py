from typing import Tuple

import cv2
import numpy as np

from features.FeatureExtractor import FeatureExtractor


class ColorLayoutExtractor(FeatureExtractor):
    def __init__(self, size: Tuple[int, int]):
        self.size = size
        self.descriptor_size = size[0] * size[1] * 3

    # TODO: test
    def extract_features(self, data: np.ndarray):
        n = len(data)
        features = np.zeros((n, self.descriptor_size))

        for i in range(n):
            features[i] = color_layout_descriptor(data[i], self.size)

        return features


def color_layout_descriptor(img: np.ndarray, size: Tuple[int, int] = (8, 8)):
    """
    :param img: la imagen de la cual extraer características.
    :param size: tamaño del descriptor.
    :return: un vector de tamaño (size x 3), el descriptor de la imagen.
    """
    rows = size[0]
    cols = size[1]
    resized = cv2.resize(img, dsize=(cols, rows), interpolation=cv2.INTER_AREA)

    y, cr, cb = cv2.split(cv2.cvtColor(np.array(resized, dtype=np.uint8), cv2.COLOR_BGR2YCR_CB))

    dct_y = cv2.dct(np.float32(y))
    dct_cb = cv2.dct(np.float32(cb))
    dct_cr = cv2.dct(np.float32(cr))

    dct_y_zigzag = []
    dct_cb_zigzag = []
    dct_cr_zigzag = []

    flipped_dct_y = np.fliplr(dct_y)
    flipped_dct_cb = np.fliplr(dct_cb)
    flipped_dct_cr = np.fliplr(dct_cr)

    flip = True
    for i in range(rows + cols - 1):
        k_diag = rows - 1 - i
        diag_y = np.diag(flipped_dct_y, k=k_diag)
        diag_cb = np.diag(flipped_dct_cb, k=k_diag)
        diag_cr = np.diag(flipped_dct_cr, k=k_diag)

        if flip:
            diag_y = diag_y[::-1]
            diag_cb = diag_cb[::-1]
            diag_cr = diag_cr[::-1]

        dct_y_zigzag.append(diag_y)
        dct_cb_zigzag.append(diag_cb)
        dct_cr_zigzag.append(diag_cr)

        flip = not flip

    return np.concatenate([
        np.concatenate(dct_y_zigzag),
        np.concatenate(dct_cb_zigzag),
        np.concatenate(dct_cr_zigzag),
    ])


if __name__ == '__main__':
    image = cv2.imread('../utils/ejemplo.png')
    decriptor = cv2.resize(image, dsize=(8, 8), interpolation=cv2.INTER_AREA)

    image = cv2.resize(image, dsize=(600, 600))
    decriptor = cv2.resize(decriptor, dsize=(600, 600), interpolation=cv2.INTER_NEAREST)

    image = cv2.hconcat([image, decriptor])
    cv2.imshow('', image)
    cv2.waitKey(0)
