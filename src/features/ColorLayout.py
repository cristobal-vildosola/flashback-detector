import cv2
import numpy as np

from features.FeatureExtractor import FeatureExtractor


class ColorLayoutFE(FeatureExtractor):
    def __init__(self, size: int = 8):
        self.size = size

    def extract_features(self, data: np.ndarray):
        n = len(data)
        features = np.zeros((n, self.descriptor_size()))

        for i in range(n):
            features[i] = color_layout_descriptor(data[i], self.size)

        return features

    def descriptor_size(self) -> int:
        return self.size ** 2 * 3

    def name(self) -> str:
        return f'CL_{self.size}'


def color_layout_descriptor(img: np.ndarray, size: int = 8):
    """
    extracts the Color Layout descriptor for the given image

    :param img: the image to process
    :param size: size of the descriptor.
    """
    # resize to desired size and transform color
    resized = cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_AREA)
    y, cr, cb = cv2.split(cv2.cvtColor(np.array(resized, dtype=np.uint8), cv2.COLOR_BGR2YCR_CB))

    # calculate discrete coscine transform
    dct_y = cv2.dct(np.float32(y))
    dct_cb = cv2.dct(np.float32(cb))
    dct_cr = cv2.dct(np.float32(cr))

    dct_y_zigzag = []
    dct_cb_zigzag = []
    dct_cr_zigzag = []

    flipped_dct_y = np.fliplr(dct_y)
    flipped_dct_cb = np.fliplr(dct_cb)
    flipped_dct_cr = np.fliplr(dct_cr)

    # traverse matrixes in diagonal order
    flip = True
    for i in range(size * 2 - 1):
        k_diag = size - 1 - i
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
