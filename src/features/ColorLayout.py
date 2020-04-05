import cv2
import numpy as np

from features.FeatureExtractor import FeatureExtractor


class ColorLayoutFE(FeatureExtractor):

    def extract_features(self, data: np.ndarray):
        n = len(data)
        features = np.zeros((n, self.descriptor_size()), dtype='int8')

        for i in range(n):
            features[i] = color_layout_descriptor(data[i])

        return features

    def descriptor_size(self) -> int:
        return 192

    def name(self) -> str:
        return f'CL'


def color_layout_descriptor(img: np.ndarray):
    """
    extracts the Color Layout descriptor for the given image

    :param img: the image to process
    """
    # resize to desired size and convert color
    resized = cv2.resize(img, dsize=(8, 8), interpolation=cv2.INTER_AREA)
    y, cr, cb = cv2.split(cv2.cvtColor(resized, cv2.COLOR_BGR2YCR_CB))

    # calculate discrete coscine transform
    dct_y = cv2.dct(np.float32(y)).flatten()
    dct_cb = cv2.dct(np.float32(cb)).flatten()
    dct_cr = cv2.dct(np.float32(cr)).flatten()

    # traverse diagonally and quantize
    quant_dct_y = np.zeros(64, dtype='int8')
    quant_dct_cb = np.zeros(64, dtype='int8')
    quant_dct_cr = np.zeros(64, dtype='int8')
    for i in range(64):
        zigzag_i = zigzag_scan[i]
        quant_dct_y[i] = round(dct_y[zigzag_i] / quant_y[zigzag_i])
        quant_dct_cb[i] = round(dct_cb[zigzag_i] / quant_cr[zigzag_i])
        quant_dct_cr[i] = round(dct_cr[zigzag_i] / quant_cr[zigzag_i])

    return np.concatenate([quant_dct_y, quant_dct_cb, quant_dct_cr]).astype('int8')


zigzag_scan = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
]

quant_y = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]

quant_cr = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
]

if __name__ == '__main__':
    image = cv2.imread('../utils/ejemplo.png')
    print(color_layout_descriptor(image))
