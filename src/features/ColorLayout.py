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


zigzag_scan = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
]


def color_layout_descriptor(img: np.ndarray):
    """
    extracts the Color Layout descriptor for the given image

    :param img: the image to process
    """
    # resize to desired size and transform color
    resized = cv2.resize(img, dsize=(8, 8), interpolation=cv2.INTER_AREA)
    y, cr, cb = cv2.split(cv2.cvtColor(np.array(resized, dtype=np.uint8), cv2.COLOR_BGR2YCR_CB))

    # calculate discrete coscine transform
    dct_y = cv2.dct(np.float32(y)).flatten()
    dct_cb = cv2.dct(np.float32(cb)).flatten()
    dct_cr = cv2.dct(np.float32(cr)).flatten()

    # traverse diagonally and quantize
    dct_y[0] = quant_ydc(dct_y[0]) >> 1
    dct_cb[0] = quant_cdc(dct_cb[0] / 8)
    dct_cr[0] = quant_cdc(dct_cr[0] / 8)
    for i in range(64):
        dct_y[i] = quant_ac(dct_y[zigzag_scan[i]] / 2) >> 3
        dct_cb[i] = quant_ac(dct_cb[zigzag_scan[i]]) >> 3
        dct_cr[i] = quant_ac(dct_cr[zigzag_scan[i]]) >> 3

    return np.concatenate([dct_y, dct_cb, dct_cr]).astype('int8')


def quant_ydc(i):
    i = int(i)
    if i > 192:
        j = 112 + (i - 192) >> 2
    elif i > 160:
        j = 96 + (i - 160) >> 1
    elif i > 96:
        j = 32 + (i - 96)
    elif i > 64:
        j = 16 + (i - 64) >> 1
    else:
        j = i >> 2

    return int(j)


def quant_cdc(i):
    i = int(i)
    j = 0
    if i > 191:
        j = 63
    elif i > 160:
        j = 56 + (i - 160) >> 2
    elif i > 143:
        j = 48 + (i - 144) >> 1
    elif i > 111:
        j = 16 + (i - 112)
    elif i > 95:
        j = 8 + (i - 96) >> 1
    elif i > 63:
        j = (i - 64) >> 2

    return int(j)


def quant_ac(i):
    i = int(i)
    if i > 255:
        i = 255
    if i < -255:
        i = -256

    if abs(i) > 127:
        j = 64 + abs(i) >> 2
    elif abs(i) > 63:
        j = 32 + abs(i) >> 1
    else:
        j = abs(i)

    if i < 0:
        j = -j

    j += 128
    return int(j)


if __name__ == '__main__':
    image = cv2.imread('../utils/ejemplo.png')
    print(color_layout_descriptor(image))
