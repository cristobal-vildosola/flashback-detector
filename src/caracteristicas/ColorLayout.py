import cv2
import numpy as np


def color_layout_descriptor(img, tamano=(8, 8)):
    """
    :param img: la imagen de la cual extraer características.
    :param tamano: tamaño del descriptor.
    :return: un vector de tamaño x 3, el descriptor de la imagen.
    """
    rows = tamano[0]
    cols = tamano[1]
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

    img = cv2.imread('ejemplo.png')
    resized = cv2.resize(img, dsize=(8, 8), interpolation=cv2.INTER_AREA)

    img = cv2.resize(img, dsize=(600, 600))
    resized = cv2.resize(resized, dsize=(600, 600), interpolation=cv2.INTER_NEAREST)

    img = cv2.hconcat([img, resized])
    cv2.imshow('', img)
    res = cv2.waitKey(0)
