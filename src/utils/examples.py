import cv2


def gray_rescale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    decriptor = cv2.resize(gray, dsize=(8, 8), interpolation=cv2.INTER_AREA)
    decriptor = cv2.resize(decriptor, dsize=(300, 300), interpolation=cv2.INTER_NEAREST)

    comp = cv2.hconcat([image, decriptor])
    cv2.imshow('', comp)
    cv2.waitKey(0)
    return


def rescale(image):
    decriptor = cv2.resize(image, dsize=(8, 8), interpolation=cv2.INTER_AREA)
    decriptor = cv2.resize(decriptor, dsize=(300, 300), interpolation=cv2.INTER_NEAREST)

    comp = cv2.hconcat([image, decriptor])
    cv2.imshow('', comp)
    cv2.waitKey(0)
    return


def box_blur(image):
    blurred = cv2.blur(image, ksize=(5, 5))

    comp = cv2.hconcat([image, blurred])
    cv2.imshow('', comp)
    cv2.waitKey(0)
    return


def gauss_blur(image):
    blurred = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=1)

    comp = cv2.hconcat([image, blurred])
    cv2.imshow('', comp)
    cv2.waitKey(0)
    return


def main():
    image = cv2.imread('ejemplo.png')
    image = cv2.resize(image, dsize=(600, 600))

    # rescale(image)
    # gray_rescale(image)
    box_blur(image)
    # gauss_blur(image)
    return


if __name__ == '__main__':
    main()
