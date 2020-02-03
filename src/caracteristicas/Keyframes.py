import time
from typing import Generator, Tuple

import cv2
import numpy


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

    res = numpy.array(res).flatten()
    if norm:
        res = normalize(res)

    return res


def cityblock_dist(hist1, hist2):
    return numpy.linalg.norm(hist1 - hist2, ord=1)


def euclidean_dist(hist1, hist2):
    return numpy.linalg.norm(hist1 - hist2)


def normalize(v):
    norm = numpy.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


KeyframeGeneratorType = Generator[Tuple[float, numpy.ndarray], None, None]


def n_frames_per_fps(file: str, n: int = 6) -> KeyframeGeneratorType:
    video = cv2.VideoCapture(file)

    fps = video.get(cv2.CAP_PROP_FPS)  # frames por segundo (para calcular tiempo)
    salto_frames = round(fps / n)  # frames a saltar

    processed = 0  # actual frame index
    t0 = time.time()

    while video.grab():

        processed += 1
        if processed % 1000 == 0:
            print(f'processed {processed} frames in {time.time() - t0:.1f} secs')

        # retrieve n frames per second
        if processed % salto_frames != 0:
            continue

        # retrieve frame
        retval, frame = video.retrieve()
        if not retval:
            continue

        yield processed / fps, frame

    video.release()
    return


def window_max_diff(file: str, frames_per_window: int = 2) -> KeyframeGeneratorType:
    video = cv2.VideoCapture(file)
    fps = video.get(cv2.CAP_PROP_FPS)

    window_size = round(fps)
    window_step = round(window_size * 0.5)

    frames = []
    keyframes = []
    keyframe_indexes = []
    hists = []
    hist_diffs = []

    t0 = time.time()
    processed = 0  # actual frame index

    run = True
    while run:

        # retrieve frame
        ret, frame = video.read()
        if not ret:
            run = False
        else:
            frames.append(frame)

        processed += 1
        if processed % 1000 == 0:
            print(f'processed {processed} frames in {time.time() - t0:.1f} secs')

        # extract keyframes
        if len(frames) == window_size or not run:

            # calculate histograms
            for i in range(len(hists), len(frames)):
                hists.append(calc_hist(image=frames[i]))

            # calculate histogram differences
            for i in range(len(hist_diffs), len(frames)):
                hist_diffs.append(cityblock_dist(hists[i], hists[i - 1]))

            # sort indexes
            ordered_indexes = sorted(
                range(len(hist_diffs)),
                key=lambda x: hist_diffs[x],
                reverse=True
            )

            # save keyframes
            for i in ordered_indexes[:frames_per_window]:
                index = processed - window_size + i

                if index not in keyframe_indexes:
                    cv2.putText(frames[i], f'{index}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=10)
                    cv2.putText(frames[i], f'{index}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),
                                thickness=3)

                    keyframes.append([index, frames[i]])
                    keyframe_indexes.append(index)

            frames = frames[window_step:]
            hists = hists[window_step:]
            hist_diffs = hist_diffs[window_step:]

            # yield current sorted keyframes
            keyframes = sorted(keyframes, key=lambda x: x[0])
            sent = 0
            for index, keyframe in keyframes:
                if index < processed - (window_size - window_step):
                    yield index / fps, keyframe
                    sent += 1
                else:
                    break

            keyframes = keyframes[sent:]

    video.release()
    return


def threshold_diff(file: str, threshold: float = 1.3) -> KeyframeGeneratorType:
    video = cv2.VideoCapture(file)
    fps = video.get(cv2.CAP_PROP_FPS)

    t0 = time.time()
    processed = 1  # actual frame index

    _, frame = video.read()
    cv2.putText(frame, '1', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=10)
    cv2.putText(frame, '1', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), thickness=3)

    yield processed / fps, frame
    last_hist = calc_hist(image=frame, norm=True)

    while video.grab():

        # retrieve frame
        ret, frame = video.retrieve()
        if not ret:
            continue

        processed += 1
        if processed % 1000 == 0:
            print(f'processed {processed} frames in {time.time() - t0:.1f} secs')

        hist = calc_hist(image=frame)
        if cityblock_dist(hist, last_hist) > threshold:
            cv2.putText(frame, f'{processed}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=10)
            cv2.putText(frame, f'{processed}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), thickness=3)

            yield processed / fps, frame
            last_hist = hist

            # skip next frame
            video.read()
            processed += 1
            if processed % 1000 == 0:
                print(f'processed {processed} frames in {time.time() - t0:.1f} secs')

    video.release()
    return


if __name__ == '__main__':
    cap = '003'
    keyframes2 = [[i, frame] for i, frame in window_max_diff(f'../../videos/Shippuden/{cap}.mp4')]
    keyframes1 = [[i, frame] for i, frame in n_frames_per_fps(f'../../videos/Shippuden/{cap}.mp4')]

    print(len(keyframes1), len(keyframes2))

    for i1 in range(len(keyframes1)):
        keyframes1[i1][1] = cv2.resize(keyframes1[i1][1], (593, 336))
    for i1 in range(len(keyframes2)):
        keyframes2[i1][1] = cv2.resize(keyframes2[i1][1], (593, 336))

    # show keyframes
    i1 = 0
    i2 = 0
    while True:
        cv2.imshow(f'Keyframes', cv2.vconcat([keyframes1[i1][1], keyframes2[i2][1]]))

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
