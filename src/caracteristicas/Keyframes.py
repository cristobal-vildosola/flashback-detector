import time

import cv2
import numpy


def calc_hist(image, bins, grid_size=3):
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

    return numpy.array(res)


def hist_diff(hist1, hist2):
    return numpy.sum(numpy.abs(hist1 - hist2))


def extract_keyframes(video_filename, frames_per_window=2, bins=(5, 5, 5)):
    video = cv2.VideoCapture(video_filename)

    fps = video.get(cv2.CAP_PROP_FPS)
    window_size = round(fps)
    window_step = window_size // 2

    frames = []
    keyframes = []
    keyframe_indexes = []
    hists = []
    hist_diffs = []

    t0 = time.time()
    processed = 0

    while video.isOpened():

        # retreve frame
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)

        processed += 1
        if processed % 1000 == 0:
            print(f'processed {processed} frames in {time.time() - t0:.1f} secs')

        # extract keyframes
        if len(frames) == window_size:

            # calculate histograms
            for i in range(len(hists), len(frames)):
                hists.append(calc_hist(image=frames[i], bins=bins))

            # calculate histogram differences
            for i in range(len(hist_diffs), len(frames)):
                hist_diffs.append(hist_diff(hists[i], hists[i - 1]))

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

    video.release()
    keyframes = sorted(keyframes, key=lambda x: x[0])
    print(len(keyframes))

    # show keyframes
    i = 0
    while True:
        cv2.imshow(f'Keyframes', keyframes[i][1])

        res = cv2.waitKey(0)
        if res & 0xff == ord('d'):
            i = min(i + 1, len(keyframes) - 1)
        elif res & 0xff == ord('a'):
            i = max(i - 1, 0)
        elif res == 27:  # esc
            break

    return


def extract_keyframes_2(video_filename, bins=(5, 5, 5), threshold=20000):
    video = cv2.VideoCapture(video_filename)

    _, frame = video.read()
    cv2.putText(frame, '0', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=10)
    cv2.putText(frame, '0', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), thickness=3)
    keyframes = [frame]
    last_hist = calc_hist(image=frame, bins=bins)

    t0 = time.time()
    processed = 0

    while video.isOpened():

        # retreve frame
        ret, frame = video.read()
        if not ret:
            break

        processed += 1
        if processed % 1000 == 0:
            print(f'processed {processed} frames in {time.time() - t0:.1f} secs')

        hist = calc_hist(image=frame, bins=bins)
        if hist_diff(hist, last_hist) > threshold:

            cv2.putText(frame, f'{processed}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=10)
            cv2.putText(frame, f'{processed}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), thickness=3)
            keyframes.append(frame)
            last_hist = hist

            # skip next frame
            video.read()
            processed += 1

    video.release()
    print(len(keyframes))

    # show keyframes
    i = 0
    while True:
        cv2.imshow(f'Keyframes', keyframes[i])

        res = cv2.waitKey(0)
        if res & 0xff == ord('d'):
            i = min(i + 1, len(keyframes) - 1)
        elif res & 0xff == ord('a'):
            i = max(i - 1, 0)
        elif res == 27:  # esc
            break

    return


if __name__ == '__main__':
    # extract_keyframes('../../videos/Shippuden/003.mp4', bins=(5, 5, 5), frames_per_window=2)
    extract_keyframes_2('../../videos/Shippuden/003.mp4', bins=(5, 5, 5), threshold=70000)
