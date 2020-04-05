import os

import cv2


def count_frames(folder='../../videos/Shippuden_low'):
    frames = 0
    videos = os.listdir(folder)
    for video in videos:
        if video.endswith('.mp4'):
            video = cv2.VideoCapture(f'{folder}/{video}')
            frames += video.get(cv2.CAP_PROP_FRAME_COUNT)
            video.release()

    return frames


def show_fps(folder='../../videos/Shippuden_low'):
    videos = os.listdir(folder)
    for video_name in videos:
        if video_name.endswith('.mp4'):

            video = cv2.VideoCapture(f'{folder}/{video_name}')
            print(f'{video_name} {video.get(cv2.CAP_PROP_FPS)}')
            video.release()

    return


if __name__ == '__main__':
    show_fps('../../videos/Shippuden_original')
    # print(f'{count_frames():.0f}')
