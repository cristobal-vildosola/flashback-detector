import os
import time

import cv2


def resize_video(file, output, size):
    t0 = time.time()
    os.system(f'ffmpeg -n -i {file} -vf scale={size[0]}:{size[1]} {output}')
    print(f'resizing {file} took {int(time.time() - t0)} seconds')
    return


def resize_videos(folder='../../videos/Shippuden_original', output_folder='../../videos/resized', size=(100, 100)):
    # create folder
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # resize all videos
    videos = os.listdir(folder)
    for video in videos:
        if video.endswith('.mp4'):
            resize_video(file=f'{folder}/{video}', output=f'{output_folder}/{video}', size=size)

    return


def count_frames(folder='../../videos/Shippuden_low'):
    frames = 0
    videos = os.listdir(folder)
    for video in videos:
        if video.endswith('.mp4'):
            video = cv2.VideoCapture(f'{folder}/{video}')
            frames += video.get(cv2.CAP_PROP_FRAME_COUNT)
            video.release()

    return frames


if __name__ == '__main__':
    print(f'{count_frames():.0f}')
    # resize_videos()
