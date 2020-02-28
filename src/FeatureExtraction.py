import os
import re
import time

import numpy

import keyframes.KeyframeSelector as Keyframes
from features.AutoEncoder import AutoEncoderFE
from features.ColorLayout import ColorLayoutFE
from features.FeatureExtractor import FeatureExtractor
from utils.files import get_features_path, get_videos_path, group_features


def extract_features_directory(
        videos_folder: str,
        selector: Keyframes.KeyframeSelector = Keyframes.SimpleKS(),
        extractor: FeatureExtractor = ColorLayoutFE(),
        force=False
):
    """
    Extracts features for the all the videos in the directory and saves them in a new directory obtained using
    get_features_path.

    :param videos_folder: the directory containing the videos.
    :param selector: .
    :param extractor: .
    :param force: when True, calculates features even if it was done previously.
    """

    # create directory when necessary
    feats_path = get_features_path(videos_folder, selector=selector, extractor=extractor)
    if not os.path.isdir(feats_path):
        os.makedirs(feats_path)

    # create log file
    log_path = f'{feats_path}/log.txt'
    if not os.path.isfile(log_path) or force:
        open(log_path, 'w').close()

    # obtain all files in the directory
    videos_path = get_videos_path(videos_folder)
    videos = os.listdir(videos_path)

    # extract features from each video
    for video in videos:
        if video.endswith('.mp4'):
            extract_features(
                file_path=f'{videos_path}/{video}', save_dir=feats_path,
                selector=selector, extractor=extractor, force=force)

    # group all features into 2 files (features and tags)
    group_features(
        videos_folder=videos_folder,
        selector=selector,
        extractor=extractor,
        force=force)

    return


def extract_features(
        file_path: str,
        save_dir: str,
        selector: Keyframes.KeyframeSelector = Keyframes.SimpleKS(),
        extractor: FeatureExtractor = ColorLayoutFE(),
        force=False
):
    """
    Extracts features for the video and saves them in the given dir.

    :param file_path: video path.
    :param save_dir: directory to save the features.
    :param selector: .
    :param extractor: .
    :param force: when True, calculates features even if it was done previously.
    """

    video_name = re.split('[/.]', file_path)[-2]
    save_path = f'{save_dir}/{video_name}.npy'

    # skip already processed videos
    if not force and os.path.isfile(save_path):
        print(f'Skipping video {video_name}')
        return

    print(f'Extracting features from video {video_name}')

    # obtain keyframes
    keyframes, timestamps = selector.select_keyframes(file_path)

    # measure time
    t0 = time.time()

    # extract features and combine with timestamps
    features = extractor.extract_features(keyframes)
    features = numpy.insert(features, 0, values=timestamps, axis=1)
    features = features.astype('f4')

    # save feats
    numpy.save(save_path, numpy.array(features))

    duration = time.time() - t0
    print(f'feature extraction for {len(timestamps)} frames took {duration:.2f} seconds\n')

    # log time required
    log = open(f'{save_dir}/log.txt', 'a')
    log.write(f'{len(timestamps):.0f}\t{duration:.2f}\n')
    log.close()
    return


def main():
    videos_folder = 'Shippuden_low'

    autoencoder = AutoEncoderFE.load_autoencoder(name='features/model')
    extract_features_directory(
        videos_folder=videos_folder,
        selector=Keyframes.SimpleKS(n=6),
        extractor=autoencoder)
    extract_features_directory(
        videos_folder=videos_folder,
        selector=Keyframes.MaxHistDiffKS(frames_per_window=2),
        extractor=autoencoder)
    extract_features_directory(
        videos_folder=videos_folder,
        selector=Keyframes.ThresholdHistDiffKS(threshold=1.3),
        extractor=autoencoder)

    color_layout = ColorLayoutFE(size=8)
    extract_features_directory(
        videos_folder=videos_folder,
        selector=Keyframes.SimpleKS(n=6),
        extractor=color_layout)
    extract_features_directory(
        videos_folder=videos_folder,
        selector=Keyframes.MaxHistDiffKS(frames_per_window=2),
        extractor=color_layout)
    extract_features_directory(
        videos_folder=videos_folder,
        selector=Keyframes.ThresholdHistDiffKS(threshold=1.3),
        extractor=color_layout)

    return


if __name__ == '__main__':
    main()
