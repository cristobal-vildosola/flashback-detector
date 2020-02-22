import os
import re
import time
from typing import Tuple

import numpy

import keyframes.KeyframeSelector as Keyframes
from features.AutoEncoder import AutoEncoderFE
from features.ColorLayout import ColorLayoutFE
from features.FeatureExtractor import FeatureExtractor

VIDEOS_DIR = '../videos'
FEATURES_DIR = '../video_features'
FEATURES_FILE = 'features'
TAG_FILE = 'tags'


def get_features_path(videos_folder, selector, extractor):
    return f'{FEATURES_DIR}/{videos_folder}/{selector.name()}_{extractor.name()}'


def get_videos_path(videos_folder):
    return f'{VIDEOS_DIR}/{videos_folder}'


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
    log.write(f'{(timestamps[-1]):.0f}\t{duration:.2f}\n')
    log.close()
    return


def read_features(file: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    reads the data for a given video and returns the features and tags separated in 2 numpy arrays.

    :param file: the file containing the features
    """

    datos = numpy.load(file)
    features = datos[:, 1:]

    # generar tags
    video_name = re.split('[/.]', file)[-2]
    tags = []
    for i in range(datos.shape[0]):
        tags.append(f'{video_name} # {datos[i][0]} # {i + 1}')

    tags = numpy.array(tags)
    return tags, features.astype('f4')


def group_features(
        videos_folder: str,
        selector: Keyframes.KeyframeSelector,
        extractor: FeatureExtractor,
        force: bool = False
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Groups all the features and tags in a directory and saves them in a file each.

    :param videos_folder: the directory containing the videos.
    :param selector: .
    :param extractor: .
    :param force: when True, groups features even if it was done previously.
    """
    # full path to the features directory
    directory = get_features_path(videos_folder=videos_folder, selector=selector, extractor=extractor)

    # reload files if grouping was already done
    if os.path.isfile(f'{directory}/{FEATURES_FILE}.npy') \
            and os.path.isfile(f'{directory}/{TAG_FILE}.npy') \
            and not force:
        all_features = numpy.load(f'{directory}/{FEATURES_FILE}.npy')
        all_tags = numpy.load(f'{directory}/{TAG_FILE}.npy')

        return all_tags, all_features

    # obtain all files in the directory
    files = os.listdir(directory)

    all_tags = numpy.empty(0, dtype=numpy.str)
    all_features = numpy.empty((0, extractor.size()), dtype='f4')

    # reads all the features files and groups them in one
    i = 0
    for file in files:
        if not file.endswith('.npy') or file == f'{FEATURES_FILE}.npy' or file == f'{TAG_FILE}.npy':
            continue

        # leer características y juntar con los arreglos.
        tags, features = read_features(f'{directory}/{file}')
        all_tags = numpy.concatenate((all_tags, tags))
        all_features = numpy.concatenate((all_features, features))

        i += 1
        print(f'{all_features.shape[0]:,d} feats read in {i} file')

    # save files
    numpy.save(f'{directory}/{FEATURES_FILE}.npy', all_features)
    numpy.save(f'{directory}/{TAG_FILE}.npy', all_tags)

    return all_tags, all_features


def main():
    videos_folder = 'Shippuden_low'
    selector = Keyframes.SimpleKS(n=6)
    extractor = ColorLayoutFE(size=8)
    force = True

    extract_features_directory(
        videos_folder=videos_folder,
        selector=selector,
        extractor=extractor,
        force=force)
    return


if __name__ == '__main__':
    main()
