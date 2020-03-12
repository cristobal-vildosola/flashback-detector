import os
import re
from typing import Tuple

import numpy as np

import keyframes.KeyframeSelector as Keyframes
from features.FeatureExtractor import FeatureExtractor

VIDEOS_DIR = '../videos'
FEATURES_DIR = '../videos_features'
NEIGHBOURS_DIR = '../videos_neighbours'
RESULTS_DIR = '../videos_results'

FEATURES_FILE = 'features'
TAG_FILE = 'tags'


def get_videos_path(videos_folder):
    return f'{VIDEOS_DIR}/{videos_folder}'


def get_features_path(videos_folder, selector, extractor):
    return f'{FEATURES_DIR}/{videos_folder}/{selector.name()}_{extractor.name()}'


def get_neighbours_path(videos_folder, selector, extractor):
    return f'{NEIGHBOURS_DIR}/{videos_folder}/{selector.name()}_{extractor.name()}'


def get_results_path(videos_folder, selector, extractor):
    return f'{RESULTS_DIR}/{videos_folder}/{selector.name()}_{extractor.name()}'


def read_features(file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    reads the data for a given video and returns the features and tags separated in 2 numpy arrays.

    :param file: the file containing the features
    """

    data = np.load(file)
    features = data[:, 1:]

    # generar tags
    video_name = re.split('[/.]', file)[-2]
    tags = []
    for i in range(data.shape[0]):
        tags.append(f'{video_name} # {data[i][0]} # {i + 1}')

    tags = np.array(tags)
    return tags, features.astype('f4')


def group_features(
        videos_folder: str,
        selector: Keyframes.KeyframeSelector,
        extractor: FeatureExtractor,
        force: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
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
        print(f'Grouping already done for {directory}')
        all_features = np.load(f'{directory}/{FEATURES_FILE}.npy')
        all_tags = np.load(f'{directory}/{TAG_FILE}.npy')

        return all_tags, all_features

    # obtain all files in the directory
    files = os.listdir(directory)

    all_tags = np.empty(0, dtype=np.str)
    all_features = np.empty((0, extractor.descriptor_size()), dtype='f4')

    # reads all the features files and groups them in one
    i = 0
    for file in files:
        if not file.endswith('.npy') or file == f'{FEATURES_FILE}.npy' or file == f'{TAG_FILE}.npy':
            continue

        # leer características y juntar con los arreglos.
        tags, features = read_features(f'{directory}/{file}')
        all_tags = np.concatenate((all_tags, tags))
        all_features = np.concatenate((all_features, features))

        i += 1
        print(f'{all_features.shape[0]:,d} feats read in {i} file')

    # save files
    np.save(f'{directory}/{FEATURES_FILE}.npy', all_features)
    np.save(f'{directory}/{TAG_FILE}.npy', all_tags)

    return all_tags, all_features
