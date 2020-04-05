import os
from typing import Tuple

import numpy as np

from features import FeatureExtractor
from indexes import SearchIndex
from keyframes import KeyframeSelector

project_root = 'C:/Users/Cristobal/Documents/U/TrabajoTitulo/proyectoTitulo'

VIDEOS_DIR = f'{project_root}/videos/Shippuden_low'
ORIG_VIDEOS_DIR = f'{project_root}/videos/Shippuden_original'
FEATURES_DIR = f'{project_root}/videos_features'
NEIGHBOURS_DIR = f'{project_root}/videos_neighbours'
RESULTS_DIR = f'{project_root}/videos_results'
GROUND_TRUTH_DIR = f'{project_root}/ground_truth'

FEATURES_FILE = 'features'
TAGS_FILE = 'tags'


def get_videos_dir():
    return f'{VIDEOS_DIR}'


def get_orig_videos_dir():
    return f'{ORIG_VIDEOS_DIR}'


def get_features_dir(selector: KeyframeSelector, extractor: FeatureExtractor):
    return f'{FEATURES_DIR}/{selector.name()}_{extractor.name()}'


def get_neighbours_dir(selector: KeyframeSelector, extractor: FeatureExtractor, index: SearchIndex):
    return f'{NEIGHBOURS_DIR}/{selector.name()}_{extractor.name()}_{index.name()}'


def get_results_dir(selector: KeyframeSelector, extractor: FeatureExtractor, index: SearchIndex):
    return f'{RESULTS_DIR}/{selector.name()}_{extractor.name()}_{index.name()}'


def get_ground_truth_dir():
    return f'{GROUND_TRUTH_DIR}'


def read_features(video_name: str, directory: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    reads the data for a given video and returns the features and tags separated in 2 numpy arrays.

    :param video_name: .
    :param directory: .
    """
    if not os.path.isfile(f'{directory}/{video_name}-feats.npy'):
        raise Exception(f'Missing features file for video {video_name} in {directory}')
    if not os.path.isfile(f'{directory}/{video_name}-tags.npy'):
        raise Exception(f'Missing tags file for video {video_name} in {directory}')

    features = np.load(f'{directory}/{video_name}-feats.npy')
    tags = np.load(f'{directory}/{video_name}-tags.npy')

    return tags, features


def group_features(
        selector: KeyframeSelector,
        extractor: FeatureExtractor,
        force: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Groups all the features and tags in a directory and saves them in a file each.

    :param selector: .
    :param extractor: .
    :param force: when True, groups features even if it was done previously.
    """
    # full path to the features directory
    feats_dir = get_features_dir(selector=selector, extractor=extractor)

    # reload files if grouping was already done
    if os.path.isfile(f'{feats_dir}/{FEATURES_FILE}.npy') \
            and os.path.isfile(f'{feats_dir}/{TAGS_FILE}.npy') \
            and not force:
        print(f'Grouping already done for {feats_dir}')
        all_features = np.load(f'{feats_dir}/{FEATURES_FILE}.npy')
        all_tags = np.load(f'{feats_dir}/{TAGS_FILE}.npy')

        return all_tags, all_features

    # obtain all videos
    videos = os.listdir(get_videos_dir())

    all_tags = np.empty(0, dtype=np.str)
    all_features = np.empty((0, extractor.descriptor_size()), dtype='int8')

    i = 0

    # reads all the features files and groups them in one
    for video in videos:
        if video.endswith('.mp4'):
            video_name = video.split('.')[0]
            tags, features = read_features(video_name, feats_dir)

            all_tags = np.concatenate((all_tags, tags))
            all_features = np.concatenate((all_features, features))

            i += 1
            print(f'{all_features.shape[0]:,d} feats read in {i} file{"s" if i > 1 else ""}')

    assert all_features.shape[0] == all_tags.shape[0], 'features and tags length must match'

    # save files
    np.save(f'{feats_dir}/{FEATURES_FILE}.npy', all_features)
    np.save(f'{feats_dir}/{TAGS_FILE}.npy', all_tags)

    return all_tags, all_features


def log_persistent(text: str, log_path: str):
    if not os.path.isfile(log_path):
        log = open(log_path, 'w')
    else:
        log = open(log_path, 'a')
    log.write(text)
    log.close()
