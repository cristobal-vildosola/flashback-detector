import os
import re
import time

import numpy as np

import keyframes.KeyframeSelector as Keyframes
from features.AutoEncoder import AutoEncoderFE
from features.ColorLayout import ColorLayoutFE
from features.FeatureExtractor import FeatureExtractor
from indexes.LSHIndex import BynaryLSHIndex
from indexes.SearchIndex import SearchIndex
from utils.files import read_features, group_features, get_features_path, get_neighbours_path


def nearest_neighbours(
        video_name: str,
        selector: Keyframes.KeyframeSelector,
        extractor: FeatureExtractor,
        index: SearchIndex):
    """
    Searches the nearest neighbours for all the frames in a given video and saves them in a corresponding dir.

    :param video_name: the name of the target video.
    :param selector: the keyframe selector used during feature extraction.
    :param extractor: the feature extractor used during feature extraction.
    :param index: the search index to use.
    """

    # read target video features
    features_path = get_features_path(selector=selector, extractor=extractor)
    tags, features = read_features(video_name=video_name, directory=features_path)

    # analize number of candidates to ensure enough neighbours are found for each frame (and not too much)
    candidates_num = []
    for i in range(features.shape[0]):
        candidates_num.append(index.candidate_count(features[i]))

    print(f'\ncandidates stats:\n'
          f'\tmax: {max(candidates_num)}\n'
          f'\tmean: {np.mean(candidates_num):.1f}\n'
          f'\tmin: {min(candidates_num)}\n')

    res = input('continue? (y/n) ')
    if res.lower() != 'y':
        return

    # open log
    neighbours_path = get_neighbours_path(selector=selector, extractor=extractor, index=index)
    if not os.path.isdir(neighbours_path):
        os.makedirs(neighbours_path)
    neighbours_log = open(f'{neighbours_path}/{video_name}.txt', 'w')

    print(f'searching {index.neighbours_retrieved()} closest frames for video {video_name}')
    t0 = time.time()

    # search closest neighbours for each frame
    for i in range(features.shape[0]):
        closest = index.search(features[i])

        # save results
        tiempo = re.split(' # ', tags[i])[1]
        neighbours_log.write(f'{tiempo} $ {" | ".join(closest)}\n')

        # display progress
        if (i + 1) % (features.shape[0] // 10) == 0:
            print(f'searched {i + 1} ({(i + 1) / features.shape[0]:.0%}) vectors in {time.time() - t0:.1f} seconds')

    duration = time.time() - t0
    neighbours_log.close()
    print(f'\nthe search took {duration:.1f} seconds for {features.shape[0]} frames')

    log_path = f'{neighbours_path}/log.txt'
    if not os.path.isfile(log_path):
        log = open(log_path, 'w')
    else:
        log = open(log_path, 'a')

    log.write(f'{duration:.1f}\t{features.shape[0]}')
    log.close()
    return


def main():
    np.random.seed(1209)
    video_name = '417'
    selector = Keyframes.ThresholdHistDiffKS(threshold=1.3)
    extractor = ColorLayoutFE()

    selectors = [
        Keyframes.FPSReductionKS(n=6),
        Keyframes.MaxHistDiffKS(frames_per_window=2),
        Keyframes.ThresholdHistDiffKS(threshold=1.3),
    ]
    extractors = [
        ColorLayoutFE(),
        AutoEncoderFE.load_autoencoder(name='features/model'),
    ]

    t0 = time.time()
    all_tags, all_features = group_features(selector=selector, extractor=extractor)
    print(f'loading {all_features.shape[0]:,} features took {int(time.time() - t0)} seconds')

    index = BynaryLSHIndex(data=all_features, labels=all_tags, k=100, projections=16, tables=2)
    print(f'index construction took {index.build_time:.1f} seconds\n')

    index.engine.analize_storage()

    nearest_neighbours(
        video_name=video_name,
        selector=selector,
        extractor=extractor,
        index=index)
    return


if __name__ == '__main__':
    main()
