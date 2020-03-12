import os
import re
import time

import numpy

import keyframes.KeyframeSelector as Keyframes
from utils.files import read_features, group_features, get_features_path, get_neighbours_path
from features.ColorLayout import ColorLayoutFE
from indexes.LSHIndex import LSHIndex
from features.FeatureExtractor import FeatureExtractor


def nearest_neighbours(
        video_name: str,
        videos_folder: str,
        selector: Keyframes.KeyframeSelector,
        extractor: FeatureExtractor,
        index: LSHIndex):
    """
    Searches the nearest neighbours for all the frames in a given video and saves them in a corresponding dir.

    :param video_name: the name of the target video.
    :param videos_folder: the directory containing the videos.
    :param selector: the keyframe selector used during feature extraction.
    :param extractor: the feature extractor used during feature extraction.
    :param index: the search index to use.
    """

    # read target video features
    features_path = get_features_path(videos_folder=videos_folder, selector=selector, extractor=extractor)
    tags, features = read_features(f'{features_path}/{video_name}.npy')

    # analize number of candidates to ensure enough neighbours are ound for each frame
    print(f'counting number of candidates per frame')
    candidates_num = []
    for i in range(features.shape[0]):
        cand = index.engine.candidate_count(features[i])
        candidates_num.append(cand)

    print(f'candidates stats:\n'
          f'\tmean: {numpy.mean(candidates_num):.1f}\n'
          f'\tmax: {max(candidates_num)}\n'
          f'\tmin: {min(candidates_num)}')

    res = input('continue? (y/n) ')
    if res.lower() != 'y':
        return

    # open log
    neighbours_path = get_neighbours_path(videos_folder=videos_folder, selector=selector, extractor=extractor)
    if not os.path.isdir(neighbours_path):
        os.makedirs(neighbours_path)
    neighbours_log = open(f'{neighbours_path}/{video_name}.txt', 'w')

    print(f'searching {index.k} closest frames for video {video_name}')
    t0 = time.time()

    # search closest neighbours for each frame
    for i in range(features.shape[0]):
        closest = index.search(features[i])

        # save results
        tiempo = re.split(' # ', tags[i])[1]
        neighbours_log.write(f'{tiempo} $ {" | ".join(closest)}\n')

        # display progress
        if (i + 1) % (features.shape[0] // 10) == 0:
            print(f'searched {i + 1} ({(i + 1) / features.shape[0]:%.0}) vectors'
                  f'in {int(time.time() - t0):.1f} seconds')

    neighbours_log.close()
    print(f'the search took {int(time.time() - t0):.1f} seconds for {features.shape[0]} frames')
    return


def main():
    numpy.random.seed(1209)
    video_name = '417'
    videos_folder = 'Shippuden_low'
    selector = Keyframes.MaxHistDiffKS()
    extractor = ColorLayoutFE()

    t0 = time.time()
    all_tags, all_features = group_features(videos_folder=videos_folder, selector=selector, extractor=extractor)
    print(f'loading {all_features.shape[0]:,} features took {int(time.time() - t0)} seconds')

    index = LSHIndex(data=all_features, labels=all_tags, k=100, projections=3, bin_width=150, tables=2)
    print(f'index construction took {index.build_time:.1f} seconds')

    nearest_neighbours(
        video_name=video_name,
        videos_folder=videos_folder,
        selector=selector,
        extractor=extractor,
        index=index)
    return


if __name__ == '__main__':
    main()
