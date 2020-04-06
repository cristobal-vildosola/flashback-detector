import os
import re
import time

import numpy as np

from features import AutoEncoderFE, ColorLayoutFE, FeatureExtractor
from indexes import LSHIndex, SGHIndex, LinearIndex, KDTreeIndex, SearchIndex
from keyframes import KeyframeSelector, MaxHistDiffKS, FPSReductionKS
from utils.files import read_features, group_features, get_features_dir, get_neighbours_dir, log_persistent


def nearest_neighbours(
        video_name: str,
        selector: KeyframeSelector,
        extractor: FeatureExtractor,
        index: SearchIndex,
        analize: bool = False,
        force: bool = False,
):
    """
    Searches the nearest neighbours for all the frames in a given video and saves them in a corresponding dir.

    :param video_name: the name of the target video.
    :param selector: the keyframe selector used during feature extraction.
    :param extractor: the feature extractor used during feature extraction.
    :param index: the search index to use.
    :param analize: whether to analize candidate count or not.
    :param force: whether to force search or not.
    """

    neighbours_dir = get_neighbours_dir(selector=selector, extractor=extractor, index=index)
    neighbours_path = f'{neighbours_dir}/{video_name}.txt'
    if os.path.isfile(neighbours_path) and not force:
        print(f'skipping video {video_name}, already searched with index {index.name()}')
        return

    # read target video features
    features_path = get_features_dir(selector=selector, extractor=extractor)
    labels, features = read_features(video_name=video_name, directory=features_path)

    if analize:
        # analize number of candidates to ensure enough neighbours are found for each frame (and not too much)
        candidates_num = []
        for i in range(features.shape[0]):
            candidates_num.append(index.candidate_count(features[i]))

        print(f'\ncandidates stats:\n'
              f'\tmax: {max(candidates_num)}\n'
              f'\tmean: {np.mean(candidates_num):.1f}\n'
              f'\tmedian: {np.median(candidates_num):.1f}\n'
              f'\tmin: {min(candidates_num)}\n')

        # if input('continue? (y/n) ').lower().strip() != 'y':
        #    return

    # open log
    # neighbours_file = open(neighbours_path, 'w')

    print(f'searching {index.neighbours_retrieved()} closest frames for video {video_name}'
          f' using {index.name()}')
    t0 = time.time()

    # search closest neighbours for each frame
    for i in range(features.shape[0]):
        nearest_points = index.search(features[i])

        if type(nearest_points) != np.ndarray:
            nearest_points = [nearest_points]

        # save results
        timestamp = re.split(' # ', labels[i])[1]
        # neighbours_file.write(f'{timestamp} $ {" | ".join(nearest_points)}\n')

        # display progress
        if (i + 1) % (features.shape[0] // 10) == 0:
            print(f'\tsearched {i + 1} ({(i + 1) / features.shape[0]:.0%}) vectors in {time.time() - t0:.1f} seconds')

    duration = time.time() - t0
    # neighbours_file.close()
    print(f'the search took {duration:.1f} seconds for {features.shape[0]} frames\n')

    log_path = f'{neighbours_dir}/log.txt'
    log_persistent(f'{duration:.1f}\t{features.shape[0]}\n', log_path=log_path)
    return


def main():
    np.random.seed(1209)
    videos = ['119-120', '143', '178', '215', '385', '417', ]  # ['190', '227', ]
    k = 100

    selectors = [
        MaxHistDiffKS(frames_per_window=1),
        FPSReductionKS(n=3),
    ]
    extractors = [
        ColorLayoutFE(),
        # AutoEncoderFE.load_autoencoder(model_name='features/model'),
    ]
    indexes = [
        [LinearIndex, {}],
        [KDTreeIndex, {'trees': 5, 'checks': 1000}],
        [SGHIndex, {'projections': 10, 'training_split': 0.3}],
        [LSHIndex, {'projections': 5, 'tables': 10, 'bin_width': 20}],  # for color layout
        # [LSHIndex, {'projections': 3, 'tables': 10, 'bin_width': 20}],  # for autoencoder
    ]

    for selector in selectors:
        for extractor in extractors:
            print('\n')
            all_tags, all_features = group_features(selector=selector, extractor=extractor)
            print(f'loaded {all_features.shape[0]:,} features')

            for index_class, kwargs in indexes:
                print(f'\nstarting index construction')
                index = index_class(data=all_features, labels=all_tags, k=k, **kwargs)
                print(f'{index.name()} construction took {index.build_time:.1f} seconds\n')

                try:
                    index.engine.analize_storage()
                except:
                    pass

                neighbours_path = get_neighbours_dir(selector=selector, extractor=extractor, index=index)
                if not os.path.isdir(neighbours_path):
                    os.makedirs(neighbours_path)

                log_path = f'{neighbours_path}/constructions.txt'
                log_persistent(f'{index.name()}\t{index.build_time:.2f}\t{all_features.shape[0]}\n', log_path=log_path)

                for video_name in videos:
                    nearest_neighbours(video_name=video_name, selector=selector, extractor=extractor, index=index,
                                       analize=True, force=True)
    return


if __name__ == '__main__':
    main()
