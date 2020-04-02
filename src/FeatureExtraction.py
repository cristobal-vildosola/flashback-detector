import os
import re
import time

import numpy as np

from features import AutoEncoderFE, ColorLayoutFE, FeatureExtractor
from keyframes import KeyframeSelector, MaxHistDiffKS, FPSReductionKS
from utils.files import get_features_path, get_videos_path, group_features


def extract_features_directory(
        selector: KeyframeSelector,
        extractor: FeatureExtractor,
        force=False
):
    """
    Extracts features for the all the videos in the directory and saves them in a new directory obtained using
    get_features_path.

    :param selector: .
    :param extractor: .
    :param force: when True, calculates features even if it was done previously.
    """

    # create directory when necessary
    feats_path = get_features_path(selector=selector, extractor=extractor)
    if not os.path.isdir(feats_path):
        os.makedirs(feats_path)

    # create or empty log files
    if not os.path.isfile(f'{feats_path}/extraction_log.txt') or force:
        open(f'{feats_path}/extraction_log.txt', 'w').close()
        open(f'{feats_path}/selection_log.txt', 'w').close()

    # obtain all files in the directory
    videos_path = get_videos_path()
    videos = os.listdir(videos_path)

    # extract features from each video
    for video in videos:
        if video.endswith('.mp4'):
            extract_features(
                file_path=f'{videos_path}/{video}', save_dir=feats_path,
                selector=selector, extractor=extractor, force=force)

    # group all features into 2 files (features and tags)
    group_features(
        selector=selector,
        extractor=extractor,
        force=force
    )
    return


def extract_features(
        file_path: str,
        save_dir: str,
        selector: KeyframeSelector,
        extractor: FeatureExtractor,
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
    save_path_feats = f'{save_dir}/{video_name}-feats.npy'
    save_path_tags = f'{save_dir}/{video_name}-tags.npy'

    # skip already processed videos
    if not force and os.path.isfile(save_path_feats) and os.path.isfile(save_path_tags):
        print(f'Skipping video {video_name}')
        return

    print(f'Extracting features from video {video_name}')

    # obtain keyframes
    t0 = time.time()
    keyframes, timestamps, total_frames = selector.select_keyframes(file_path)

    selection = time.time() - t0
    print(f'selected {len(keyframes)} of {total_frames} frames in {selection:.1f} secs')

    # log selection time
    log = open(f'{save_dir}/selection_log.txt', 'a')
    log.write(f'{len(timestamps)}\t{selection:.2f}\n')
    log.close()

    # measure time
    t0 = time.time()

    # extract features and save
    features = extractor.extract_features(keyframes)
    np.save(save_path_feats, features)

    # generate tags and save
    tags = np.empty(timestamps.shape[0], dtype='<U30')
    for i in range(timestamps.shape[0]):
        tags[i] = f'{video_name} # {timestamps[i]:.2f} # {i + 1}'
    np.save(save_path_tags, tags)

    extraction = time.time() - t0
    print(f'feature extraction for {len(timestamps)} frames took {extraction:.2f} seconds\n')

    # log extraction time
    log = open(f'{save_dir}/extraction_log.txt', 'a')
    log.write(f'{len(timestamps)}\t{extraction:.2f}\n')
    log.close()
    return


def main():
    selectors = [
        FPSReductionKS(n=3),
        MaxHistDiffKS(frames_per_window=1),
    ]
    extractors = [
        ColorLayoutFE(),
        AutoEncoderFE.load_autoencoder(model_name='features/model'),
    ]

    for selector in selectors:
        for extractor in extractors:
            extract_features_directory(selector=selector, extractor=extractor)

    return


if __name__ == '__main__':
    main()
