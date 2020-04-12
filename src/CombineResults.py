import os
from typing import List

import cv2

from ManualEvaluation import Duplicate, compare_videos
from utils.files import get_ground_truth_dir


def read_ground_truth(ground_truth_path: str) -> List[Duplicate]:
    duplicates = list()

    with open(ground_truth_path, 'r') as ground_truth:
        for line in ground_truth:
            copy_video, copy_start, orig_video, orig_start, duration = line.strip().split(' ')
            dup = Duplicate(
                copy_video=copy_video,
                copy_start=float(copy_start),
                orig_video=orig_video,
                orig_start=float(orig_start),
                duration=float(duration),
            )
            duplicates.append(dup)

    return duplicates


def get_ground_truth(video_name):
    ground_truth_dir = get_ground_truth_dir()
    filtered_ground_truth_path = f'{ground_truth_dir}/{video_name}_filtered.txt'

    if os.path.isfile(filtered_ground_truth_path):
        return read_ground_truth(filtered_ground_truth_path)

    combine_duplicates(video_name)
    return read_ground_truth(filtered_ground_truth_path)


def combine_duplicates(
        video_name: str,
        max_offset: float = 2,
):
    ground_truth_dir = get_ground_truth_dir()
    ground_truth_path = f'{ground_truth_dir}/{video_name}.txt'
    filtered_ground_truth_path = f'{ground_truth_dir}/{video_name}_filtered.txt'

    if os.path.isfile(filtered_ground_truth_path):
        res = input(f'video results already combined for {video_name}. check current(c), force(f) or skip(default)?')
        if res == 'c':
            check_ground_truth(video_name)
            return
        elif res != 'f':
            return

    duplicates = read_ground_truth(ground_truth_path)
    print(f'read {len(duplicates)} duplicates for video {video_name}')

    # separate copies by video
    video_dups = {}
    for dup in duplicates:
        video = dup.orig_video

        if video not in video_dups:
            video_dups[video] = []
        video_dups[video].append(dup)

    # first remove repeated copies and sort by start
    for video in video_dups:
        current_dups = video_dups[video]
        contained = set()

        for i in range(len(current_dups)):
            dup_a = current_dups[i]

            for j in range(i+1, len(current_dups)):
                dup_b = current_dups[j]

                if dup_a.contains(dup_b):
                    contained.add(dup_b)
                elif dup_b.contains(dup_a):
                    contained.add(dup_a)
                    break

        for dup in contained:
            current_dups.remove(dup)

        video_dups[video] = sorted(current_dups, key=lambda c: c.copy_start)

    print(f'\t{num_copies(video_dups)} copies kept after removing contained videos')

    # combine copies when possible
    for video in video_dups:
        current_dups = video_dups[video]
        for i in range(len(current_dups)):
            dup_a = current_dups[i]

            for j in range(i + 1, len(current_dups)):
                dup_b = current_dups[j]

                if dup_a.distance(dup_b) < 1 and dup_a.offset_diff(dup_b) < max_offset:
                    dup_a.combine(dup_b)

    # detect overlapped duplicates
    for video in video_dups:
        current_dups = video_dups[video]
        repeated = set()

        for i in range(len(current_dups)):
            dup_a = current_dups[i]

            for j in range(i + 1, len(current_dups)):
                dup_b = current_dups[j]

                # if there's overlapping, delete shortest
                if dup_a.distance(dup_b) < .5 and dup_a.offset_diff(dup_b) < max_offset + 1:
                    if dup_a.duration >= dup_b.duration:
                        repeated.add(dup_b)
                    else:
                        repeated.add(dup_a)

        for dup in repeated:
            current_dups.remove(dup)

    print(f'\t{num_copies(video_dups)} duplicates kept after combining and removing overlapped')

    # write to file
    log = open(filtered_ground_truth_path, 'w')
    for video in sorted(video_dups.keys()):
        for dup in video_dups[video]:
            log.write(f'{dup}\n')
    log.close()

    check_ground_truth(video_name)
    return


def num_copies(video_copies: dict):
    n = 0
    for video in video_copies:
        n += len(video_copies[video])
    return n


def check_ground_truth(video_name):
    ground_truth_dir = get_ground_truth_dir()
    ground_truth_path = f'{ground_truth_dir}/{video_name}_filtered.txt'

    duplicates = read_ground_truth(ground_truth_path)
    print(f'read {len(duplicates)} duplicates for video {video_name}')

    for i in range(len(duplicates)):
        if not compare_videos(duplicates[i]):
            print(i+1, duplicates[i])

    return


def main():
    videos = ['417', '143', '215', '385', ]  # '178', '119-120', ]

    for video in videos:
        combine_duplicates(video_name=video, max_offset=2)
    return


if __name__ == '__main__':
    main()
