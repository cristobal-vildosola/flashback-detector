from features import AutoEncoderFE, ColorLayoutFE
from indexes import LSHIndex, SGHIndex, LinearIndex, KDTreeIndex
from keyframes import MaxHistDiffKS, FPSReductionKS
from utils.files import get_features_dir, get_neighbours_dir, get_results_dir


def read_selection_log(selector, extractors):
    total_frames = 0
    total_time = 0

    for extractor in extractors:
        features_dir = get_features_dir(selector, extractor)

        with open(f'{features_dir}/selection_log.txt', 'r') as selection_log:
            for line in selection_log:
                frames, time = line.split('\t')
                total_frames += int(frames)
                total_time += float(time)

    naruto_frames = 16544314 * 2
    print(f'{selector.name()} selected {naruto_frames / total_time:.3f} frames per second')
    return


def read_extraction_log(extractor, selectors):
    total_frames = 0
    total_time = 0

    for selector in selectors:
        features_dir = get_features_dir(selector, extractor)

        with open(f'{features_dir}/extraction_log.txt', 'r') as selection_log:
            for line in selection_log:
                frames, time = line.split('\t')
                total_frames += int(frames)
                total_time += float(time)

    print(f'{extractor.name()} extracted {total_frames / total_time:.3f} frames per second')
    return


def read_index_logs(selectors, extractors, index):
    search_frames = 0
    search_time = 0
    const_frames = 0
    const_time = 0

    for selector in selectors:
        for extractor in extractors:
            neighbours_dir = get_neighbours_dir(selector, extractor, index)

            with open(f'{neighbours_dir}/log.txt', 'r') as search_log:
                for line in search_log:
                    time, frames = line.split('\t')
                    search_frames += int(frames)
                    search_time += float(time)

            with open(f'{neighbours_dir}/constructions.txt', 'r') as const_log:
                for line in const_log:
                    _, time, frames = line.split('\t')
                    const_frames += int(frames)
                    const_time += float(time)

    print(f'{index.name()} searched {search_frames / search_time:.3f} frames per second')
    print(f'{index.name()} indexed {const_frames / const_time:.3f} frames per second')
    return


def main():
    selectors = [
        FPSReductionKS(n=3),
        MaxHistDiffKS(frames_per_window=1),
    ]
    extractors = [
        ColorLayoutFE(),
        AutoEncoderFE(dummy=True),
    ]
    indexes = [
        LinearIndex(dummy=True),
        KDTreeIndex(dummy=True, trees=5),
        SGHIndex(dummy=True, projections=10),
    ]
    lshs = [
        LSHIndex(dummy=True, projections=5, tables=10),  # color layout
        LSHIndex(dummy=True, projections=3, tables=10),  # auto encoder
    ]

    for selector in selectors:
        read_selection_log(selector, extractors)

    for extractor in extractors:
        read_extraction_log(extractor, selectors)

    for index in indexes:
        read_index_logs(selectors, extractors, index)

    for i in range(len(lshs)):
        read_index_logs(selectors, [extractors[i]], lshs[i])


    return


if __name__ == '__main__':
    main()
