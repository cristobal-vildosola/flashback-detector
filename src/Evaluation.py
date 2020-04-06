from CombineResults import get_ground_truth
from ManualEvaluation import read_results
from features import AutoEncoderFE, ColorLayoutFE, FeatureExtractor
from indexes import LSHIndex, SGHIndex, LinearIndex, KDTreeIndex, SearchIndex
from keyframes import KeyframeSelector, MaxHistDiffKS, FPSReductionKS
from utils.files import get_results_dir, log_persistent, GROUND_TRUTH_DIR


def calc_precision_recall(
        video_name: str,
        selector: KeyframeSelector,
        extractor: FeatureExtractor,
        index: SearchIndex,
):
    results_path = get_results_dir(selector=selector, extractor=extractor, index=index)
    duplicates = read_results(results_path, video_name)
    ground_truth = get_ground_truth(video_name)

    gt_recalled = 0
    iou_sum = 0

    for gt in ground_truth:
        recalled = 0
        iou = 0

        for dup in duplicates:

            intersection = dup.intersection(gt)
            if intersection > 0 and dup.offset_diff(gt) < 5:
                recalled = 1

                # sum iou for different partial dups
                iou += intersection

        gt_recalled += recalled
        iou_sum += iou

    recall = gt_recalled / len(ground_truth)
    iou_mean = iou_sum / gt_recalled

    correct = 0
    for dup in duplicates:
        for gt in ground_truth:

            if dup.intersection(gt) > 0 and dup.offset_diff(gt) < 5:
                correct += 1
                break

    precision = correct / len(duplicates)

    folder = results_path.split('/')[-1]
    print(f'{folder}\t{video_name}\t{recall:.2f}\t{iou_mean:.2f}\t{precision:.2f}')

    log_persistent(f'{folder}\t{video_name}\t{recall:.2f}\t{iou_mean:.2f}\n', f'{GROUND_TRUTH_DIR}/results.txt')
    return


def main():
    videos = ['417', '143', '215', '385', ]  # '178', '119-120', ]

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
        LSHIndex(dummy=True, projections=5, tables=10),  # color layout
        LSHIndex(dummy=True, projections=3, tables=10),  # auto encoder
    ]

    datasets = [
        (selectors[0], extractors[0], indexes[0],),  # linear
        (selectors[0], extractors[0], indexes[1],),  # kdtree
        (selectors[0], extractors[0], indexes[2],),  # sgh
        (selectors[0], extractors[0], indexes[3],),  # lsh

        (selectors[0], extractors[1], indexes[1],),  # kdtree
        (selectors[0], extractors[1], indexes[2],),  # sgh
        (selectors[0], extractors[1], indexes[4],),  # lsh

        (selectors[1], extractors[0], indexes[1],),  # kdtree
        (selectors[1], extractors[0], indexes[2],),  # sgh
        (selectors[1], extractors[0], indexes[3],),  # lsh

        (selectors[1], extractors[1], indexes[1],),  # kdtree
        (selectors[1], extractors[1], indexes[2],),  # sgh
        (selectors[1], extractors[1], indexes[4],),  # lsh
    ]

    for video in videos:
        for selector, extractor, index in datasets:
            calc_precision_recall(
                video_name=video,
                selector=selector,
                extractor=extractor,
                index=index,
            )
    return


if __name__ == '__main__':
    main()
