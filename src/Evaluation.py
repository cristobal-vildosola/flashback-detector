import os

import cv2

from features import AutoEncoderFE, ColorLayoutFE, FeatureExtractor
from indexes import LSHIndex, SGHIndex, LinearIndex, KDTreeIndex, SearchIndex
from keyframes import KeyframeSelector, MaxHistDiffKS, FPSReductionKS
from utils.files import get_results_dir, get_orig_videos_dir, get_ground_truth_dir, log_persistent


def timestamp(time):
    return f'{int(time / 60):02}:{int(time % 60):02}'


class Duplicate:
    def __init__(
            self,
            video: str,
            video_start: float,
            orig_video: str,
            orig_start: float,
            duration: float,
            score: float,
    ):
        self.video = video
        self.video_start = video_start

        self.orig_video = orig_video
        self.orig_start = orig_start

        self.duration = duration
        self.score = score

        self.correct = False

    def video_timestamp(self) -> str:
        start = self.video_start
        end = self.video_start + self.duration
        return f'{self.video} {timestamp(start)}-{timestamp(end)}'

    def original_timestamp(self) -> str:
        start = self.orig_start
        end = self.orig_start + self.duration
        return f'{self.orig_video} {timestamp(start)}-{timestamp(end)}'

    def __str__(self) -> str:
        return f'{self.video_start} {self.orig_video} {self.orig_start} {self.duration}'


def compare_videos(duplicate: Duplicate, folder: str):
    # open videos
    videos_path = get_orig_videos_dir()
    copy_video = cv2.VideoCapture(f'{videos_path}/{duplicate.video}.mp4')
    orig_video = cv2.VideoCapture(f'{videos_path}/{duplicate.orig_video}.mp4')

    # add duplicate information
    text = f'{duplicate.video_timestamp()} -> {duplicate.original_timestamp()} ({duplicate.score:.1f})'
    font = cv2.FONT_HERSHEY_COMPLEX
    scale = 1
    thickness = 3
    width, heigth = cv2.getTextSize(text, font, scale, thickness)[0]

    # move pointers to start of videos
    copy_video.set(cv2.CAP_PROP_POS_MSEC, duplicate.video_start * 1000)
    orig_video.set(cv2.CAP_PROP_POS_MSEC, duplicate.orig_start * 1000)

    # reproduction variables
    img = None
    time = 0
    fps = copy_video.get(cv2.CAP_PROP_FPS)
    frame_duration = 1 / fps
    frame_duration_milis = int(frame_duration * 1000)
    pause = False

    while True:
        finished = not (time < duplicate.duration and copy_video.isOpened() and orig_video.isOpened())

        # retrieve next frames
        if not pause and not finished:
            _, frame1 = copy_video.read()
            _, frame2 = orig_video.read()

            frame1 = cv2.resize(frame1, (640, 358))
            frame2 = cv2.resize(frame2, (640, 358))

            # concatenate frames and add info
            img = cv2.hconcat([frame1, frame2])
            cv2.putText(img, text, (int(640 - width / 2) - 1, heigth), font, scale, (0, 0, 0), thickness=thickness + 4)
            cv2.putText(img, text, (int(640 - width / 2), heigth), font, scale, (255, 255, 255), thickness=thickness)

            time += frame_duration

        # time step and user input
        cv2.imshow(f'resultados {folder}', img)
        res = cv2.waitKey(frame_duration_milis)

        # return value
        if res & 0xff == ord('y'):
            return True
        elif res & 0xff == ord('n'):
            return False

        # reset video
        elif res & 0xff == ord('r'):
            time = 0
            copy_video.set(cv2.CAP_PROP_POS_MSEC, duplicate.video_start * 1000)
            orig_video.set(cv2.CAP_PROP_POS_MSEC, duplicate.orig_start * 1000)

        # pause video (p or space)
        elif res & 0xff == ord('p') or res == 32:
            pause = not pause


def evaluate_duplicates(
        video_name: str,
        selector: KeyframeSelector,
        extractor: FeatureExtractor,
        index: SearchIndex,
):
    results_path = get_results_dir(selector=selector, extractor=extractor, index=index)

    # read duplicates
    duplicates = []
    with open(f'{results_path}/{video_name}.txt') as results:
        for line in results:
            video_start, orig_video, orig_start, duration, score = line.split(' ')

            duplicates.append(
                Duplicate(
                    video=video_name,
                    video_start=float(video_start),
                    orig_video=orig_video,
                    orig_start=float(orig_start),
                    duration=float(duration),
                    score=float(score),
                )
            )

    correct = 0
    total = len(duplicates)
    folder = results_path.split('/')[-1]

    ground_truth_path = get_ground_truth_dir()
    if not os.path.isdir(ground_truth_path):
        os.makedirs(ground_truth_path)

    for duplicate in duplicates:
        if compare_videos(duplicate, folder):
            correct += 1
            log_persistent(str(duplicate), f'{ground_truth_path}/{video_name}.txt')

    cv2.destroyAllWindows()
    precision = correct / total * 100
    print(f'precision: {precision:.1f}%')
    log_persistent(f'{video_name} {precision:.1f}', f'{results_path}/results.txt')
    return


def main():
    videos = ['417', '143', '215', '385', '178', '119-120', ]

    selectors = [
        FPSReductionKS(n=3),
        MaxHistDiffKS(frames_per_window=1),
    ]
    extractors = [
        ColorLayoutFE(),
        AutoEncoderFE(dummy=True, model_name='model'),
    ]

    indexes = [
        LinearIndex(dummy=True),
        KDTreeIndex(dummy=True, trees=5),
        SGHIndex(dummy=True, projections=10),
        LSHIndex(dummy=True, projections=5, tables=10),  # color layout
        # LSHIndex(dummy=True, projections=3, tables=10),  # auto encoder
    ]

    for video in videos:
        for selector in selectors:
            for extractor in extractors:
                for index in indexes:
                    evaluate_duplicates(
                        video_name=video,
                        selector=selector,
                        extractor=extractor,
                        index=index,
                    )
    return


if __name__ == '__main__':
    main()
