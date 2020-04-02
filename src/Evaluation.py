import cv2

from features import AutoEncoderFE, ColorLayoutFE, FeatureExtractor
from indexes import LSHIndex, SGHIndex, LinearIndex, KDTreeIndex, SearchIndex
from keyframes import KeyframeSelector, MaxHistDiffKS, FPSReductionKS
from utils.files import get_results_path


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

    def video_timestamp(self):
        start = self.video_start
        end = self.video_start + self.duration
        return f'{self.video} {timestamp(start)}-{timestamp(end)}'

    def original_timestamp(self):
        start = self.orig_start
        end = self.orig_start + self.duration
        return f'{self.orig_video} {timestamp(start)}-{timestamp(end)}'


def compare_videos(duplicate: Duplicate):
    video = cv2.VideoCapture(f'../videos/Shippuden_original/{duplicate.video}.mp4')
    orig_video = cv2.VideoCapture(f'../videos/Shippuden_original/{duplicate.orig_video}.mp4')

    text = f'{duplicate.video_timestamp()} -> {duplicate.original_timestamp()} ({duplicate.score:.1f})'
    font = cv2.FONT_HERSHEY_COMPLEX
    scale = 1
    thickness = 3
    width, heigth = cv2.getTextSize(text, font, scale, thickness)[0]

    # mover videos a puntos de inicio
    video.set(cv2.CAP_PROP_POS_MSEC, duplicate.video_start * 1000)
    orig_video.set(cv2.CAP_PROP_POS_MSEC, duplicate.orig_start * 1000)

    # variables para reproducir videos con fps distintos
    time = 0
    fps_1 = video.get(cv2.CAP_PROP_FPS)
    fps_2 = orig_video.get(cv2.CAP_PROP_FPS)
    frame_duration_1 = 1.0 / fps_1
    frame_duration_2 = 1.0 / fps_2
    next_frame_1 = frame_duration_1
    next_frame_2 = frame_duration_2

    # frames iniciales
    _, frame1 = video.read()
    _, frame2 = orig_video.read()
    frame1 = cv2.resize(frame1, (640, 358))
    frame2 = cv2.resize(frame2, (640, 358))

    while time < duplicate.duration and video.isOpened() and orig_video.isOpened():

        # avanzar el frame solo cuando pase 1 fps_1 o más
        if time > next_frame_1:
            next_frame_1 += frame_duration_1
            _, frame1 = video.read()
            frame1 = cv2.resize(frame1, (640, 358))

        # avanzar el frame solo cuando pase 1 fps_2 o más
        if time > next_frame_2:
            next_frame_2 += frame_duration_2
            _, frame2 = orig_video.read()
            frame2 = cv2.resize(frame2, (640, 358))

        # concatenar frames y agregar texto
        img = cv2.hconcat([frame1, frame2])
        cv2.putText(img, text, (int(640 - width / 2), heigth), font, scale, (255, 255, 255), thickness=thickness)
        cv2.imshow(f'resultados {duplicate.video}', img)

        # espera de tiempo
        time += 0.02
        res = cv2.waitKey(20)

        # permitir terminar de manera temprana
        if res & 0xff == ord('y'):
            duplicate.correct = True
            return

        elif res & 0xff == ord('n'):
            return

    res = cv2.waitKey(0)
    duplicate.correct = res & 0xff == ord('y')
    return


def evaluate_duplicates(
        video_name: str,
        selector: KeyframeSelector,
        extractor: FeatureExtractor,
        index: SearchIndex,
):
    results_path = get_results_path(selector=selector, extractor=extractor, index=index)

    duplicates = []
    with open(f'{results_path}/{video_name}.txt') as resultados:
        for linea in resultados:
            video_start, orig_video, orig_start, duration, score = linea.split(' ')

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

    for prediccion in duplicates:
        compare_videos(prediccion)

        if prediccion.correct:
            correct += 1

    print(f'precission: {correct / total * 100:.1f}%')
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

    k = 100
    indexes = [
        LinearIndex(dummy=True, k=k),
        KDTreeIndex(dummy=True, trees=5, k=k),
        SGHIndex(dummy=True, projections=14, k=k),
        LSHIndex(dummy=True, projections=16, tables=2, k=k),
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
