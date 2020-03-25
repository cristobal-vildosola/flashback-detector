import cv2

import keyframes.KeyframeSelector as Keyframes
from features.AutoEncoder import AutoEncoderFE
from features.ColorLayout import ColorLayoutFE
from features.FeatureExtractor import FeatureExtractor
from indexes.LSHIndex import BynaryLSHIndex
from indexes.SearchIndex import SearchIndex
from utils.files import get_results_path


class Prediccion:
    def __init__(self, video: str, inicio_video: float, capitulo: str, inicio_cap: float, duracion: float):
        self.video = video
        self.inicio_video = inicio_video
        self.duracion = duracion

        self.capitulo = capitulo
        self.inicio_cap = inicio_cap

        self.correcta = False


def comparar_videos(prediccion: Prediccion):
    amv = cv2.VideoCapture(f'../videos/Shippuden_original/{prediccion.video}.mp4')
    cap = cv2.VideoCapture(f'../videos/Shippuden_original/{prediccion.capitulo}.mp4')

    text = f'{int(prediccion.inicio_video / 60)}:{int(prediccion.inicio_video) % 60} ({prediccion.duracion:.1f}) ' + \
           f'capitulo {prediccion.capitulo} - {int(prediccion.inicio_cap / 60)}:{int(prediccion.inicio_cap) % 60}'
    font = cv2.FONT_HERSHEY_COMPLEX
    scale = 1
    thick = 3
    width, heigth = cv2.getTextSize(text, font, scale, thick)[0]

    # mover videos a puntos de inicio
    amv.set(cv2.CAP_PROP_POS_MSEC, prediccion.inicio_video * 1000)
    cap.set(cv2.CAP_PROP_POS_MSEC, prediccion.inicio_cap * 1000)

    # variables para reproducir videos con fps distintos
    tiempo = 0
    fps1 = amv.get(cv2.CAP_PROP_FPS)
    fps2 = cap.get(cv2.CAP_PROP_FPS)
    tiempo_frame1 = 1.0 / fps1
    tiempo_frame2 = 1.0 / fps2
    siguiente_frame1 = tiempo_frame1
    siguiente_frame2 = tiempo_frame2

    # frames iniciales
    _, frame1 = amv.read()
    _, frame2 = cap.read()
    frame1 = cv2.resize(frame1, (640, 358))
    frame2 = cv2.resize(frame2, (640, 358))

    while tiempo < prediccion.duracion and amv.isOpened() and cap.isOpened():

        # avanzar el frame solo cuando pase 1 fps_1 o más
        if tiempo > siguiente_frame1:
            siguiente_frame1 += tiempo_frame1
            _, frame1 = amv.read()
            frame1 = cv2.resize(frame1, (640, 358))

        # avanzar el frame solo cuando pase 1 fps_2 o más
        if tiempo > siguiente_frame2:
            siguiente_frame2 += tiempo_frame2
            _, frame2 = cap.read()
            frame2 = cv2.resize(frame2, (640, 358))

        # concatenar frames y agregar texto
        img = cv2.hconcat([frame1, frame2])
        cv2.putText(img, text, (int(640 - width / 2), heigth), font, scale, (255, 255, 255), thickness=thick)
        cv2.imshow(f'resultados {prediccion.video}', img)

        # espera de tiempo
        tiempo += 0.02
        res = cv2.waitKey(20)

        # permitir terminar de manera temprana
        if res & 0xff == ord('y'):
            prediccion.correcta = True
            return

        elif res & 0xff == ord('n'):
            return

    res = cv2.waitKey(0)
    prediccion.correcta = res & 0xff == ord('y')
    return


def evaluar_resultados(
        video_name: str,
        selector: Keyframes.KeyframeSelector,
        extractor: FeatureExtractor,
        index: SearchIndex,
):
    results_path = get_results_path(selector=selector, extractor=extractor, index=index)

    predicciones = []
    with open(f'{results_path}/{video_name}.txt') as resultados:
        for linea in resultados:
            tiempo_video_inicio, duracion, capitulo, tiempo_cap_inicio, score = linea.split(' ')

            predicciones.append(
                Prediccion(video_name, float(tiempo_video_inicio), capitulo, float(tiempo_cap_inicio), float(duracion))
            )

    correctas = 0
    total = len(predicciones)

    tiempo_detectado = 0
    amv = cv2.VideoCapture(f'../videos/Shippuden_original/{video_name}.mp4')
    tiempo_total = amv.get(cv2.CAP_PROP_FRAME_COUNT) / amv.get(cv2.CAP_PROP_FPS)

    for prediccion in predicciones:
        comparar_videos(prediccion)

        if prediccion.correcta:
            correctas += 1
            tiempo_detectado += prediccion.duracion

    print(f'aciertos: {correctas / total * 100:.1f}%')
    print(f'tiempo detectado: {tiempo_detectado / tiempo_total * 100:.1f}%')
    return


if __name__ == '__main__':
    selectors = [
        Keyframes.FPSReductionKS(n=6),
        Keyframes.MaxHistDiffKS(frames_per_window=2),
        Keyframes.ThresholdHistDiffKS(threshold=1.3),
    ]
    extractors = [
        ColorLayoutFE(),
        AutoEncoderFE.load_autoencoder(name='features/model'),
    ]

    evaluar_resultados(
        '417',
        selector=selectors[2],
        extractor=extractors[1],
        index=BynaryLSHIndex(dummy=True, projections=16, tables=2),
    )
