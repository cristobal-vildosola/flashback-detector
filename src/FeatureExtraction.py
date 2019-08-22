import os
import re
import time
from typing import Tuple, Callable, Dict, Any

import cv2
import numpy

from caracteristicas.ColorLayout import color_layout_descriptor
import caracteristicas.Keyframes as Keyframes


def extraer_caracteristicas_video(archivo: str, carpeta_log: str, keyframe_gen: Callable = Keyframes.n_frames_per_fps,
                                  keyframe_args: Dict[str, Any] = {'n': 6}, tamano: Tuple[int, int] = (8, 8),
                                  force=False):
    """
    Extrae la caracteristicas de un video y las guarda en un archivo con el mismo nombre del video,
    dentro de la carpeta log. Mide el tiempo que tomó la extracción y la imprime.

    :param archivo: archivo del video.
    :param carpeta_log: carpeta donde guardar las características.
    :param keyframe_gen: número de frames por segundo a extraer.
    :param keyframe_args: número de frames por segundo a extraer.
    :param tamano: el tamaño del mapa al cual reducir la dimension de la imagen.
    :param force: si es que es Falso, no se reclaculan características.
    """

    # abrir video
    nombre = re.split('[/.]', archivo)[-2]
    video = cv2.VideoCapture(archivo)

    # crear carpeta de características si es que es necesario
    if not os.path.isdir(carpeta_log):
        os.mkdir(carpeta_log)

    # medir tiempo
    t0 = time.time()

    # chequear si es que ya se calcularon las características
    if not force and os.path.isfile(f'{carpeta_log}/{nombre}.npy'):
        return

    print(f'extracting features from video {nombre}')
    caracteristicas = []

    t = 0
    # recorrer todos los keyframes
    for t, frame in keyframe_gen(archivo, **keyframe_args):
        # extraer caracteristicas y guardar en el arreglo
        descriptor = color_layout_descriptor(frame, tamano)
        descriptor = numpy.insert(descriptor, 0, t)
        caracteristicas.append(descriptor.astype('f4'))

    video.release()
    numpy.save(f'{carpeta_log}/{nombre}.npy', numpy.array(caracteristicas))

    tiempo = int(time.time() - t0)
    print(f'la extracción de {int(t)} segundos de video tomo {tiempo} segundos')

    log = open(f'log_{carpeta_log}.txt', 'a')
    log.write(f'{int(t)}\t{tiempo}\n')
    log.close()
    return


def extraer_caracteristicas_videos(carpeta: str, keyframe_gen: Callable = Keyframes.n_frames_per_fps,
                                   keyframe_args: Dict[str, Any] = {'n': 6}, tamano: Tuple[int, int] = (8, 8),
                                   force=False, name: str = '6'):
    """
    Extrae las caracteristicas de todos los archivos dentro de la carpeta especificada
    y los guarda en una nueva carpeta.

    :param carpeta: la carpeta desde la cuál obtener todos los videos.
    :param keyframe_gen: número de frames por segundo a extraer.
    :param keyframe_args: número de frames por segundo a extraer.
    :param tamano: el tamaño del mapa al cual reducir la dimension de cada frame.
    :param force: si es que es Falso, no se reclaculan características.
    :param name: .
    """

    # obtener todos los archivos en la carpeta
    videos = os.listdir(carpeta)

    # extraer la caracteristicas de cada comercial
    for video in videos:
        if video.endswith('.mp4'):
            extraer_caracteristicas_video(f'{carpeta}/{video}', f'{carpeta}_car_{name}_{tamano}',
                                          keyframe_gen=keyframe_gen, keyframe_args=keyframe_args, tamano=tamano,
                                          force=force)

    return


def leer_caracteristicas(archivo: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Lee los datos de un archivo y las retorna en 2 arreglos de numpy, 1 para características y otro para etiquetas.

    :param archivo: el archivo a leer.

    :return: 2 arreglos de numpy, uno para etiquetas y otro para características, en ese orden.
    """

    datos = numpy.load(archivo)
    caracteristicas = datos[:, 1:]

    # generar etiquetas
    nombre = re.split('[/.]', archivo)[-2]
    etiqueta_list = []
    for i in range(datos.shape[0]):
        etiqueta_list.append(f'{nombre} # {datos[i][0]} # {i + 1}')

    etiquetas = numpy.array(etiqueta_list)
    return etiquetas, caracteristicas.astype('f4')


def agrupar_caracteristicas(carpeta: str, tamano=(8, 8), recargar: bool = True) \
        -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Agrupa todos los datos de la carpeta dada en 2 arreglos de numpy, 1 para características y otro para etiquetas.
    Finalmente los guarda en archivos para reutilizarlos si se vuelve a intentar agrupar la misma carpeta.

    :param carpeta: carpeta donde están las características que agrupar.
    :param recargar: determina si se deben recargar los archivos previamente generados (si es que existen).
    :param tamano: tamaño del vector de características.

    :return: 2 arreglos de numpy, uno para etiquetas y otro para características, en ese orden.
    """
    # reutilizar archivos si ya se hizo agrupación antes
    if os.path.isfile(f'{carpeta}/caracteristicas.npy') and os.path.isfile(f'{carpeta}/etiquetas.npy') and recargar:
        caracteristicas = numpy.load(f'{carpeta}/caracteristicas.npy')
        etiquetas = numpy.load(f'{carpeta}/etiquetas.npy')

        return etiquetas, caracteristicas

    # obtener todos los archivos en la carpeta
    archivos = os.listdir(carpeta)

    etiquetas = numpy.empty(0, dtype=numpy.str)
    caracteristicas = numpy.empty((0, tamano[0] * tamano[1] * 3), dtype='f4')

    # leer las caracteristicas de todos los videos y agruparlas
    i = 0
    for archivo in archivos:
        if not archivo.endswith('.npy') or archivo == 'caracteristicas.npy' or archivo == 'etiquetas.npy':
            continue

        # leer características y juntar con los arreglos.
        etiquetas_video, caracteristicas_video = leer_caracteristicas(f'{carpeta}/{archivo}')
        etiquetas = numpy.concatenate((etiquetas, etiquetas_video))
        caracteristicas = numpy.concatenate((caracteristicas, caracteristicas_video))

        i += 1
        print(f'{caracteristicas.shape[0]:,d} lineas leídas en {i} archivos')

    # guardar archivos
    numpy.save(f'{carpeta}/caracteristicas.npy', caracteristicas)
    numpy.save(f'{carpeta}/etiquetas.npy', etiquetas)

    return etiquetas, caracteristicas


def main():
    tamano = (8, 8)
    fun = Keyframes.n_frames_per_fps
    args = {'n': 6}
    name = 'flat_6'

    extraer_caracteristicas_videos('../videos/Shippuden', keyframe_gen=fun, keyframe_args=args, tamano=tamano,
                                   force=True, name=name)
    # agrupar_caracteristicas(f'../videos/Shippuden_car_{tamano}_{fps_extraccion}')
    return


if __name__ == '__main__':
    main()
