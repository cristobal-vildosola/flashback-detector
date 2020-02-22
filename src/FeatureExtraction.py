import os
import re
import time
from typing import Tuple

import numpy

from features.ColorLayout import color_layout_descriptor
import keyframes.KeyframeSelector as Keyframes
from features.AutoEncoder import AutoEncoder


# TODO: make generic extract features (receives FeatureExtractor)


def extract_features_directory(
        dir_path: str,
        keyframe_selector: Keyframes.KeyframeSelector = Keyframes.SimpleKS(),
        size: Tuple[int, int] = (8, 8),
        force=False):
    """
    Extracts features for the all the videos in the directory and saves them in a new dir depending on the
    keyframe selector and the directory name.

    :param dir_path: the directoray containing the videos.
    :param keyframe_selector: .
    :param size: .
    :param force: when True, calculates features even if it was done previously.
    """

    # create directory when necessary
    root = '../video_features/'
    if not os.path.isdir(root):
        os.mkdir(root)

    videos_name = re.split('[/.]', dir_path)[-1]
    videos_path = f'{root}/{videos_name}'
    if not os.path.isdir(videos_path):
        os.mkdir(videos_path)

    feats_path = f'{videos_path}/{keyframe_selector.name()}_{size}'
    if not os.path.isdir(feats_path):
        os.mkdir(feats_path)

    # create log file
    log_path = f'{feats_path}/log.txt'
    if not os.path.isfile(log_path) or force:
        open(log_path, 'w').close()

    # obtain all files in the directory
    videos = os.listdir(dir_path)

    # extract features from each video
    for video in videos:
        if video.endswith('.mp4'):
            extract_features(
                file_path=f'{dir_path}/{video}', feats_dir=feats_path,
                keyframe_selector=keyframe_selector, size=size, force=force)

    return


def extract_features(
        file_path: str,
        feats_dir: str,
        keyframe_selector: Keyframes.KeyframeSelector = Keyframes.SimpleKS(),
        size: Tuple[int, int] = (8, 8),
        force=False):
    """
    Extracts features for the video and saves them in the given dir.

    :param file_path: video path.
    :param feats_dir: directory to save the features.
    :param keyframe_selector: .
    :param size: .
    :param force: when True, calculates features even if it was done previously.
    """

    video_name = re.split('[/.]', file_path)[-2]
    save_path = f'{feats_dir}/{video_name}.npy'

    # chequear si es que ya se calcularon las características
    if not force and os.path.isfile(save_path):
        print(f'Skipping video {video_name}')
        return

    print(f'Extracting features from video {video_name}')

    # obtain keyframes
    keyframes, timestamps = keyframe_selector.select_keyframes(file_path)

    # medir tiempo
    t0 = time.time()
    features = []

    for i in range(len(keyframes)):
        # extraer caracteristicas y guardar en el arreglo
        descriptor = color_layout_descriptor(keyframes[i], size)
        descriptor = numpy.insert(descriptor, 0, timestamps[i])
        features.append(descriptor.astype('f4'))

    numpy.save(save_path, numpy.array(features))

    duration = time.time() - t0
    print(f'feature extraction for {timestamps[-1]:.0f} seconds took {duration:.2f} seconds\n')

    log = open(f'{feats_dir}/log.txt', 'a')
    log.write(f'{(timestamps[-1]):.0f}\t{duration:.2f}\n')
    log.close()
    return


def read_features(archivo: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    reads the data for a given video and returns the features and tags separated in 2 numpy arrays.

    :param archivo: the file containing the features
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


def group_features(directory: str, size=(8, 8), force: bool = True) \
        -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Groups all the features and tags in a directory and saves them in a file each.

    :param directory: directory containing the features.
    :param size: .
    :param force: when True, groups features even if it was done previously.
    """

    # reload files if grouing was already done
    if os.path.isfile(f'{directory}/caracteristicas.npy') and os.path.isfile(f'{directory}/etiquetas.npy') \
            and not force:
        caracteristicas = numpy.load(f'{directory}/caracteristicas.npy')
        etiquetas = numpy.load(f'{directory}/etiquetas.npy')

        return etiquetas, caracteristicas

    # obtener todos los archivos en la carpeta
    archivos = os.listdir(directory)

    etiquetas = numpy.empty(0, dtype=numpy.str)
    caracteristicas = numpy.empty((0, size[0] * size[1] * 3), dtype='f4')

    # leer las caracteristicas de todos los videos y agruparlas
    i = 0
    for archivo in archivos:
        if not archivo.endswith('.npy') or archivo == 'caracteristicas.npy' or archivo == 'etiquetas.npy':
            continue

        # leer características y juntar con los arreglos.
        etiquetas_video, caracteristicas_video = read_features(f'{directory}/{archivo}')
        etiquetas = numpy.concatenate((etiquetas, etiquetas_video))
        caracteristicas = numpy.concatenate((caracteristicas, caracteristicas_video))

        i += 1
        print(f'{caracteristicas.shape[0]:,d} lineas leídas en {i} archivos')

    # guardar archivos
    numpy.save(f'{directory}/caracteristicas.npy', caracteristicas)
    numpy.save(f'{directory}/etiquetas.npy', etiquetas)

    return etiquetas, caracteristicas


def main():
    extract_features_directory('../videos/Shippuden_low',
                               keyframe_selector=Keyframes.ThresholdHistDiffKS(threshold=1.3),
                               size=(8, 8), force=True)
    return


if __name__ == '__main__':
    main()
