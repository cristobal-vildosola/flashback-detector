import os
import re
import time
from typing import Tuple, Callable, Dict, Any

import numpy

from caracteristicas.ColorLayout import color_layout_descriptor
import caracteristicas.Keyframes as Keyframes
from caracteristicas.AutoEncoder import AutoEncoder, load_autoencoder, load_resize_frames


def extract_features(
        file: str, save_path: str, keyframe_gen: Callable = Keyframes.n_frames_per_fps,
        keyframe_args: Dict[str, Any] = None, size: Tuple[int, int] = (8, 8),
        force=False):
    """
    Extrae la caracteristicas de un video y las guarda en un archivo con el mismo nombre del video,
    dentro de la carpeta log. Mide el tiempo que tomó la extracción y la imprime.

    :param file: archivo del video.
    :param save_path: carpeta donde guardar las características.
    :param keyframe_gen: número de frames por segundo a extraer.
    :param keyframe_args: número de frames por segundo a extraer.
    :param size: el tamaño del mapa al cual reducir la dimension de la imagen.
    :param force: si es que es Falso, no se reclaculan características.
    """
    if keyframe_args is None:
        keyframe_args = {}

    # crear carpeta de características si es que es necesario
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # chequear si es que ya se calcularon las características
    video_name = re.split('[/.]', file)[-2]
    if not force and os.path.isfile(f'{save_path}/{video_name}.npy'):
        print(f'Skipping video {video_name}')
        return

    print(f'Extracting features from video {video_name}')
    features = []

    # medir tiempo
    t0 = time.time()

    t = 0
    # recorrer todos los keyframes
    for t, frame in keyframe_gen(file, **keyframe_args):
        # extraer caracteristicas y guardar en el arreglo
        descriptor = color_layout_descriptor(frame, size)
        descriptor = numpy.insert(descriptor, 0, t)
        features.append(descriptor.astype('f4'))

    numpy.save(f'{save_path}/{video_name}.npy', numpy.array(features))

    duration = int(time.time() - t0)
    print(f'feature extraction for {int(t)} seconds took {duration} seconds')

    if os.path.exists(f'{save_path}/log.txt'):
        log = open(f'{save_path}/log.txt', 'a')
    else:
        log = open(f'{save_path}/log.txt', 'w')

    log.write(f'{int(t)}\t{duration}\n')
    log.close()
    return


def extract_features_folder(
        folder: str, keyframe_gen: Callable = Keyframes.n_frames_per_fps,
        keyframe_args: Dict[str, Any] = None, size: Tuple[int, int] = (8, 8),
        force=False, name: str = 'flat_6'):
    """
    Extrae las caracteristicas de todos los archivos dentro de la carpeta especificada
    y los guarda en una nueva carpeta.

    :param folder: la carpeta desde la cuál obtener todos los videos.
    :param keyframe_gen: número de frames por segundo a extraer.
    :param keyframe_args: número de frames por segundo a extraer.
    :param size: el tamaño del mapa al cual reducir la dimension de cada frame.
    :param force: si es que es Falso, no se reclaculan características.
    :param name: .
    """
    if keyframe_args is None:
        keyframe_args = {}

    # obtener todos los archivos en la carpeta
    videos = os.listdir(folder)

    # extraer la caracteristicas de cada comercial
    for video in videos:
        if video.endswith('.mp4'):
            extract_features(
                file=f'{folder}/{video}', save_path=f'{folder}_car_{name}_{size}',
                keyframe_gen=keyframe_gen, keyframe_args=keyframe_args, size=size,
                force=force)

    return


# TODO: generalize feature extraction to 1 method
def extract_features_autoencoder(
        file: str, save_path: str, autoencoder: AutoEncoder, shape: Tuple[int, int, int] = (64, 64, 3),
        keyframe_gen: Callable = Keyframes.n_frames_per_fps, keyframe_args: Dict[str, Any] = None,
        force=False):
    """
    Extrae la caracteristicas de un video y las guarda en un archivo con el mismo nombre del video,
    dentro de la carpeta log. Mide el tiempo que tomó la extracción y la imprime.

    :param file: archivo del video.
    :param save_path: carpeta donde guardar las características.
    :param autoencoder: el tamaño del mapa al cual reducir la dimension de la imagen.
    :param shape:
    :param keyframe_gen: número de frames por segundo a extraer.
    :param keyframe_args: número de frames por segundo a extraer.
    :param force: si es que es Falso, no se reclaculan características.
    """
    if keyframe_args is None:
        keyframe_args = {}

    # crear carpeta de características si es que es necesario
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # chequear si es que ya se calcularon las características
    video_name = re.split('[/.]', file)[-2]
    if not force and os.path.isfile(f'{save_path}/{video_name}.npy'):
        print(f'Skipping video {video_name}')
        return

    print(f'Extracting features from video {video_name}')

    # medir tiempo
    t0 = time.time()

    # extract frames
    frames, ts = load_resize_frames(video=file, keyframe_gen=keyframe_gen, keyframe_args=keyframe_args, shape=shape)

    # encode and transform
    descriptors = autoencoder.encode(numpy.array(frames))
    descriptors = numpy.insert(descriptors, 0, values=ts, axis=1)
    descriptors = descriptors.astype('f4')

    numpy.save(f'{save_path}/{video_name}.npy', descriptors)

    duration = int(time.time() - t0)
    print(f'feature extraction for {int(ts[-1])} seconds took {duration} seconds')

    if os.path.exists(f'{save_path}/log.txt'):
        log = open(f'{save_path}/log.txt', 'a')
    else:
        log = open(f'{save_path}/log.txt', 'w')

    log.write(f'{int(ts[-1])}\t{duration}\n')
    log.close()
    return


def extract_features_autoencoder_folder(
        folder: str, autoencoder: AutoEncoder, shape: Tuple[int, int, int] = (64, 64, 3),
        keyframe_gen: Callable = Keyframes.n_frames_per_fps, keyframe_args: Dict[str, Any] = None,
        force=False, name: str = 'autoencoder'):
    """
    Extrae las caracteristicas de todos los archivos dentro de la carpeta especificada
    y los guarda en una nueva carpeta.

    :param folder: la carpeta desde la cuál obtener todos los videos.
    :param autoencoder: el tamaño del mapa al cual reducir la dimension de cada frame.
    :param shape: el tamaño del mapa al cual reducir la dimension de cada frame.
    :param keyframe_gen: número de frames por segundo a extraer.
    :param keyframe_args: número de frames por segundo a extraer.
    :param force: si es que es Falso, no se reclaculan características.
    :param name: .
    """
    if keyframe_args is None:
        keyframe_args = {}

    # obtener todos los archivos en la carpeta
    videos = os.listdir(folder)

    # extraer la caracteristicas de cada comercial
    for video in videos:
        if video.endswith('.mp4'):
            extract_features_autoencoder(
                f'{folder}/{video}', f'{folder}_car_{name}_{shape}',
                autoencoder=autoencoder, shape=shape,
                keyframe_gen=keyframe_gen, keyframe_args=keyframe_args,
                force=force)

    return


def read_features(archivo: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
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


def group_features(carpeta: str, tamano=(8, 8), recargar: bool = True) \
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
        etiquetas_video, caracteristicas_video = read_features(f'{carpeta}/{archivo}')
        etiquetas = numpy.concatenate((etiquetas, etiquetas_video))
        caracteristicas = numpy.concatenate((caracteristicas, caracteristicas_video))

        i += 1
        print(f'{caracteristicas.shape[0]:,d} lineas leídas en {i} archivos')

    # guardar archivos
    numpy.save(f'{carpeta}/caracteristicas.npy', caracteristicas)
    numpy.save(f'{carpeta}/etiquetas.npy', etiquetas)

    return etiquetas, caracteristicas


def main():
    # extract_features_folder('../videos/Shippuden_low',
    #                        keyframe_gen=Keyframes.window_max_diff, keyframe_args={'frames_per_window': 3},
    #                        size=(8, 8), name='window_3')

    extract_features_folder('../videos/Shippuden_low',
                            keyframe_gen=Keyframes.threshold_diff, keyframe_args={},
                            size=(8, 8), name='threshold_1,3', force=True)

    '''
    fun = Keyframes.n_frames_per_fps
    args = {'n': 6}
    name = 'flat_6_autoencoder'

    autoencoder = load_autoencoder('caracteristicas/model')
    extract_features_autoencoder_folder(
        '../videos/Shippuden_low',
        autoencoder=autoencoder, shape=autoencoder.input_shape,
        keyframe_gen=fun, keyframe_args=args,
        name=name, force=False)
    '''
    return


if __name__ == '__main__':
    main()
