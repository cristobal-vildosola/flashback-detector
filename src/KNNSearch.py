import os
import re
import time

import numpy

from busqueda.LSH import LSHIndex
from busqueda.FlannIndex import Linear
from FeatureExtraction import leer_caracteristicas, agrupar_caracteristicas


def frames_mas_cercanos_video(archivo: str, carpeta_log: str, indice: LSHIndex, k: int = 5):
    """
    Encuentra los k frames más cercanos a cada frame del video dado, dentro de todos los frames en una lista de Videos,
    registra esta información en un log txt.

    :param archivo: el archivo del cuál buscar frames cercanos.
    :param carpeta_log: la carpeta en la cual guardar el log.
    :param indice: el índice de busqueda a usar.
    :param k: el número de frames cercanos a buscar.
    """

    # leer caracteristicas del video
    etiquetas_video, caracteristicas_video = leer_caracteristicas(archivo)

    print(f'Contando número de candidatos por frame')

    num_cand = []
    for i in range(caracteristicas_video.shape[0]):
        cand = indice.index.candidate_count(caracteristicas_video[i])
        num_cand.append(cand)
    print(f'Estadísitcas número candidatos\n'
          f'promedio: {numpy.mean(num_cand):.1f}\n'
          f'max: {max(num_cand)}  min: {min(num_cand)}')

    res = input('desea continuar? (y/n) ')
    if res.lower() != 'y':
        return

    # abrir log
    nombre = re.split('[/.]', archivo)[-2]
    if not os.path.isdir(carpeta_log):
        os.mkdir(carpeta_log)
    log = open(f'{carpeta_log}/{nombre}.txt', 'w')

    print(f'buscando {k} frames más cercanos para {nombre}')

    # medir tiempo
    t0 = time.process_time()

    # buscar los frames más cercanos de cada frame
    for i in range(caracteristicas_video.shape[0]):
        cercanos = indice.search(caracteristicas_video[i])
        cercanos_str = ' | '.join(cercanos)
        # registrar resultado
        tiempo = re.split(' # ', etiquetas_video[i])[1]
        log.write(f'{tiempo} $ {cercanos_str}\n')

        if (i + 1) % (caracteristicas_video.shape[0] // 10) == 0:
            print(f'searched {i + 1} vectors')

    log.close()
    print(f'la búsqueda de {k} frames más cercanos de {caracteristicas_video.shape[0]} frames'
          f' tomó {int(time.process_time() - t0)} segundos')
    return


def main(video: str):
    fps = 6
    tamano = (8, 8)
    k = 50

    t0 = time.process_time()
    etiquetas, caracteristicas = agrupar_caracteristicas(f'../videos/Shippuden_car_{tamano}_{fps}',
                                                         tamano=tamano, recargar=True)

    # buscar inicio y fin de las caracteristicas del video a buscar
    inicio = fin = -1
    for i in range(etiquetas.shape[0]):
        if inicio == -1 and re.split(' # ', etiquetas[i])[0] == video:
            inicio = i
        if inicio != -1 and fin == -1 and re.split(' # ', etiquetas[i])[0] != video:
            fin = i

    # eliminar características del video a buscar
    caracteristicas = numpy.concatenate([caracteristicas[:inicio], caracteristicas[fin:]], axis=0)
    etiquetas = numpy.concatenate([etiquetas[:inicio], etiquetas[fin:]], axis=0)

    print(f'la agrupación de datos ({caracteristicas.shape[0]}) tomó {int(time.process_time() - t0)} segundos')

    indice = LSHIndex(data=caracteristicas, labels=etiquetas, k=k, projections=3, bin_width=150, tables=2)
    print(f'la construcción del índice tomó {indice.build_time:.1f} segundos')

    frames_mas_cercanos_video(f'../videos/Shippuden_car_{tamano}_{fps}/{video}.npy',
                              f'../videos/cercanos_{tamano}_{fps}',
                              indice=indice, k=k)

    return


if __name__ == '__main__':
    main('417')
