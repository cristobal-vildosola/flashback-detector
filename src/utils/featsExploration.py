import matplotlib.pyplot as plt
import numpy as np

from features import AutoEncoderFE, ColorLayoutFE
from keyframes import FPSReductionKS
from utils.files import group_features


def main():
    selector = FPSReductionKS(n=3)
    extractors = [
        ColorLayoutFE(),
        AutoEncoderFE.load_autoencoder(model_name='../features/model'),
    ]

    for extractor in extractors:
        tags, features = group_features(selector=selector, extractor=extractor)

        x = range(features.shape[1])
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        mins = np.min(features, axis=0)
        maxs = np.max(features, axis=0)

        plt.figure()
        plt.title(f'{selector.name()} - {extractor.name()}')

        plt.errorbar(x, mean, std, fmt='.b', ecolor='r', capsize=3)

        plt.plot(x, maxs, 'g')
        plt.plot(x, mins, 'y')
        plt.legend(['max', 'min', ])

        plt.xlabel('features')
        plt.ylabel('value')

    plt.show()
    return


if __name__ == '__main__':
    main()
