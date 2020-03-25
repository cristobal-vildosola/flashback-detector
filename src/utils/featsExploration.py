import matplotlib.pyplot as plt
import numpy as np

import keyframes.KeyframeSelector as Keyframes
from features.AutoEncoder import AutoEncoderFE
from features.ColorLayout import ColorLayoutFE
from utils.files import group_features


def main():
    selectors = [
        Keyframes.FPSReductionKS(n=6),
        Keyframes.MaxHistDiffKS(frames_per_window=2),
        Keyframes.ThresholdHistDiffKS(threshold=1.3),
    ]
    extractors = [
        ColorLayoutFE(),
        AutoEncoderFE.load_autoencoder(name='../features/model'),
    ]

    for selector in selectors:
        for extractor in extractors:
            tags, features = group_features(selector=selector, extractor=extractor)

            x = range(features.shape[1])
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            mins = np.min(features, axis=0)
            maxs = np.max(features, axis=0)

            plt.figure()
            plt.title(f'{selector.name()} - {extractor.name()}')

            plt.errorbar(x, mean, std, fmt='b', ecolor='r', capsize=3)
            plt.plot(x, mins, 'g')
            plt.plot(x, maxs, 'g')

            plt.show()
    return


if __name__ == '__main__':
    main()
