import time
from typing import Tuple

import cv2
import numpy
from keras.layers import Input, Conv2D, Flatten, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model

from keyframes import FPSReductionKS
from features.FeatureExtractor import FeatureExtractor


class AutoEncoderFE(FeatureExtractor):
    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (256, 256, 3),
            cells: int = 3,
            convs: int = 2,
            kern_size: int = 3,
            filters: int = 16,
            activation: str = 'tanh',
            pool_size: int = 2,
            output_activation: str = 'tanh',
            autoencoder: Model = None,
            encoder: Model = None,
            model_name: str = 'model',
            dummy: bool = False,
    ):
        if not dummy:
            if autoencoder is None and encoder is None:
                self.create_model(
                    input_shape=input_shape, cells=cells, convs=convs,
                    kernel_size=kern_size, filters=filters, activation=activation,
                    pool_size=pool_size, output_activation=output_activation
                )

            self.autoencoder = autoencoder
            self.encoder = encoder

            self.input_shape = self.autoencoder.get_input_shape_at(0)[1:]
            self.output_size = self.encoder.output_shape[1]

        self.model_name = model_name

    def create_model(
            self,
            input_shape: Tuple[int, int, int] = (256, 256, 3),
            cells: int = 3,
            convs: int = 2,
            kernel_size: int = 3,
            filters: int = 16,
            activation: str = 'tanh',
            pool_size: int = 2,
            output_activation: str = 'tanh',
    ):
        conv_args = {
            'filters': filters,
            'kernel_size': kernel_size,
            'activation': activation,
            'padding': 'same',
        }

        input_img = Input(shape=input_shape)

        # convolutional encoding
        pool = input_img
        for _ in range(cells):

            conv = Conv2D(**conv_args)(pool)
            for _ in range(1, convs):
                conv = Conv2D(**conv_args)(conv)

            pool = MaxPooling2D(pool_size=pool_size, padding='same')(conv)

        # flatten encoding
        encoded = Flatten()(pool)

        # deconvolutional decoding
        deconv = pool
        for i in range(cells):
            up = UpSampling2D(size=pool_size)(deconv)

            deconv = Conv2D(**conv_args)(up)
            for _ in range(1, convs):
                deconv = Conv2D(**conv_args)(deconv)

        # return to original number of channels
        recontruction = Conv2D(
            filters=input_shape[2],
            kernel_size=kernel_size,
            activation=output_activation,
            padding='same',
        )(deconv)

        # complete model
        self.autoencoder = Model(inputs=input_img, outputs=recontruction)
        self.autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error')

        # encoder
        self.encoder = Model(inputs=input_img, outputs=encoded)
        return

    def adapt_input(self, data) -> numpy.ndarray:
        t0 = time.time()
        new_data = numpy.zeros((len(data), *self.input_shape), dtype='float32')

        for i in range(len(data)):
            new_data[i] = cv2.resize(data[i], dsize=self.input_shape[:2], interpolation=cv2.INTER_AREA) / 255

        duration = time.time() - t0
        print(f'resizing {len(data)} frames took {duration:.2f} seconds')

        return new_data

    def train(
            self,
            data: numpy.ndarray,
            epochs: int,
            batch_size: int = 32,
            validation_split: float = 0.2,
            verbose: bool = True,
    ):

        t = time.time()
        adapted = self.adapt_input(data)

        self.autoencoder.fit(
            adapted, adapted,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose)

        if verbose:
            print(f'\nTraining took {time.time() - t:.2f} seconds')
        return

    def encode(self, data: numpy.ndarray) -> numpy.ndarray:
        return self.encoder.predict(data)

    def encode_decode(self, data: numpy.ndarray) -> numpy.ndarray:
        return self.autoencoder.predict(data)

    def extract_features(self, data) -> numpy.ndarray:
        return (self.encode(self.adapt_input(data)) * 127).astype('int8')

    def test(self, data: numpy.ndarray):
        # obtain reconstructed images
        reconstructed = self.encode_decode(self.adapt_input(data))
        reconstructed = (reconstructed.clip(0, 1) * 255).astype('uint8')

        # resize
        def resize(img: numpy.ndarray):
            return cv2.resize(img, dsize=(593, 336), interpolation=cv2.INTER_AREA)

        reconstructed = [resize(reconstructed[i]) for i in range(len(reconstructed))]
        original = [resize(data[i]) for i in range(len(reconstructed))]

        i = 0
        while True:
            cv2.imshow(f'autoencoder reconstruction', cv2.vconcat([reconstructed[i], original[i]]))

            key = cv2.waitKey(0)
            if key & 0xff == ord('a'):
                i = max(i - 1, 0)
            elif key & 0xff == ord('d'):
                i = min(i + 1, len(reconstructed) - 1)
            elif key == 27 or key == -1:  # esc
                break

        return

    def save(self):
        self.autoencoder.save(f'{self.model_name}-autoencoder.h5')
        self.encoder.save(f'{self.model_name}-encoder.h5')
        return

    def descriptor_size(self) -> int:
        return self.output_size

    def name(self) -> str:
        return f'AE_{self.model_name}'

    @staticmethod
    def load_autoencoder(model_name: str = 'model') -> 'AutoEncoderFE':
        autoencoder = load_model(f'{model_name}-autoencoder.h5')
        encoder = load_model(f'{model_name}-encoder.h5')

        # remove / from the name to avoid nested directories
        model_name = model_name.split('/')[-1]
        print(f'model {model_name} loaded')
        return AutoEncoderFE(autoencoder=autoencoder, encoder=encoder, model_name=model_name)


def main():
    numpy.random.seed(1209)
    load = True

    # input shape
    shape = (64, 64, 3)

    # select keyframes and shuffle
    selector = FPSReductionKS(n=1)
    frames, _, _ = selector.select_keyframes('../../videos/Shippuden_low/003.mp4')
    numpy.random.shuffle(frames)

    if load:
        autoencoder = AutoEncoderFE.load_autoencoder(model_name='model')
        shape = autoencoder.input_shape
        print(f'input size: {shape}')

    else:
        # best: tanh-tanh, second: tanh-relu
        autoencoder = AutoEncoderFE(
            input_shape=shape,
            cells=4,
            convs=2,
            activation='tanh',
            output_activation='tanh',
            model_name='model2'
        )

        # autoencoder.train(frames, epochs=50)
        # autoencoder.save()

    print(f'descriptor size: {autoencoder.output_size}')
    autoencoder.test(frames[:30])
    return


if __name__ == '__main__':
    main()
