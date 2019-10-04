import time
from typing import Tuple, Iterable

import cv2
import numpy
from keras.layers import Input, Conv2D, Flatten, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model

import caracteristicas.Keyframes as Keyframes


def model(
        input_shape: Tuple[int, int, int] = (256, 256, 3), cells: int = 3, convs: int = 2,
        kernel_size: int = 3, filters: int = 16, activation: str = 'relu', pool_size: int = 2,
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
    conv_args['filters'] = input_shape[2]
    recontruction = Conv2D(**conv_args)(deconv)

    # complete model
    autoencoder = Model(inputs=input_img, outputs=recontruction)
    autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error')
    autoencoder.summary()

    # encoder
    encoder = Model(inputs=input_img, outputs=encoded)

    return autoencoder, encoder


class AutoEncoder:
    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (256, 256, 3), cells: int = 3, convs: int = 2,
            kern_size: int = 3, filters: int = 16, act: str = 'relu', pool_size: int = 2,
            autoencoder=None, encoder=None,
    ):

        if autoencoder is None and encoder is None:
            autoencoder, encoder = model(
                input_shape=input_shape, cells=cells, convs=convs,
                kernel_size=kern_size, filters=filters, activation=act, pool_size=pool_size
            )

        self.autoencoder = autoencoder
        self.encoder = encoder

        self.input_shape = self.autoencoder.get_input_shape_at(0)
        self.output_size = self.encoder.output_shape[1]

    def train(self, data: numpy.ndarray, epochs: int, batch_size: int = 32, validation_split: float = 0.2,
              verbose: int = 2):

        t = time.time()

        self.autoencoder.fit(
            data, data,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose)

        if verbose > 0:
            print(f'\nTraining took {time.time() - t:.2f} seconds')
        return

    def encode(self, data: numpy.ndarray):
        return self.encoder.predict(data)

    def save(self, name: str = 'model'):
        self.autoencoder.save(f'{name}-autoencoder.h5')
        self.encoder.save(f'{name}-encoder.h5')
        return

    def encode_decode(self, data: numpy.ndarray):
        return self.autoencoder.predict(data)

    def test(self, data: numpy.ndarray):
        # obtain reconstructed images
        predictions = self.encode_decode(data)

        # convert to ints
        predictions = (predictions * 255).astype('uint8')
        original = (data * 255).astype('uint8')

        # resize
        def resize(img: numpy.ndarray):
            return cv2.resize(img, dsize=(593, 336), interpolation=cv2.INTER_AREA)

        resized_p = [resize(predictions[i]) for i in range(len(predictions))]
        resized_o = [resize(original[i]) for i in range(len(predictions))]

        i = 0
        while True:
            cv2.imshow(f'autoencoder reconstruction', cv2.vconcat([resized_o[i], resized_p[i]]))

            key = cv2.waitKey(0)
            if key & 0xff == ord('a'):
                i = max(i - 1, 0)
            elif key & 0xff == ord('d'):
                i = min(i + 1, len(predictions) - 1)
            elif key == 27 or key == -1:  # esc
                break

        return


def load_autoencoder(name: str = 'model'):
    autoencoder = load_model(f'{name}-autoencoder.h5')
    encoder = load_model(f'{name}-encoder.h5')
    return AutoEncoder(autoencoder=autoencoder, encoder=encoder)


def load_frames(videos: Iterable[str] = ('003',), shape: Tuple[int, int, int] = (64, 64, 3), gray: bool = False):
    frames = []

    for video in videos:
        print(f'extracting frames from video {video}')
        for _, frame in Keyframes.n_frames_per_fps(f'../../videos/Shippuden/{video}.mp4', n=1):

            if gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames.append(
                numpy.reshape(
                    cv2.resize(frame, dsize=shape[:2], interpolation=cv2.INTER_AREA), shape))

    return numpy.array(frames)


def main():

    # auto encoder config
    shape = (64, 64, 3)
    cells = 4
    convs = 2

    # train config
    epochs = 50
    batch_size = 32

    # extract frames
    videos = ['003', ]  # '098', '253', '345', ]
    frames = load_frames(videos, shape)

    # shuffle and normalize to [0, 1]
    frames = frames.astype('float32') / 255
    numpy.random.shuffle(frames)

    # create model and train
    autoencoder = AutoEncoder(input_shape=shape, cells=cells, convs=convs, act='tanh')
    autoencoder.train(frames, epochs=epochs, batch_size=batch_size)
    # autoencoder.save()

    # load model
    # autoencoder = load_autoencoder()
    print(f'descriptor size: {autoencoder.output_size}')

    autoencoder.test(frames[:30])
    return


if __name__ == '__main__':
    numpy.random.seed(1209)
    main()
