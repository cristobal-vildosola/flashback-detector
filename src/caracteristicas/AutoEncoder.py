import cv2
import numpy
from keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Flatten, MaxPooling2D, Reshape, UpSampling2D
from keras.models import Model

import caracteristicas.Keyframes as Keyframes


def model(input_shape=(256, 256, 3), descriptor_size=192, act='relu', kern=3, stri=2, pool_size=2, pad='same'):
    input_img = Input(shape=input_shape)  # (256, 256, 3)

    # convolutional encoding
    conv1 = Conv2D(filters=4, kernel_size=kern, strides=stri, padding=pad, activation=act)(input_img)  # (128, 128, 4)
    pool1 = MaxPooling2D(pool_size=pool_size, padding=pad)(conv1)  # (64, 64, 4)

    conv2 = Conv2D(filters=8, kernel_size=kern, strides=stri, padding=pad, activation=act)(pool1)  # (32, 32, 8)
    pool2 = MaxPooling2D(pool_size=pool_size, padding=pad)(conv2)  # (16, 16, 8)

    conv3 = Conv2D(filters=16, kernel_size=kern, strides=stri, padding=pad, activation=act)(pool2)  # (8, 8, 16)
    pool3 = MaxPooling2D(pool_size=pool_size, padding=pad)(conv3)  # (4, 4, 16)

    # flatten and dense
    flattened = Flatten()(pool3)  # (256)
    encoded = Dense(descriptor_size, activation=act)(flattened)  # (192)

    # reverse dense and reshape
    dense2 = Dense(numpy.prod(pool3._keras_shape[1:]), activation=act)(encoded)  # (256)
    reshaped = Reshape(pool3._keras_shape[1:])(dense2)  # (4, 4, 16)

    # deconvolutional decoding
    up1 = UpSampling2D(size=pool_size)(reshaped)  # (8, 8, 16)
    deconv1 = Conv2DTranspose(filters=8, kernel_size=kern, strides=stri, padding=pad, activation=act)(up1)
    # (16, 16, 8)

    up2 = UpSampling2D(size=pool_size)(deconv1)  # (32, 32, 8)
    deconv2 = Conv2DTranspose(filters=4, kernel_size=kern, strides=stri, padding=pad, activation=act)(up2)
    # (64, 64, 4)

    up3 = UpSampling2D(size=pool_size)(deconv2)  # (128, 128, 4)
    decoded = Conv2DTranspose(filters=3, kernel_size=kern, strides=stri, padding=pad)(up3)
    # (256, 256, 3)

    # complete model
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error')

    # encoder
    encoder = Model(inputs=input_img, outputs=encoded)

    return autoencoder, encoder


def main():
    frames = []
    for _, frame in Keyframes.n_frames_per_fps('../../videos/Shippuden/003.mp4', n=1):
        frames.append(cv2.resize(frame, dsize=(256, 256), interpolation=cv2.INTER_AREA))

    frames = numpy.array(frames)

    print(frames.shape)
    frames = frames.astype('float32') / 255

    numpy.random.seed(1209)
    numpy.random.shuffle(frames)

    autoencoder, encoder = model()
    autoencoder.fit(frames, frames,
                    validation_split=0.2,
                    epochs=50,
                    batch_size=64,
                    verbose=2)

    predictions = autoencoder.predict(frames[:5])

    original1 = cv2.resize((frames[0] * 255).astype('uint8'), dsize=(593, 336), interpolation=cv2.INTER_AREA)
    reconstructed1 = cv2.resize((predictions[0] * 255).astype('uint8'), dsize=(593, 336), interpolation=cv2.INTER_AREA)

    original2 = cv2.resize((frames[1] * 255).astype('uint8'), dsize=(593, 336), interpolation=cv2.INTER_AREA)
    reconstructed2 = cv2.resize((predictions[1] * 255).astype('uint8'), dsize=(593, 336), interpolation=cv2.INTER_AREA)

    cv2.imshow(f'reconstruction',
               cv2.hconcat(
                   [cv2.vconcat([original1, reconstructed1]),
                    cv2.vconcat([original2, reconstructed2])]))

    original1 = cv2.resize((frames[2] * 255).astype('uint8'), dsize=(593, 336), interpolation=cv2.INTER_AREA)
    reconstructed1 = cv2.resize((predictions[2] * 255).astype('uint8'), dsize=(593, 336), interpolation=cv2.INTER_AREA)

    original2 = cv2.resize((frames[3] * 255).astype('uint8'), dsize=(593, 336), interpolation=cv2.INTER_AREA)
    reconstructed2 = cv2.resize((predictions[3] * 255).astype('uint8'), dsize=(593, 336), interpolation=cv2.INTER_AREA)

    cv2.imshow(f'reconstruction2',
               cv2.hconcat(
                   [cv2.vconcat([original1, reconstructed1]),
                    cv2.vconcat([original2, reconstructed2])]))

    cv2.waitKey(0)

    return


if __name__ == '__main__':
    main()
