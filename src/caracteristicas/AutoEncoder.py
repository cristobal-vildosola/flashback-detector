from keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Flatten, MaxPooling2D, Reshape, UpSampling2D
from keras.models import Model
import numpy

descriptor_size = 192  # 8 * 8 * 3
input_img = Input(shape=(256, 256, 3))

# conv parameters
act = 'relu'
pad = 'same'
stri = 2
kern = 3
pool_size = 2

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
deconv1 = Conv2DTranspose(filters=8, kernel_size=kern, strides=stri, padding=pad, activation=act)(up1)  # (16, 16, 8)

up2 = UpSampling2D(size=pool_size)(deconv1)  # (32, 32, 8)
deconv2 = Conv2DTranspose(filters=4, kernel_size=kern, strides=stri, padding=pad, activation=act)(up2)  # (64, 64, 4)

up3 = UpSampling2D(size=pool_size)(deconv2)  # (128, 128, 4)
decoded = Conv2DTranspose(filters=3, kernel_size=kern, strides=stri, padding=pad, activation=act)(up3)  # (256, 256, 3)

# complete model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()

# encoder
encoder = Model(input_img, encoded)

"""
autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=kernel_size56,
                shuffle=True,
                validation_data=(x_test, x_test))
"""
