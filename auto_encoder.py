import tensorflow as tf
import keras, keras.layers as L
import numpy as np
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# suppose:
IMG_SHAPE = (32, 32, 3)


def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))


def build_pca_autoencoder(img_shape, code_size):
    """
    Here we define a simple linear autoencoder as described above.
    We also flatten and un-flatten data to be compatible with image shapes
    """

    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Flatten())                  #flatten image to vector
    encoder.add(L.Dense(code_size))           #actual encoder

    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(np.prod(img_shape)))  #actual decoder, height*width*3 units
    decoder.add(L.Reshape(img_shape))         #un-flatten

    return encoder, decoder


def visualize(img, encoder, decoder):
    """Draws original, encoded and decoded images"""
    code = encoder.predict(img[None])[0]  # img[None] is the same as img[np.newaxis, :]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1, 3, 1)
    plt.title("Original")
    show_image(img)
    plt.subplot(1, 3, 2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()


def build_deep_autoencoder(img_shape, code_size):
    """PCA's deeper brother. See instructions above. Use `code_size` in layer definitions."""

    # encoder
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))

    encoder.add(L.Conv2D(filters=32, kernel_size=3, padding='same', activation='elu'))
    encoder.add(L.MaxPooling2D(pool_size=2))
    encoder.add(L.Conv2D(filters=64, kernel_size=3, padding='same', activation='elu'))
    encoder.add(L.MaxPooling2D(pool_size=2))
    encoder.add(L.Conv2D(filters=128, kernel_size=3, padding='same', activation='elu'))
    encoder.add(L.MaxPooling2D(pool_size=2))
    encoder.add(L.Conv2D(filters=256, kernel_size=3, padding='same', activation='elu'))
    encoder.add(L.MaxPooling2D(pool_size=2))

    encoder.add(L.Flatten())

    encoder.add(L.Dense(code_size))
    # decoder
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(2*2*256))
    decoder.add(L.Reshape((2, 2, 256)))
    decoder.add(L.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='elu', padding='same')
                )
    decoder.add(L.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='elu', padding='same')
                )
    decoder.add(L.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='elu', padding='same')
                )
    decoder.add(L.Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=2, activation=None, padding='same')
                )

    return encoder, decoder


s = tf.Session()

encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=32)

inp = L.Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = keras.models.Model(inputs=inp, outputs=reconstruction)
autoencoder.compile(optimizer="adamax", loss='mse')

# example of training
# autoencoder.fit(x=X_train, y=X_train, epochs=25,
#                                  validation_data=[X_test, X_test],
#                                  verbose=0
#


