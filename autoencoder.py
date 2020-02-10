
from keras.layers import Input, Dense
from keras.models import Model
encoding_dim = 32
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)




autoencoder = Model(input_img, decoded)

from keras.datasets import mnist


encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

encoder = Model(input_img, encoded)


autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')

import numpy as np
(x_train, y_train),(x_test, y_test) = mnist.load_data()
import matplotlib.pyplot as plt



encoded_input = Input(shape=(encoding_dim,))
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((len(x_train)), np.prod(x_train.shape[1:]))
x_test = x_test.reshape((len(x_test)), np.prod(x_test.shape[1:]))

autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

print(x_test[0].shape)
print(encoded_imgs[0].shape)
print(decoded_imgs[0].shape)

plt.figure("original")
plt.imshow(x_test[2].reshape(28,28))


plt.figure("encoded")
plt.imshow(encoded_imgs[2].reshape(1,32))

plt.figure("decoded")
plt.imshow(decoded_imgs[2].reshape(28,28))

plt.show()

# encoder = Model(input_img, encoded)
#
# encoded_input = Input(shape=(encoding_dim,))
# decoder_layer = autoencoder.layers[-1]
# decoder = Model(encoded_input, decoder_layer(encoded_input))
#
# encoder = Model(input_img, encoded)
#
# encoded_input = Input(shape=(encoding_dim,))