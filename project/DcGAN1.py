
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, Conv2D, Conv2DTranspose, Dropout, Reshape, BatchNormalization

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plt.imshow(X_train[0])
# plt.show()

X_train = X_train/255
X_train = X_train.reshape(-1,28, 28, 1) * 2.0 -1.0
# print (type(X_train))
# print (X_train.shape)
# print (X_train[0])

print('nim: ', X_train.min(), ' max: ', X_train.max())

only_zeros = X_train[y_train==0]

print(only_zeros.shape)

np.random.seed(42)
tf.random.set_seed(42)

coding_size = 100

generator = Sequential()
generator.add(Dense(7*7*128, input_shape=[coding_size]))
generator.add(Reshape([7, 7, 128]))
generator.add(BatchNormalization())
generator.add(Conv2DTranspose(64, kernel_size=5,
                              strides=2, padding='same',
                              activation='relu'))
generator.add(BatchNormalization())
generator.add(Conv2DTranspose(1, kernel_size=5,
                              strides=2, padding='same',
                              activation='tanh'))

discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=5, strides=2, padding='same'
                         ,activation=LeakyReLU(0.3),
                         input_shape=[28, 28, 1]))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same'
                         ,activation=LeakyReLU(0.3)))
discriminator.add(Dropout(0.5))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))

GAN = Sequential([generator, discriminator])

discriminator.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.trainable = False

GAN.compile(loss='binary_crossentropy', optimizer='adam')

batch_size = 32

my_data = only_zeros
dataset = tf.data.Dataset.from_tensor_slices(my_data).shuffle(buffer_size=1000)

print (type(dataset))

dataset = dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(1)

epochs = 100

#training

generator, discriminator = GAN.layers

for epoch in range(epochs):
    print (f'Currently on epoch {epoch+1}')

    i=0
    #for every batch in dataset
    for X_batch in dataset:
        i=i+1
        if i%20==0:
            print (f'\nCurrently on batch no. {i} of {len(my_data)//batch_size}')

            #create noise
            noise = tf.random.normal(shape=[batch_size, coding_size])
            #generate no. based just on noise input
            gen_image = generator(noise)

            #concatenate generate noised image with the real images
            X_fake_real = tf.concat([gen_image, tf.dtypes.cast(X_batch, tf.float32)], axis=0)

            #target set for fake and real images
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)

            discriminator.trainable = True

            discriminator.train_on_batch(X_fake_real, y1)

            noise = tf.random.normal(shape=[batch_size, coding_size])
            y2 = tf.constant([[1.]] * batch_size)

            discriminator.trainable = False

            GAN.train_on_batch(noise, y2)


print ("trining done...")

noise = tf.random.normal(shape=[10, coding_size])
print (noise.shape)

plt.imshow(noise)
plt.show()

noisy_images = generator(noise)

for image in noisy_images:
    plt.imshow(image.numpy().reshape(28, 28))
    plt.show()



