
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import  tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

label_meaning = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(type(X_train))

print ('shape: ', X_train.shape, y_train.shape)

"""
print (X_train[0][:][:])

single_image = X_train[0][:][:]

# plt.imshow(single_image)
# plt.show()

# y_example = to_categorical(y_train)
# print (y_example.shape)

y_cat_train = to_categorical(y_train, num_classes=10)
y_cat_test = to_categorical(y_test, num_classes=10)

print ('shape: ', y_cat_train.shape)
print (y_cat_train[0])
# print (label_meaning[y_cat_train[y_cat_train==1.0]])

#noramlise b/w 0 or 1 probability as max value in image matrix is 255 only
X_train = X_train/255
X_test = X_test/255

single_image = X_train[0][:][:]
print (single_image)

# plt.imshow(single_image)
# plt.show()

early_stopping = EarlyStopping(monitor='val_loss', patience=1)

model = Sequential()

model.add(Conv2D(filters=128,
                 kernel_size=(4, 4),
                 input_shape=(32, 32, 3),
                 activation='relu'))
model.add(Dropout(0.5))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128,
                 kernel_size=(4, 4),
                 activation='relu'))
model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64,
                 kernel_size=(4, 4),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32,
                 kernel_size=(4, 4),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))

#output layer
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x=X_train, y=y_cat_train, epochs=150,
          validation_data=(X_test,y_cat_test),
          verbose=1,
          callbacks=[early_stopping])

print (model.summary())

model.save('cifar-model-21-04-2020.h5')

model_evaluation = model.evaluate(X_test, y_cat_test)
print ('model evaluation: ', model_evaluation)

"""

def load_imgtest(path_to_img):
    # max_dim = 150
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [32,32])
    print(img)
    img = img[tf.newaxis, :]
    return img

def test_trained_model(img_path):
    #load model for testing purpose
    test_model = load_model('cifar-model-21-04-2020.h5')
    train_img = load_imgtest(img_path)
    print(train_img.shape)
    prediction_ext = test_model.predict_classes(train_img)
    print('prediction_ext came out: ', label_meaning[prediction_ext[0]])

test_trained_model('E:\\pythonDatasets\\CIFAR-10-dataset\\CIFAR-10-test-external-images\\automobile_32.jpg')