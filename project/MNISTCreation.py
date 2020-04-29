
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print ('shape: ', X_train.shape, y_train)

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

#noramlise b/w 0 or 1 probability as max value in image matrix is 255 only
X_train = X_train/255
X_test = X_test/255

single_image = X_train[0][:][:]
print (single_image)

# plt.imshow(single_image)
# plt.show()

#reshaping for conv to tell its 1 channel image
#batch size=60000, width, height, channel
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

print ('shape: ', X_train.shape)

early_stopping = EarlyStopping(monitor='val_loss', patience=1)

model = Sequential()

model.add(Conv2D(filters=64,
                 kernel_size=(4, 4),
                 input_shape=(28, 28, 1),
                 activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32,
                 kernel_size=(4, 4),
                 activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))

#output layer
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x=X_train, y=y_cat_train, epochs=15, validation_data=(X_test,y_cat_test), verbose=1,
          callbacks=[early_stopping])

model.save('mnist-model-19-04-2020.h5')

model_evaluation = model.evaluate(X_test, y_cat_test)
print ('model evaluation: ', model_evaluation)

#prediction
my_number = X_test[10]
my_number_is = y_test[10]

print ('number is : ', my_number_is)

prediction = model.predict_classes(my_number.reshape(1, 28, 28, 1))

print ('prediction came out: ', prediction)

loss_comparison = pd.DataFrame(model.history.history)

plt.plot(loss_comparison[['loss', 'val_loss']])
plt.show()