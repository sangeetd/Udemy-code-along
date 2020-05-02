
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble._hist_gradient_boosting import loss

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('E:\\pythonDatasets\\irisData\\iris-data.csv', header=None)

# print (df.head())

X = df.iloc[:, :-1]
X = np.array(X).astype(np.float)

y = df.iloc[:, -1]
print ('typw of X from dataframe : ', type(X))
print (X, ' -- ', y[0])

outputs_vals, outputs_ints = np.unique(y, return_inverse=True)
# print (outputs_ints)
print (outputs_vals)

y_cat = to_categorical(outputs_ints)
print(y_cat)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

print (type(X_train))
print(X_train.shape)
print(y_train.shape)

scalar = MinMaxScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

# plt.plot(X_train)
# plt.show()
def doModeling():

    model = Sequential()
    model.add(Dense(25, activation='sigmoid', input_shape=(X_train.shape[1],)))
    # model.add(Dropout(0.23))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    model_history = model.fit(X_train, y_train, epochs=150,
              validation_data=(X_test, y_test))

    metric = model.evaluate(X_test, y_test)

    model.save('iris-data-model.h5')

    # Plot training & validation accuracy values
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()



def doPrediction():
    model = load_model('iris-data-model.h5')
    xtest, ytest = visualizeTestData()
    pred = model.predict(xtest)
    print ('pred : ', pred)

def visualizeTestData():
    test = df.to_numpy()
    # print (test[0])
    xtest = test[132][:-1]
    xtest = np.asarray(xtest.reshape(1, 4), dtype='float32')
    xtest = scalar.transform(xtest)
    ytest = test[132][-1:]
    return xtest, ytest

# doModeling()
xtest, ytest = visualizeTestData()
print (xtest, ' is ', ytest)
doPrediction()