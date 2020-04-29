
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('E:\\pythonDatasets\\udemyCourseResources\\TF_2_Notebooks_and_Data\\DATA\\cancer_classification.csv')

print (df.shape)
print (df.columns)
print (df.head())

X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values

print ('X shape: ',X.shape, 'y shape: ', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print ('train shape(Data points, Feature): ', X_train)

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model = Sequential()

#X_train feature = 30
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5))

#For binary classifcation sigmoid
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(x=X_train,
          y=y_train,
          validation_data=(X_test, y_test),
          epochs=600,
          callbacks=[early_stop])

# loss_comparision = pd.DataFrame(model.history.history)
# loss_comparision.plot()
# plt.show()

prediction = model.predict_classes(X_test)

print ('Classification report: ', classification_report(y_test, prediction))

print ('Confusion matrix: ', confusion_matrix(y_test, prediction))