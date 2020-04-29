
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow_core.python.training import optimizer
from sklearn.metrics import mean_squared_error,mean_absolute_error, explained_variance_score

df = pd.read_csv('E:\\pythonDatasets\\udemyCourseResources\\TF_2_Notebooks_and_Data\\DATA\\kc_house_data.csv')
print (df.shape)

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)

df = df.drop('date', axis=1)
df = df.drop('id', axis=1)
df = df.drop('zipcode', axis=1)

X = df.drop('price', axis=1).values
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(x=X_train,
          y=y_train,
          validation_data=(X_test, y_test),
          batch_size=128,
          epochs=400)


# loss_comparision = pd.DataFrame(model.history.history)
#
# loss_comparision.plot()
# plt.show()

y_pred = model.predict(X_test)


# print (mean_squared_error(y_test, y_pred))
#
# print (mean_absolute_error(y_test, y_pred))
#
# print (df['price'].describe())
#
# print (explained_variance_score(y_test,y_pred))

single_house_original_price = df.iloc[0]
print ('original price: ', single_house_original_price)
single_house = df.drop('price', axis=1).iloc[0]
single_house = single_house.values.reshape(-1, 19)
single_house = scaler.transform(single_house)
#print(single_house)

single_house_pred = model.predict(single_house)
print ('expected price: ', single_house_pred)

print ('Hello world')