import numpy as np
import pandas as pd
import project.Text_cleaning_techniques
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('E:\\pythonDatasets\\sentimentAnalysisForFinancialNews\\all-data.csv',
                 header=None,
                 encoding='ISO-8859-1')

print (df.head())

print (df.shape)

#segregating dataset into X, y
X = df.iloc[:, -1:]
y = df.iloc[:, :-1]

print (X[:1])
print (y[:1])

X = X.to_numpy().reshape(df.shape[0], 1)

print (type(X))
print (X[0])
print (X.shape)

y = y.to_numpy().reshape(df.shape[0], 1)
print (type(y))
print (y[0])
print (y.shape)

y_vals, y_int = np.unique(y, return_inverse=True)
print (y_vals)

y_cat = to_categorical(y_int)

print (y_cat)

