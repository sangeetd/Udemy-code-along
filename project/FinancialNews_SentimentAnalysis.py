import numpy as np
import pandas as pd
import project.Text_cleaning_techniques as textClean
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

# refining X input
max_length = 10
X_processed = []
i=0
for str in X:
    output = textClean.toTokenizedWithoutStopword(
            textClean.punctuationRemoved(
                textClean.numberRemoved(
                    textClean.stringToLower(str[0])
                              )
            )
        )

    # print(output)
    X_processed.append(output)
    if len(output) > max_length:
        max_length = len(output)

print (type(X_processed))
print (X_processed[0])
max_vocab = len(X_processed)
print (max_vocab)
print ('max length : ', max_length)

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(X_processed)
sequences = tokenizer.texts_to_sequences(X_processed)

X_train, X_test, y_train, y_test = train_test_split(sequences, y_cat, test_size=0.3, random_state=42)

print (type(X_train))
print (len(X_train))
print (X_train)

print (type(y_train))
print (len(y_train))
print (y_train)

print ('sequence : ------------')
print (sequences)

from tensorflow.keras.preprocessing.sequence import pad_sequences
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=max_length)

print ('word index : ------------')
print (word_index)

print ('data : ------------')
print (data)


print ('model creation : ')
embedding_mat_columns=32
model = Sequential()
model.add(Embedding(input_dim=max_vocab,
                    output_dim=embedding_mat_columns,
                    input_length=max_length))
model.add(LSTM(units=embedding_mat_columns, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['acc'])
model.summary()

model.fit(X_train, y_train, epochs=20)

model.save('financialNews-Sentiment.h5')

acc = model.evaluate(X_test, y_test)
print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc[0],acc[1]))