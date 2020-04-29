
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

df_info = pd.read_csv('E:\\pythonDatasets\\udemyCourseResources\\TF_2_Notebooks_and_Data\\DATA\\lending_club_info.csv',
                 index_col='LoanStatNew')

print (df_info.shape)
print (df_info.columns)
print (df_info.info())

print (df_info.loc['revol_util']['Description'])

def featureInfo(colNum):
    print (df_info.loc[colNum]['Description'])

featureInfo('mort_acc')

df = pd.read_csv('E:\\pythonDatasets\\udemyCourseResources\\TF_2_Notebooks_and_Data\\DATA\\lending_club_loan_two.csv')

# sns.countplot(x='loan_status', data=df)
# sns.distplot(df['loan_amnt'], kde=False)
# plt.show()

# print (df.corr())
# sns.heatmap(df.corr(), annot=True, cmap='viridis')
# plt.show()

print ('length: ', len(df))

#no of null data in dataframe
print (df.isnull().sum())

#percentage wise missing data as per the total data
print ((df.isnull().sum()/len(df))*100)

#from above output significant data point to be takeen care of for missing data
# are 'emp-title', 'emp_length', 'mort_acc'. other percentages are relatively less we can drop if required
# in following processes

#unique emp_title count
print (df['emp_title'].nunique())

#distinct count of enp_title
print (df['emp_title'].value_counts())

#since emp-title has 173105 emp title
# its not posisble to map/classify each one of them so they
#are not going to  be that much signiificant
df = df.drop('emp_title', axis=1)

#checking for emp__length
print (sorted(df['emp_length'].dropna().unique()))

#dropping emp_length because this has unconsiderable effect on loan status
df = df.drop('emp_length', axis=1)

#dropping title because this has unconsiderable effect on loan status
df = df.drop('title', axis=1)

#mort_acc fixing missing data
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']

def fillMortAcc(totalAcc, mortAcc):
    if np.isnan(mortAcc):
        return total_acc_avg[totalAcc]
    else:
        return mortAcc

df['mort_acc'] = df.apply(lambda x: fillMortAcc(x['total_acc'], x['mort_acc']), axis=1)

#drop rows with very less percentages of missing data
df.dropna()

#figuring out the column that are not numerical
print (df.select_dtypes(['object']).columns)

#processing loan_status
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1, 'Charged Off':0})

#looking for column - term
print (df['term'].value_counts())
#classifying term column into only its binary form (36 or 60)
df['term'] = df['term'].apply(lambda term: int(term[:3]))

#dropping grade because sub_grade is giving the same info
df = df.drop('grade', axis=1)
#get dummies out of sub_grade
dummies = pd.get_dummies(df['sub_grade'], drop_first=True)
#dropping the actual sub_grade
df = pd.concat([df.drop('sub_grade', axis=1), dummies], axis=1)

#processing home_ownership column
print (df['home_ownership'].value_counts())
#replacing NONE & ANY to OTHER
df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

#processing address column
df['zip_code'] = df['address'].apply(lambda address: address[-5:])


#perfoming same processing for column = verification_status, application_type, initial_list_status, purpose
dummies = pd.get_dummies(df[['verification_status', 'application_type',
                             'initial_list_status', 'purpose', 'home_ownership',
                             'zip_code']], drop_first=True)
#dropping the actual sub_grade
df = pd.concat([df.drop(['verification_status', 'application_type',
                         'initial_list_status', 'purpose', 'home_ownership',
                         'zip_code'], axis=1), dummies], axis=1)

df =df.drop('address', axis=1)

#proccessiing issue_d since while predicting the real world
#scenario we wont be having issue_d like thing because we are trying to predict
#wheather a already customer a genuine one or not
df = df.drop('issue_d', axis=1)

#processiing earliest_cr_line column
df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda yr: int(yr[-4:]))

#processing loan_status column to dropped off
df = df.drop('loan_status', axis=1)

#creating data variables
X = df.drop('loan_repaid', axis=1).values
y = df['loan_repaid'].values

print(df.columns)

#train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

print ('shape: ', X_train.shape, y_train.shape)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# print (pd.DataFrame(X_train[:10][:]).head())
# print (pd.DataFrame(y_train[:10][:]).head())

model = Sequential()

model.add(Dense(78, activation='relu'))
model.add(Dropout(0.20))

model.add(Dense(39, activation='relu'))
model.add(Dropout(0.20))

model.add(Dense(19, activation='relu'))

model.add(Dropout(0.20))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, y=y_train, epochs=25, batch_size=256, validation_data=(X_test, y_test))