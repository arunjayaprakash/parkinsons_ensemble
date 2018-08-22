#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 00:20:50 2018

@author: arunjp
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import gc

import warnings; warnings.filterwarnings('ignore')

user_file_list = os.listdir('data/ArchivedUsers/')
user_set_v1 = set(map(lambda x: x[5: 15], user_file_list)) # [5: 15] to return just the user IDs


tappy_file_list = os.listdir('data/TappyData/')
user_set_v2 = set(map(lambda x: x[: 10], tappy_file_list)) # [: 10] to return just the user IDs


user_set = user_set_v1.intersection(user_set_v2) # contains IDs present in tapyp data and user data

#READING USER DATA
def read_user_file(file_name):
    f = open('data/ArchivedUsers/' + file_name)
    data = [line.split(': ')[1][: -1] for line in f.readlines()]
    f.close()

    return data

files = os.listdir('data/ArchivedUsers/')

columns = [
    'BirthYear', 'Gender', 'Parkinsons', 'Tremors', 'DiagnosisYear',
    'Sided', 'UPDRS', 'Impact', 'Levadopa', 'DA', 'MAOB', 'Other'
]

user_df = pd.DataFrame(columns=columns) # empty Data Frame for now

for user_id in user_set:
    temp_file_name = 'User_' + user_id + '.txt' # tappy file names have the format of `User_[UserID].txt`
    if temp_file_name in files: # check to see if the user ID is in our valid user set
        temp_data = read_user_file(temp_file_name)
        user_df.loc[user_id] = temp_data # adding data to our DataFrame
        
#CLEANING DATA

#int coercion

user_df['BirthYear'] = pd.to_numeric(user_df['BirthYear'], errors='coerce')
user_df['DiagnosisYear'] = pd.to_numeric(user_df['DiagnosisYear'], errors='coerce')

#binary
user_df = user_df.rename(index=str, columns={'Gender': 'Female'}) # renaming `Gender` to `Female`
user_df['Female'] = user_df['Female'] == 'Female' # change string data to boolean data
user_df['Female'] = user_df['Female'].astype(int) # change boolean data to binary data

str_to_bin_columns = ['Parkinsons', 'Tremors', 'Levadopa', 'DA', 'MAOB', 'Other'] # columns to be converted to binary data

for column in str_to_bin_columns:
    user_df[column] = user_df[column] == 'True'
    user_df[column] = user_df[column].astype(int)
    
#ONE HOT ENCODING
    # prior processing for `Impact` column
user_df.loc[
    (user_df['Impact'] != 'Medium') &
    (user_df['Impact'] != 'Mild') &
    (user_df['Impact'] != 'Severe'), 'Impact'] = 'None'

to_dummy_column_indices = ['Sided', 'UPDRS', 'Impact'] # columns to be one-hot encoded
for column in to_dummy_column_indices:
    user_df = pd.concat([
        user_df.iloc[:, : user_df.columns.get_loc(column)],
        pd.get_dummies(user_df[column], prefix=str(column)),
        user_df.iloc[:, user_df.columns.get_loc(column) + 1 :]
    ], axis=1)
    

    
#READING TAPPY DATA
def read_tappy(file_name):
    df = pd.read_csv(
        'data/TappyData/' + file_name,
        delimiter = '\t',
        index_col = False,
        names = ['UserKey', 'Date', 'Timestamp', 'Hand', 'Hold time', 'Direction', 'Latency time', 'Flight time']
    )

    df = df.drop('UserKey', axis=1)

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%y%M%d').dt.date

    # converting time data to numeric
    #print(df[df['Hold time'] == '0105.0EA27ICBLF']) # for 0EA27ICBLF_1607.txt
    for column in ['Hold time', 'Latency time', 'Flight time']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(axis=0)

    # cleaning data in Hand
    df = df[
        (df['Hand'] == 'L') |
        (df['Hand'] == 'R') |
        (df['Hand'] == 'S')
    ]

    # cleaning data in Direction
    df = df[
        (df['Direction'] == 'LL') |
        (df['Direction'] == 'LR') |
        (df['Direction'] == 'LS') |
        (df['Direction'] == 'RL') |
        (df['Direction'] == 'RR') |
        (df['Direction'] == 'RS') |
        (df['Direction'] == 'SL') |
        (df['Direction'] == 'SR') |
        (df['Direction'] == 'SS')
    ]

    direction_group_df = df.groupby('Direction').mean()
    del df; gc.collect()
    direction_group_df = direction_group_df.reindex(['LL', 'LR', 'LS', 'RL', 'RR', 'RS', 'SL', 'SR', 'SS'])
    direction_group_df = direction_group_df.sort_index() # to ensure correct order of data
    
    return direction_group_df.values.flatten() # returning a numppy array

def process_user(user_id, filenames):
    running_user_data = np.array([])

    for filename in filenames:
        if user_id in filename:
            running_user_data = np.append(running_user_data, read_tappy(filename))
    
    running_user_data = np.reshape(running_user_data, (-1, 27))
    return np.nanmean(running_user_data, axis=0) # ignoring NaNs while calculating the mean

filenames= os.listdir('data/TappyData/')

column_names = [first_hand + second_hand + '_' + time for first_hand in ['L', 'R', 'S'] for second_hand in ['L', 'R', 'S'] for time in ['Hold time', 'Latency time', 'Flight time']]

user_tappy_df = pd.DataFrame(columns=column_names)
c=0
for user_id in user_df.index:
    user_tappy_data = process_user(str(user_id), filenames)
    user_tappy_df.loc[user_id] = user_tappy_data
    print(c)
    c=c+1

# some preliminary data cleaning
user_tappy_df = user_tappy_df.fillna(0)
user_tappy_df[user_tappy_df < 0] = 0 

#combining
combined_user_df = pd.concat([user_df, user_tappy_df], axis=1)

#combined_user_df.to_csv('consolidated.csv')

#COMMON PREPROCESSING

X = combined_user_df.drop(['BirthYear','DiagnosisYear','Parkinsons'], axis =1)

y = combined_user_df.iloc[:,2].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#---------------------LINEAR REGRESSION--------------------------------------------------------------------------------

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(y_test, y_pred)





