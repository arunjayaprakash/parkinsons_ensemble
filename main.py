#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 10:12:43 2018

@author: arunjp
"""
import pandas as pd
import numpy as np

dataset = pd.read_csv('consolidated.csv')
#---------------------LINEAR REGRESSION--------------------------------------------------------------------------------
X = dataset.drop(['BirthYear','DiagnosisYear','Parkinsons'], axis =1)
X = X.iloc[:,1:].values
y = dataset.iloc[:,2].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting LOGISTIC REGRESSOR to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# BACKWARD ELIMINATION
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((217,1)).astype(int), values= X, axis=1)
X_opt = X[:,:]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()