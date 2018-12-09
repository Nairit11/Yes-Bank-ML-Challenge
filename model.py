#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 07:50:11 2018

@author: nairit-11
"""

import pandas as pd
import numpy as np 

# Importing the Training Dataset
data=pd.read_csv('Yes_Bank_Training.csv',header=0)
data=data.dropna()

# Dropping non-important columns
data=data.drop(labels="serial_number",axis=1)
data=data.drop(labels="date",axis=1)
data=data.drop(labels="month_of_year",axis=1)

# Mapping binary column to 0 and 1
outcome={"no":0, "yes":1}
data['outcome'] = data['outcome'].map(outcome)
data['has_default'] = data['has_default'].map(outcome)
data['housing_status'] = data['housing_status'].map(outcome)
data['previous_loan'] = data['previous_loan'].map(outcome)

# One-hot encoding categorical columns
one_hot = pd.get_dummies(data['poutcome_of_campaign'])
data = data.drop('poutcome_of_campaign',axis = 1)
data = data.join(one_hot)
data=pd.get_dummies(data=data, columns=['phone_type','job_description', 'marital_status','education_details'])

# Separate the target variable
Y_train=data["outcome"]
data = data.drop('outcome',axis = 1)

# Importing and Preprocessing test data
test=pd.read_csv('Yes_Bank_Test.csv',header=0)
test=test.drop(labels="serial_number",axis=1)
test=test.drop(labels="date",axis=1)
test=test.drop(labels="month_of_year",axis=1)
outcome={"no":0, "yes":1}
test['has_default'] = test['has_default'].map(outcome)
test['housing_status'] = test['housing_status'].map(outcome)
test['previous_loan'] = test['previous_loan'].map(outcome)
one_hot = pd.get_dummies(test['poutcome_of_campaign'])
test = test.drop('poutcome_of_campaign',axis = 1)
test = test.join(one_hot)
test=pd.get_dummies(data=test, columns=['phone_type','job_description', 'marital_status','education_details'])

# Applying Logistic Regression
# Training the model
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs').fit(data, Y_train)
# Predictions
results=clf.predict(test)
results = pd.DataFrame({'outcome':results[:]})
# Creating submission file
sub=pd.read_csv('sample_submission.csv',header=0)
outcome={0:"no", 1:"yes"}
sub['outcome'] = results["outcome"].map(outcome)
sub.to_csv('log_class.csv')
# Score: 75.269134

# Applying Randon Forest
# Training the model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=500, max_depth=30, random_state=0).fit(data, Y_train)
# Predictions
results=clf.predict(test)
results = pd.DataFrame({'outcome':results[:]})
# Creating submission file
sub=pd.read_csv('sample_submission.csv',header=0)
outcome={0:"no", 1:"yes"}
sub['outcome'] = results["outcome"].map(outcome)
sub.to_csv('random_for.csv')
#Score: 75.1364
print(clf.feature_importances_)

# Again with important features
clf2 = RandomForestClassifier(n_estimators=500, max_depth=30, random_state=0).fit(data[data.columns[clf.feature_importances_>0.01]], Y_train)
# Predictions
results=clf2.predict(test[data.columns[clf.feature_importances_>0.01]])
results = pd.DataFrame({'outcome':results[:]})
# Creating submission file
sub=pd.read_csv('sample_submission.csv',header=0)
outcome={0:"no", 1:"yes"}
sub['outcome'] = results["outcome"].map(outcome)
sub.to_csv('random_for_imp.csv')
#Score: 75.6083

# SGDClassifier
from sklearn import linear_model
clf = linear_model.SGDClassifier(max_iter=1000).fit(data,Y_train)
results=clf.predict(test)
results = pd.DataFrame({'outcome':results[:]})
# Creating submission file
sub=pd.read_csv('sample_submission.csv',header=0)
outcome={0:"no", 1:"yes"}
sub['outcome'] = results["outcome"].map(outcome)
sub.to_csv('sgdc.csv')
#Score: 76.2646

# LinearSVC with feature selection
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(data, Y_train)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(data)
lsvc2 = LinearSVC(C=0.01, penalty="l1", dual=False).fit(data[data.columns[abs(lsvc.coef_[0])>0]], Y_train)
results=lsvc2.predict(test[data.columns[abs(lsvc.coef_[0])>0]])
results = pd.DataFrame({'outcome':results[:]})
# Creating submission file
sub=pd.read_csv('sample_submission.csv',header=0)
outcome={0:"no", 1:"yes"}
sub['outcome'] = results["outcome"].map(outcome)
sub.to_csv('svc.csv')
# Score: 74.8857