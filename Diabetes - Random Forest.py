#!/usr/bin/env python
# coding: utf-8

# # Diabetes Prediction : Random Forest


#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Import dataset
data = pd.read_csv( r"C:\Users\Dell\Desktop\diabetes.csv")


#Check the size of datset
data.shape


#Displaying first 5 dataset entries
data.head(5)


#Check if any null value is present
data.isnull().values.any()


#Correlation
import seaborn as sns
import matplotlib.pyplot as plt
cormat = data.corr()
top_corr_features = cormat.index
plt.figure(figsize=(20,20))

#Plotting heat map
g = sns.heatmap(data[top_corr_features].corr(), annot = True)


data.corr()


#Checking True and false counts of diabetes
true_count = len(data.loc[data['Outcome'] == True ])
false_count = len(data.loc[data['Outcome'] == False ])
(true_count, false_count)


#Splitting Training and Testing datset
from sklearn.model_selection import train_test_split
features_column = [ 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age' ]
prediction_class = ['Outcome']

X = data[features_column].values
Y = data[prediction_class].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 10)

#Check how many zero values
print("Total no of rows : {0}".format(len(data)))
print("Number of missing rows in Pregnancies : {0}" .format(len(data.loc[data['Pregnancies'] ==0 ])))
print("Number of missing rows in Glucose : {0}" .format(len(data.loc[data['Glucose'] ==0 ])))
print("Number of missing rows in BloodPressure : {0}" .format(len(data.loc[data['BloodPressure'] ==0 ])))
print("Number of missing rows in SkinThickness : {0}" .format(len(data.loc[data['SkinThickness'] ==0 ])))
print("Number of missing rows in Insulin : {0}" .format(len(data.loc[data['Insulin'] ==0 ])))
print("Number of missing rows in BMI : {0}" .format(len(data.loc[data['BMI'] ==0 ])))
print("Number of missing rows in DiabetesPedigreeFunction : {0}" .format(len(data.loc[data['DiabetesPedigreeFunction'] ==0 ])))
print("Number of missing rows in Age : {0}" .format(len(data.loc[data['Age'] ==0 ])))


#Fixing Null values
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan, strategy='mean')

X_train = impute.fit_transform(X_train)
X_test = impute.fit_transform(X_test)


#Applying Random Forest Algorithm
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, Y_train.ravel())
#Predicting Accuracy of model
predict = model.predict(X_test)
from sklearn import metrics
print("Accuracy of Random Forest = {0:.3f}".format(metrics.accuracy_score(Y_test, predict)))
