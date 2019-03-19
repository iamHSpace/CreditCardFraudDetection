#import required libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#import dataset
df = pd.read_csv('/home/hspace/Desktop/Credit Card Fraud/creditcard.csv')

#preprocessing 
feature_names =  df.iloc[:,1:30].columns
target = df.iloc[:1,30:].columns

data_features = df[feature_names]
data_target = df[target]

# 70:30 splitting for train and test data
x_train, x_test, y_train, y_test = train_test_split(data_features,data_target, train_size=0.7, test_size=0.3, random_state=1)

#modeling 
model = LogisticRegression()
model.fit(x_train,y_train.values.ravel())

#prediction
pred = model.predict(x_test)
print(pred)