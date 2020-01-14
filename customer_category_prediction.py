import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

##  load the csv file
df = pd.read_csv('teleCust1000t.csv')
df.head()

##  Data Visualization and Analysis
##  checking the number of categories in the data set
df['custcat'].value_counts()

##281 Plus Service, 266 Basic-service, 236 Total Service, and 217 E-Service customers
## checking the income histogram
df.hist(column='income', bins=50)

##  Feature set

df.columns

##  converting pandas data frame into numpy array
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]

##  the labels
y = df['custcat'].values
y[0:5]

##  Normalize Data
##  because data should have zero mean and unit variance, since knn classification is based on distance
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

##  Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

##  Classification
from sklearn.neighbors import KNeighborsClassifier

##  number of neighbours
k = 5#4 - 4 gives better accuracy
##  Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

##  Predicting

yhat = neigh.predict(X_test)
yhat[0:5]

##  Accurcy
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

