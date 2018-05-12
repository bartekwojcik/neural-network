import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:12].values
y = dataset.iloc[:,13].values
from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=123,stratify=y)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit(x_train)
x_test = sc.transform(x_test)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


