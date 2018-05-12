import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
import sys


def main(args):
    dataset = pd.read_csv('Churn_Modelling.csv')
    x = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values

    # preprocessing
    # Encoding categorical data
    # Encoding the Independent Variable
    labelencoder_X_1 = LabelEncoder()
    x[:, 1] = labelencoder_X_1.fit_transform(x[:, 1])
    labelencoder_X_2 = LabelEncoder()
    x[:, 2] = labelencoder_X_2.fit_transform(x[:, 2])

    onehotencoder = OneHotEncoder(categorical_features=[1])
    x = onehotencoder.fit_transform(x).toarray()
    x = x[:, 1:]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123, stratify=y)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # train network

    classifier = Sequential()
    # adding hidden layer
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    # adding SECOND HIDDEN layer
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    # OUPUT layer
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    # compile
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics.binary_accuracy])
    # train
    classifier.fit(x_train, y_train, batch_size=10, epochs=30) #slow virtual machine
    # tell me the truth
    y_pred = classifier.predict(x_test)
    y_pred = (y_pred > 0.5)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    debug =5

if __name__ == '__main__':
    main(sys.argv)
