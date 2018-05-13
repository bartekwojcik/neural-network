import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from keras import metrics
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json
import sys
from keras.layers import Dropout


def build_classifier(optimizer):
    classifier = Sequential()
    # adding hidden layer
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dropout(rate=0.1))
    # adding SECOND HIDDEN layer
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=0.1))
    # OUPUT layer
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    # compile
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
    return classifier


def check_this_one_guy(labelencoder_X_1, labelencoder_X_2, onehotencoder, sc):
    json_file = open('/home/bartek/PycharmProjects/ann/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("/home/bartek/PycharmProjects/ann/model.h5")
    print("Loaded model from disk")

    loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics.binary_accuracy])
    thisOneGuy = np.array([[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]])
    thisOneGuy[:, 1] = labelencoder_X_1.transform(thisOneGuy[:, 1])
    thisOneGuy[:, 2] = labelencoder_X_2.transform(thisOneGuy[:, 2])
    thisOneGuy = onehotencoder.transform(thisOneGuy).toarray()
    thisOneGuy = thisOneGuy[:, 1:]

    thisOneGuy = sc.transform(thisOneGuy)

    guyPred = loaded_model.predict(thisOneGuy)
    debug = 5
    print(guyPred)

def train(should_train):
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
    if should_train == "True":
        classifier = KerasClassifier(build_fn=build_classifier)
        parameters = {'batch_size': {25, 32},
                      'epochs':[100,500],
                      'optimizer':['adam','rmsprop']}

        grid_search = GridSearchCV(estimator=classifier, param_grid= parameters, scoring='accuracy', cv= 10)

        best_parameters = grid_search.best_params_
        best_accuracy = grid_search.best_score_
        # old train
        #accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train,cv =10,n_jobs=-1)
        # before using k-fold cv classifier.fit(x_train, y_train, batch_size=10, epochs=15)
        #mean = accuracies.mean()
        #variance = accuracies.std()
        #print("mean: " + str(mean) + ", std: "+str(variance))
        # tell me the truth
        y_pred = classifier.predict(x_test)
        y_pred = (y_pred > 0.5)

        cm = confusion_matrix(y_test, y_pred)
        print(*cm)

        model_json = classifier.to_json()
        with open("/home/bartek/PycharmProjects/ann/model.json", "w") as json_file:
            json_file.write(model_json)
        classifier.save_weights("/home/bartek/PycharmProjects/ann/model.h5")
        print("Saved model to disk")

    check_this_one_guy(labelencoder_X_1, labelencoder_X_2, onehotencoder, sc)


if __name__ == '__main__':
    should_train = sys.argv[1]
    train(should_train)


