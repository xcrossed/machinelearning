
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn import model_selection
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    # load data
    x, y = load_iris(return_X_y=True)
    # x = X[Y < 2, :2]
    print(x.shape)
    # y = Y[Y < 2]
    print(y.shape)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2, random_state=6)

    model = Sequential()
    model.add(Dense(100, input_shape=(4,), activation='relu'))
    # 8 = no. of neurons in second hidden layer
    model.add(Dense(10, activation='relu'))
    model.add(Dense(units=3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='sgd', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100, batch_size=32)

    loss_and_metrics = model.evaluate(x_test, y_test)
    print(loss_and_metrics)
