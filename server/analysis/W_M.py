#ttertune - iris.py

"""
Created on : 25th December, 2018

This code is to test the implementation of self designed ML code in ottertune code

It is an implementation of iris classifier, in which we classify the 3 species of iris flower based on the petal length, width and sepal length and width

"""
import numpy as np
import scipy as sp
from sklearn.preprocessing import StandardScaler
import sklearn
import random 
import time 

from sklearn import preprocessing, model_selection

from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle

from sklearn import datasets
# Initializing the class and data

class W_M(object):
    def __init__(self):
        model = Sequential()
        input_dim = 92
        model.add(Dense(8, input_dim = input_dim , activation = 'relu'))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dense(7, activation = 'softmax'))
        model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
        self.model = model
    def fit(self, Zs, labels):
        z = np_utils.to_categorical(labels)
        train_xz, test_xz, train_yz, test_yz = model_selection.train_test_split(Zs,z,test_size = 0.1, random_state = 0)
        self.model.fit(train_xz, train_yz, epochs = 60, batch_size = 2)
        scores = self.model.evaluate(test_xz, test_yz)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
    
    def predict(self,test_data):
        a = np.array(test_data)
        print(a.shape)
        predicted_label = self.model.predict_classes(a)
        print(predicted_label)
        return   
