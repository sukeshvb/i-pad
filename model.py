# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:15:34 2019

@author: Omkar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, 1:]

y = dataset.iloc[:,:-4]

def screen_to_int(word):
    word_dict = {'Mini':1, 'Air':2, 'Pro':3}
    return word_dict[word]

def capacity_to_int(word):
    word_dict = {'16GB':1, '32GB':2, '64GB':3, '128GB':4}
    return word_dict[word]

def Connectivity_to_int(word):
    word_dict = {'Wifi':0, 'wifi':0, 'Cellular':1}
    return word_dict[word]

def Gen_to_int(word):
    word_dict = {'Previous':0, 'Current':1, 'current':1}
    return word_dict[word]

X['Screen'] = X['Screen'].apply(lambda x : screen_to_int(x))

X['Capacity'] = X['Capacity'].apply(lambda x : capacity_to_int(x))

X['Connectivity'] = X['Connectivity'].apply(lambda x : Connectivity_to_int(x))

X['Gen'] = X['Gen'].apply(lambda x : Gen_to_int(x))

#X.info()


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X, y)

pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,2,1,0]]))

