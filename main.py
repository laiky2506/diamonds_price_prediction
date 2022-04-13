# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 12:16:23 2022

@author: laiky
"""

import numpy as np
import tensorflow as tf
import pandas as pd

#load csv data into pd dataframe
data = pd.read_csv('diamonds.csv')

#check data completeness
print(data.info())

#Perform one hot encoding on column with string object
data = pd.get_dummies(data)


#%%
#set labels and features
label = data['price']

#features: drop id and diagnosis
features = data.drop(labels=['Unnamed: 0','price'], axis=1)


#%%
#Train Test Split and Standardize data
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
SEED = 12345
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=SEED)
x_train = np.array(x_train)
x_test = np.array(x_test)

standardizer = preprocessing.StandardScaler()
standardizer.fit(x_train)
x_train = standardizer.transform(x_train)
x_test = standardizer.transform(x_test)

#%%
#Setting of the neural network model

inputs = tf.keras.Input(shape=(x_train.shape[-1],))
dense = tf.keras.layers.Dense(64,activation='relu')
x = dense(inputs)
dense = tf.keras.layers.Dense(32,activation='relu')
x = dense(x)
dense = tf.keras.layers.Dense(16,activation='relu')
x = dense(x)
dense = tf.keras.layers.Dense(8,activation='relu')
x = dense(x)
dense = tf.keras.layers.Dense(4,activation='relu')
x = dense(x)
dense = tf.keras.layers.Dense(2,activation='relu')
x = dense(x)
outputs = tf.keras.layers.Dense(1,activation='relu')(x)
model = tf.keras.Model(inputs=inputs,outputs=outputs,name='breast_cancer_model')
model.summary()

#%%
#start the training

callback = tf.keras.callbacks.EarlyStopping(patience=5)
model.compile(optimizer='adam',loss='mse',metrics=['mse','mae'])
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=32,epochs=128,callbacks=[callback])

#%%
#Print MSE and MAE, also plot scatterplot of Actual vs Prediction
import matplotlib.pyplot as plt

print(f"Mean validation MSE = {np.mean(history.history['mse'])}")
print(f"Mean validation MAE = {np.mean(history.history['mae'])}")

y_pred = model.predict(x_test)
y_pred = np.array(y_pred)
y_test = np.array(y_test)

fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(y_pred,y_test, color=(0,0,1,0.1),s=18)
ax.set_title('Test Data: Actual vs Prediction')
ax.set_xlabel('prediction')
ax.set_ylabel('actual')

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, linestyle='--')
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

plt.show()
