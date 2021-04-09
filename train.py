# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 13:07:44 2021

@author: sense
"""
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM,Dropout,Dense
from tensorflow.keras.callbacks import ModelCheckpoint

from utils import get_data, \
    get_feature_vector_from_mfcc

_DATA_PATH = 'dataset'
_CLASS_LABELS = ("Neutral", "Angry", "Happy", "Sad")


def extract_data(flatten):
    data, labels = get_data(_DATA_PATH, class_labels=_CLASS_LABELS,
                            flatten=flatten)
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(
        y_test), len(_CLASS_LABELS)


x_train, x_test, y_train, y_test, num_labels = extract_data(flatten=False)

y_train = to_categorical(y_train)
y_test_train = to_categorical(y_test)

model = Sequential()
model.add(LSTM(128,input_shape=(x_train[0].shape[0], x_train[0].shape[1])))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(num_labels, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
print(model.summary())

model_checkpoint_callback = ModelCheckpoint(
    filepath='tmp/checkpoint',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.fit(x_train,y_train,epochs=100,batch_size=32,validation_data=(x_test,y_test_train),callbacks=[model_checkpoint_callback])


model.load_weights('tmp/checkpoint')#'best_model_LSTM.h5'
model.evaluate(x_test,y_test_train)


filename = 'dataset/Sad/09b02Tb.wav'
print('prediction', np.argmax(model.predict(np.array([get_feature_vector_from_mfcc(filename, flatten=False)]))),'Actual 3')

