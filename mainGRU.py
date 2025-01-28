from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import sys
import statistics
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense,GRU,LSTM


dt=pd.read_csv(r"D:\Users\User\Desktop\Studia\MGR\SEM 1\IMS\F16Data_SineSw_Level3.csv")

dt=dt.drop(columns=["Acceleration2","Acceleration3","Fs"])

dt = dt.loc[(dt != 0).any(axis=1)]
print(dt)



X = dt.drop(columns=["Acceleration1"])
Y = dt.drop(columns=["Force","Voltage"])

print(Y)

procent = 0.5
forward = 10

X_train = X.iloc[:int(procent*len(dt)),:].values
Y_train = Y.iloc[:int(procent*len(dt)),:].values

# DO PREDYKCJI:

# WSZYSTKIE
# X_test = X.iloc[int(procent*len(dt)):,:].values
# Y_test = Y.iloc[int(procent*len(dt)):,:].values

# PLUS WAROTŚĆ FORWARD
X_test = X.iloc[int(procent*len(dt)):int(procent*len(dt)+forward),:].values
Y_test = Y.iloc[int(procent*len(dt)):int(procent*len(dt)+forward),:].values


def create_dataset (X, y, time_steps = 1):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        v = X[i:i+time_steps, :]
        Xs.append(v)
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)




X_train, y_train = create_dataset(train_x_norm, train_y_norm, 
                                  TIME_STEPS)   #ty bo tu ci cos nie dziala XD 
print('X_train.shape: ', X_test.shape)
print('y_train.shape: ', y_train.shape)
print('X_test.shape: ', X_test.shape)
print('y_test.shape: ', y_train.shape)#seks? tak


