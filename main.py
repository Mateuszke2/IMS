
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import sys
import statistics
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

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



# print(len(Y_train),len(Y_test))


parametry = {'learning_rate': ['constant','adaptive'], 'activation' : ['tanh','logistic'], 'learning_rate_init' : (0.1,0.01,0.001), 'hidden_layer_sizes' : [(20,) ,(40,), (60,), (80,), (100,)],'solver' : ['adam', 'lbfgs', 'sgd']}



mlpr = MLPRegressor()

mlpr.fit(X_train,np.ravel(Y_train))

pred = mlpr.predict(X_test)

# print(Y_test)

# print(pred)

print(mlpr.score(X_test,Y_test))

# a=np.array([1,2,3,5,6,7,8,9,9,0,12,3,4])
# b = a.reshape(-1,1)
# scaler = StandardScaler()

# scaler.fit(b)

# x = scaler.transform(b)

