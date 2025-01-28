import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r"D:\Users\User\Desktop\Studia\MGR\SEM 1\IMS\F16Data_SineSw_Level3.csv") #zwraca Data Frame

x=data[['Force','Voltage','Acceleration1']].astype(float)

x = x.loc[(x != 0).any(axis=1)]
print(x)

scaler = StandardScaler()
scaler = scaler.fit(x)
x_scaled = scaler.transform(x)
X = x[['Force','Voltage']].values
Y = x[['Acceleration1']].values




X_t=X[80000:,:]
Y_t=Y[80000:,:]
X1=X[:80000,:]
Y1=Y[:80000,:]
# X_t=X
# Y_t=Y
# X1=X
# Y1=Y




def create_dataset (X, y, time_steps = 1):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        v = X[i:i+time_steps, :]
        Xs.append(v)
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)
TIME_STEPS = 100
X_test, y_test = create_dataset(X_t, Y_t,TIME_STEPS)
X_train, y_train = create_dataset(X1, Y1,TIME_STEPS)

print(X_train)
print(X_train.shape)
print(y_train)
print(y_train.shape)

# TWORZENIE MODELU
model = Sequential()
model.add(LSTM(64,input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1]))
model.compile(optimizer='adam',loss='mse')

# TRENING
model.fit(X_train,y_train,epochs=3,batch_size=64,validation_split=0.1,verbose=1)


wyjscie=model.predict(X_test)
w=np.array(wyjscie)
w=np.c_[np.ones(len(w)),np.ones(len(w)),w]
y_train=np.c_[np.ones(len(y_train)),np.ones(len(y_train)),y_train]
w=scaler.inverse_transform(w)
Y_end=np.concatenate((y_train,w),dtype=float)

# print(Y_end)
# print(len(Y_end))
# print(Y_end[:,0])

t=np.linspace(0,1000,len(Y_end))
t2=np.linspace(0,1000,len(Y))

plt.figure(1)
plt.plot(t2,Y)
plt.figure(2)
plt.plot(t,Y_end[:,2])
plt.show()



