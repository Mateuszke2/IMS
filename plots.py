import sys
import statistics
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt




dt=pd.read_csv(r"D:\Users\User\Desktop\Studia\MGR\SEM 1\IMS\F16Data_SineSw_Level3.csv")

dt2=pd.read_csv(r"D:\Users\User\Desktop\Studia\MGR\SEM 1\IMS\F16Data_SineSw_Level2_Validation.csv")

dt = dt.loc[(dt != 0).any(axis=1)]

dt2 = dt2.loc[(dt2 != 0).any(axis=1)]

print(dt.shape)

t = np.linspace(0,0.0025*dt.shape[0],dt.shape[0])

t2 = np.linspace(0,0.0025*dt2.shape[0],dt2.shape[0])

print(t)
print(t.shape)

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

print(dt["Voltage"])

plt.plot(t, np.array(dt['Acceleration1']))
plt.title('Acceleration1')
plt.show()

fig, axs = plt.subplots(5,sharex=True)

fig.suptitle("SineSW_Level3")

axs[0].plot(t, np.array(dt['Voltage']))
axs[0].title('Voltage')
axs[1].plot(t, np.array(dt['Force']))

axs[2].plot(t, np.array(dt['Acceleration1']))

axs[3].plot(t, np.array(dt['Acceleration2']))

axs[4].plot(t, np.array(dt['Acceleration3']))

fig2, axs2 = plt.subplots(5,sharex=True)

fig2.suptitle("SineSW_Level2_Val")

axs2[0].plot(t2, np.array(dt2['Voltage']))

axs2[1].plot(t2, np.array(dt2['Force']))

axs2[2].plot(t2, np.array(dt2['Acceleration1']))

axs2[3].plot(t2, np.array(dt2['Acceleration2']))

axs2[4].plot(t2, np.array(dt2['Acceleration3']))

plt.show()


