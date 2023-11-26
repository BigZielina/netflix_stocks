import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

import pandas as pd

df = pd.read_csv('NFLX.csv')
DF = df[['Close']].dropna()

close_real_prices = df.values
print(DF)

def window_df(df, n):
    windowed_df = pd.DataFrame()
    for i in range(n, 0, -1):
        windowed_df[f'Target-{i}'] = df['Close'].shift(i)
    windowed_df['Target'] = df['Close']
    return windowed_df.dropna()

window = 125
df_window = window_df(df, window)

dataset = df.values
dataset = dataset.astype('float32')

scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = scaler.fit_transform(dataset.reshape(-1, 1))


train_size = int(len(dataset) * 0.65)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = window
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print(testX.shape)


trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
print((trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print((trainX.shape[0], 1, trainX.shape[1]))


#####

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

model = Sequential()
model.add(LSTM(200, return_sequences=True, input_shape=(trainX.shape[1], window)))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])
model.fit(trainX, trainY, epochs = 500,batch_size=16,callbacks=[callback])




testX.shape
predictions = []
latest_prices = trainX

general_prediction = model.predict(latest_prices)[-1]

x,y,z = latest_prices.shape
for i in range(len(test)):
    
    prediction = model.predict(latest_prices)[-1][0]
    predictions.append(prediction)
    print(prediction)
    temp_array = latest_prices[1:,:,:]
    last_row = temp_array[-1,:,:]
    temp_row = np.zeros(last_row.shape)
    

    
    for i in range(z-1):
        temp_row[0][i] = last_row[0][i+1]
    
    temp_row[0][-1] = prediction
    # print(temp_row)
    # temp_row_reshaped = temp_row.reshape(1, temp_row.shape[0], 1)
    latest_prices = np.concatenate([latest_prices, temp_row[np.newaxis, :, :]], axis=0)
    # print(temp_row[0,-3:])


