import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime

# load the Yahoo stock data
df = pd.read_csv("yahoo_stock_data.csv")

# normalize the data using MinMaxScaler
scaler = MinMaxScaler()
df = scaler.fit_transform(df)

# create a function to create a data set
def create_dataset(df):
    x = []
    y = []
    for i in range(len(df)-1):
        x.append(df[i])
        y.append(df[i+1])
    x = np.array(x)
    y = np.array(y)
    return x, y

# create the data set
x, y = create_dataset(df)

# create the model
model = Sequential()
model.add(LSTM(50, input_shape=(x.shape[1], x.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# fit the model
model.fit(x, y, epochs=100, batch_size=1, verbose=2)

# make a prediction for the next day
last_day = df[-1]
next_day = model.predict(np.array([last_day]))

# inverse the normalization
next_day = scaler.inverse_transform(next_day)

# print the predicted price for the next day
print("Predicted price for next day: ", next_day[0][0])
