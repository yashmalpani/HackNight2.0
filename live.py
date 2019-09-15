import numpy as np
import datetime as dt
import pandas as pd
import pandas_datareader.data as web
from scrap import data,tweets

now = dt.datetime.now()
beg = dt.datetime(now.year-10, now.month,now.day)
start = dt.datetime(now.year-2,now.month,now.day)
df = web.DataReader('MSFT','yahoo',beg, now)
df=df.reset_index()

training_set = df.iloc[:, 3:4].values


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


X_train = []
y_train = []
for i in range(60, training_set_scaled.shape[0]):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from keras.models import Sequential, model_from_json
'''
from keras.layers import LSTM, Dense, Dropout
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 30, batch_size = 254, verbose = True)



model_json = regressor.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

regressor.save_weights("model.h5")'''




json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

unit = 0.1*(0.7*data() + 0.3*tweets())
unit=0
price_predicted = []
future_days = 5

'''dataset_total = pd.concat((df['Open'], test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 76):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
'''
for i in range(future_days):
    to_predict = df.tail(1)['Adj Close']
    dataset_total = pd.concat((df['Open'][:-1], to_predict), axis = 0)
    inputs = dataset_total[len(df['Open']) - 0 - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = np.reshape(inputs,(1,60,1))
    prediction = sc.inverse_transform(loaded_model.predict(X_test))*(1+unit)
    price_predicted.append(float(str(prediction[0][0])[:8]))
    df=df.append({"Adj Close":float(str(prediction[0][0])[:8]),"Open":float((str(to_predict).split()[1])[:8])}, ignore_index = True)

import matplotlib.pyplot as plt
plt.plot(price_predicted)
plt.ylabel('prize')
plt.xlabel('days ahead')

plt.show()
'''

import matplotlib.pyplot as plt
plt.plot(real_stock_price, color = 'black', label = 'TATA Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TATA Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()
plt.show()'''