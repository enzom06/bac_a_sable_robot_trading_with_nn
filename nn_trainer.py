import tensorflow as tf

from ta.trend import ADXIndicator
from binance.client import Client
import matplotlib.pyplot as plt
import tensorflow as tf
from zigzag import *
import pandas as pd
import numpy as np
import threading
import datetime
import pickle
import pygad
import time

import sklearn

from api_var import api_key, api_secret

client = Client(api_key, api_secret)
print('init client')

symbol_ = 'BTCUSDT'
date1 = '1 juil, 2019'
date2 = '1 janv, 2021'

symbol_2 = 'BTCUSDT'
date3 = '1 janv, 2021'
date4 = '27 janv, 2022'


lst_kliine = [Client.KLINE_INTERVAL_15MINUTE]

candlesticks = client.get_historical_klines(symbol_, lst_kliine[0], date1, date2)
with open("data_train.data", "wb") as fic:
    pickle.dump(candlesticks, fic)

#with open("data_train.data", "rb") as fic:
#    candlesticks = pickle.load(fic)

#print('symbol', symbol_, 'kline', lst_kliine)

#candlesticks2 = client.get_historical_klines(symbol_2, lst_kliine[0], date3, date4)
#with open("data_test.data", "wb") as fic:
#    pickle.dump(candlesticks2, fic)

#with open("data_test.data", "rb") as fic:
#    candlesticks2 = pickle.load(fic)

#print('symbol', symbol_2, 'kline', lst_kliine)

#print('all data are downloaded')

"""
def save(fic, fic_name):
    with open(f'{fic_name}.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(list(fic), filehandle)

def read(fic_name):
    with open(f'{fic_name}.data', 'wb') as filehandle:
        return pickle.load(filehandle)


def prepa_data(lst_adx, lst_adx_pos, lst_adx_neg):
    # filtrer les données quand on aura les donnée de zigzag
    lst_x_input = []
    for i in range(6, len(lst_adx)):
        lst_temp = []
        for i2 in range(7):
            lst_temp.append(lst_adx[i - i2])

        for i2 in range(7):
            lst_temp.append(lst_adx_pos[i - i2])

        for i2 in range(7):
            lst_temp.append(lst_adx_neg[i - i2])
        #lst_temp.append(lst_price[i])
        lst_x_input.append(lst_temp) # modifier pour mettre le zig zag
    return lst_x_input

def indicator_mod(az):
    indic = az
    maxx = -9999
    minn = 9999
    for i in range(len(indic)):
        x = indic[i]
        for val in x:
            if val != x[-1]:
                if val > maxx:
                    maxx = val
                if val < minn:
                    minn = val
        for xi in range(len(x) - 1):
            indic[i][xi] = (indic[i][xi] - minn) / (maxx - minn)
            # indic[i] = (indic[i] - minn) / (maxx - minn)
    return indic


# -- -- -- -- --
data = candlesticks
df = pd.DataFrame(data=data,
                  columns=["open_time", "open", "high", "low", "close", "volume", "close_time",
                           "quote_asset_volume",
                           "nb_volume", "taker_buy_base_asset_volume", "taker_buy_quote_asset_asset_volume",
                           "ignored"],
                  dtype=float)
df = df.copy()

adx = ADXIndicator(pd.Series(df.high.values), pd.Series(df.low.values), pd.Series(df.close.values), 14)

lst_adx = [x / 100 for x in list(adx.adx().values)]  # [27:len(list(adx.adx().values))]
lst_adx_pos = [x / 100 for x in list(adx.adx_pos().values)]  # [27:len(list(adx.adx_pos().values))]
lst_adx_neg = [x / 100 for x in list(adx.adx_neg().values)]  # [27:len(list(adx.adx_neg().values))]

lst_adx = lst_adx[27:]
lst_adx_pos = lst_adx_pos[27:]
lst_adx_neg = lst_adx_neg[27:]

# uniformiser les données
lst_price_x_open = df.open.values[27:]
lst_price_x_close = df.close.values[27:]
lst_price_open = df.open.values[27:]
lst_price_high = df.high.values[27:]
lst_price_low = df.low.values[27:]
lst_price_close = df.close.values[27:]
"""

# lst_x_input = [[0, 1], [1, 2], [2, 3], [3, 2], [2, 1], [1, 2], [2, 3]]
"""

compositiond'un bloc d'information:
[7 adx, 7adx pos, 7 adx neg, price]

"""

"""for i in range(6, len(lst_adx)):
    lst_temp = []
    for i2 in range(7):
        lst_temp.append(lst_adx[i - i2])

    for i2 in range(7):
        lst_temp.append(lst_adx_pos[i - i2])

    for i2 in range(7):
        lst_temp.append(lst_adx_neg[i - i2])
    lst_temp.append(lst_price_close[i])
    lst_x_input.append(lst_temp)"""
"""for i in range(6, len(lst_price_x_close)):
    lst_temp = []
    for i2 in range(nb_in):
        lst_temp.append(lst_price_x_open[i - i2])
        lst_temp.append(lst_price_x_close[i - i2])
    #for i2 in range(nb_in):

    lst_temp.append(lst_price_close[i])
    lst_x_input.append(lst_temp)
# print('lst x', lst_x_input)
"""
"""
lst_x_input = prepa_data(lst_adx, lst_adx_pos, lst_adx_neg)

lst_x_input = indicator_mod(lst_x_input)
lst_price_x_open = lst_price_x_open[6:]
lst_price_x_close = lst_price_x_close[6:]
lst_price_open = lst_price_open[6:]
lst_price_high = lst_price_high[6:]
lst_price_low = lst_price_low[6:]
lst_price_close = lst_price_close[6:]

lst_adx = lst_adx[6:]
lst_adx_pos = lst_adx_pos[6:]
lst_adx_neg = lst_adx_neg[6:]
pivots = peak_valley_pivots(lst_price_x_close, 0.03, -0.03)
#-- -- -- -- --

for i in range(len(lst_x_input)):
    lst_x_input[i] = [lst_x_input[i][0:7], lst_x_input[i][7:14], lst_x_input[i][14:21]]
pivots = [float(i) for i in pivots]
for i in range(len(pivots)):
    if pivots[i] == 1:
        pivots[i] = 1
    elif pivots[i] == -1:
        pivots[i] = 0

#print('pivots', pivots[:10])
def plot_pivots(X, pivots):
    plt.plot(np.arange(len(X)), X, 'k:', alpha=0.5)
    plt.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
    plt.scatter(np.arange(len(X))[pivots == 0], X[pivots == 0], color='g')
    plt.scatter(np.arange(len(X))[pivots == 2], X[pivots == 2], color='r')

plot_pivots(lst_price_x_close, pivots)

#plt.show()




#fashion_mnist = tf.keras.datasets.fashion_mnist
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_data = []
train_labels = []

#test_data = []
#test_labels = []

#set data for bot
for i in range(len(pivots)):
    if pivots[i] == 0 or pivots[i] == 1:
        train_data.append(lst_x_input[i])
        train_labels.append(pivots[i])

train_data = np.array(train_data)
train_labels = np.array(train_labels)

print(train_data)
print(train_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(3, 7)), #4*OHLC
    tf.keras.layers.Dense(28, activation='relu'),
    tf.keras.layers.Dense(14, activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#model.fit(train_data, train_labels, epochs=50)
#model.save_weights("nn_weigth")

test_loss, test_acc = model.evaluate(train_data, train_labels, verbose=2)
#test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)


print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])


predictions = probability_model.predict(np.array(train_data))
print('prediction', predictions[0], predictions[1])

"""
