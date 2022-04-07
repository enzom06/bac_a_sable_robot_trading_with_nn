import numpy as np
import pandas as pd
from zigzag import *

from pandas_datareader import get_data_yahoo
import plotly.express as px

import tensorflow as tf
import tulipy as ti
import time

import matplotlib.pyplot as plt
from zigzag import *

import datetime as dt

import binance
from binance.client import Client


# Save the weights
# model.save_weights('./checkpoints/my_checkpoint')

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(1, 72)),
        tf.keras.layers.Dense(48),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(3)
    ])

    # test sur du 24h sur une autre fichier jupyter

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        name="Adam"
    )
    # model.compile(optimizer='adam',loss='mean_squared_error')

    model.compile(optimizer=opt,
                  loss=loss_fn,
                  metrics=['accuracy'])
    return model


# Create a new model instance
model = create_model()


def create_proba_model(model):
    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        name="Adam"
    )
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    probability_model.compile(optimizer=opt,
                              loss=loss_fn,
                              metrics=['accuracy'])
    return probability_model


probability_model = create_proba_model(model)

# Restore the weights
# model.load_weights('./checkpoints/my_checkpoint')

model.save_weights('./checkpoints/my_checkpoint_rsi_willr_nn_90_60_40_3')
probability_model.save_weights('./checkpoints/my_checkpoint_rsi_willr_nn_90_60_40_3_proba_model')

print('bot de test')
print('-- ' for i in range(100))
# input paramettre
print('entrez les paramettres\n')
lst_best_para = str(input('liste de paramettres\n'))
lst_best_para = list(lst_best_para)
# -- -- -- -- -- -- -- -- -- --
'''
utilisr une liste de paramettre montrer la solde
faire un fichier des prix, achat/ventes
. . .
'''

# -- -- -- -- -- -- -- -- -- --
# live
from api_var import api_key, api_secret
client = Client(api_key, api_secret, {"timeout": 20})
# -- -- -- -- -- -- -- -- -- --
buy_part = 10
cpt = 0

reserve = 100  # 100
part = 10  # 10
bot_fee_part = 0.0
max_reserve = reserve
all_fees = 0
fee = 0.0750 / 100
nb_trade = 0
lst_buy = []
lst_total_balance = []
lst_total_hold = []
lst_total_bot_benef = []
temp_bot_benef = 0


# last loop in lst pivots

# -- -- -- -- -- -- -- -- -- -- créer lst x
def indicator_mod(az):
    indic = az
    maxx = -9999
    minn = 9999
    for i in range(len(indic)):
        x = indic[i][0]
        for val in x:
            if val > maxx:
                maxx = val
            if val < minn:
                minn = val
    print(minn, maxx)

    for i in range(len(indic)):
        indic[i][0] = (indic[i][0] - minn) / (maxx - minn)

    m = (maxx + minn) / 2
    for i in range(len(indic)):
        indic[i][0] = indic[i][0] - m

    print(minn, maxx)
    maxx = -9999
    minn = 9999
    for i in range(len(indic)):
        x = indic[i][0]
        for val in x:
            if val > maxx:
                maxx = val
            if val < minn:
                minn = val
    print(minn, maxx)
    for i in range(len(indic)):
        indic[i][0] = (indic[i][0] - minn) / (maxx - minn)
    print(minn, maxx)

    return indic


def crea_lst_price_rsi_srsi_x_price(open, high, low, close):
    # -- -- -- -- -- -- -- -- -- -- rsi
    lst_price = np.array([])
    lst_price = low[- 30 - 14:]
    rsi = ti.rsi(lst_price, 14)

    lst_price = list(lst_price)
    del lst_price[0:14]

    lst_rsi = np.array([])

    for x_y in range(30, len(lst_price)):
        lst_temp = []
        for i in range(-30, 0):
            lst_temp.append(rsi[x_y + i])

        if lst_rsi.shape == (0,):
            lst_rsi = np.array([[lst_temp]])
        else:
            lst_rsi = np.append(lst_rsi, [[lst_temp]], axis=0)
    del lst_price[0:30]
    # -- -- -- -- -- -- -- -- -- -- stochrsi ->willr
    lst_price = np.array([])
    lst_price = low[- 30 - 14:]
    lst_high = np.array([])
    lst_high = high[- 30 - 14:]
    lst_close = np.array([])
    lst_close = close[- 30 - 14:]
    willr = ti.willr(lst_high, lst_price, lst_close, 14)

    lst_price = list(lst_price)
    lst_high = list(lst_high)
    lst_close = list(lst_close)
    del lst_price[0:14]
    del lst_high[0:14]
    del lst_close[0:14]

    lst_srsi = np.array([])

    for x_y in range(30, len(lst_price)):
        lst_temp = []
        for i in range(-30, 0):
            lst_temp.append(willr[x_y + i])

        if lst_srsi.shape == (0,):
            lst_srsi = np.array([[lst_temp]])
        else:
            lst_srsi = np.append(lst_srsi, [[lst_temp]], axis=0)
    del lst_price[0:30]
    del lst_high
    del lst_close
    # -- -- -- -- -- -- -- -- -- -- lst_x_price

    llst_price = np.array([])
    llst_price = low[- 30 - 14:]

    llst_price = list(llst_price)
    del llst_price[0:14]

    lst_x_price = np.array([])

    for x_y in range(30, len(llst_price)):
        lst_temp = []
        for i in range(-30, 0):
            lst_temp.append(llst_price[x_y + i])

        if lst_x_price.shape == (0,):
            lst_x_price = np.array([[lst_temp]])
        else:
            lst_x_price = np.append(lst_x_price, [[lst_temp]], axis=0)
    del llst_price[0:30]

    lst_price = np.array([])
    lst_price = low[-30 - 14:]
    rsi = ti.rsi(lst_price, 14)

    lst_price = list(lst_price)
    del lst_price[0:14]

    del lst_price[0:30]

    return lst_rsi, lst_srsi, lst_x_price, lst_price


# -- --

# -- --
"""
base de donnée min = 44 Heures
"""
lst_rsi, lst_willr, lst_x_price, lst_price = crea_lst_price_rsi_srsi_x_price()

lst_rsi = indicator_mod(lst_rsi)
lst_willr = indicator_mod(lst_willr)
lst_x_price = indicator_mod(lst_x_price)

lst_x_final = np.concatenate((lst_rsi, lst_willr, lst_x_price), axis=2)


# -- -- -- -- -- -- -- -- -- --

# x = probability_model(x_rsi)
# y
# x.numpy()
def f_choix(lst_x):
    buy, stay, sell = probability_model.predict(lst_x)[0]
    if buy > sell:
        if buy > 0.0:
            return -1
    else:
        if sell > 0.0:
            return


choix = f_choix(lst_x_final)
# -- --
# last loop in lst pivots
temp_bot_benef = 0
# -- -- -- -- -- --

if i == -1:
    if reserve > max_reserve:
        buy_part = part  # reserve*part/max_reserve
    else:
        buy_part = part
    if reserve >= part:
        temp_fees = buy_part * fee
        nb_trade += 1
        lst_buy.append(buy_part / lst_price[cpt])
        reserve -= buy_part
        reserve -= temp_fees
        all_fees += temp_fees

elif i == 1:
    if len(lst_buy) > 0:
        temp_token_restant = 0
        for i in range(len(lst_buy)):
            temp_token_restant += lst_buy[i]
        if (reserve + lst_price[cpt] * temp_token_restant) >= max_reserve:
            if len(lst_total_balance) > 14:
                if lst_total_balance[-1] - lst_total_balance[-14] > 0:
                    nb_trade += 1
                    temp_fees = lst_buy[0] * lst_price[cpt] * fee

                    if buy_part - lst_buy[0] * lst_price[cpt] > 0:
                        part_benef = buy_part - lst_buy[0] * lst_price[cpt]
                        reserve += lst_buy[0] * lst_price[cpt] - part_benef * bot_fee_part
                        temp_bot_benef = part_benef * bot_fee_part
                    else:
                        reserve += lst_buy[0] * lst_price[cpt]

                    reserve -= temp_fees
                    all_fees += temp_fees

                    lst_buy.pop(0)
                else:
                    nb_trade += 1
                    temp_fees = lst_buy[0] * lst_price[cpt] * fee
                    reserve += lst_buy[0] * lst_price[cpt]
                    reserve -= temp_fees
                    all_fees += temp_fees
                    lst_buy.pop(0)
        else:
            nb_trade += 1
            temp_fees = lst_buy[0] * lst_price[cpt] * fee
            reserve += lst_buy[0] * lst_price[cpt]
            reserve -= temp_fees
            all_fees += temp_fees
            lst_buy.pop(0)
            # if len(lst_buy)>0:
            #    nb_trade+=1
            #    temp_fees = lst_buy[0]*lst_price[cpt]*fee
            #    reserve+=lst_buy[0]*lst_price[cpt]-temp_fees
            #    all_fees+=temp_fees
            #    lst_buy.pop(0)
lst_total_hold.append(max_reserve / lst_price[0] * lst_price[cpt])
temp_token_restant = 0
for i in range(len(lst_buy)):
    temp_token_restant += lst_buy[i]

if len(lst_total_bot_benef) > 0:
    lst_total_bot_benef.append(temp_bot_benef + lst_total_bot_benef[-1])
else:
    lst_total_bot_benef.append(temp_bot_benef)
lst_total_balance.append(reserve + lst_price[cpt] * temp_token_restant)
cpt += 1

# -- -- -- --
# print('-- '*20)
# print('-- '*20)
# print(f'paramettre: nb_last_sell {para1} : nb_last_buy {para2} : nb_last_buy {para3} : nb_last_sell {para4}')
# print('-- '*20)
# print('nb trade', nb_trade, 'fees', all_fees)
if len(lst_buy) > 0:
    temp_token_restant = 0
    for i in range(len(lst_buy)):
        temp_token_restant += lst_buy[i]
    print('reserve final', reserve + lst_price[-1] * temp_token_restant)
    if reserve + lst_price[-1] * temp_token_restant > last_best_price:
        last_best_price = reserve + lst_price[-1] * temp_token_restant
    del temp_token_restant
else:
    print('reserve', reserve)

    pass

# -- -- -- -- -- -- -- -- -- --
