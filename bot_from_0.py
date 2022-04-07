import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ta.trend import ADXIndicator
from zigzag import *
import threading
import datetime

# from pandas_datareader import get_data_yahoo
import plotly.express as px

import tensorflow as tf
import tulipy as ti
import datetime as dt

from api_var import api_key, api_secret

import time
import binance
from binance.client import Client

'''

prendre les donner du cours de la crypto

donner des indicateurs

reformuler les données pour quelles coles

afficher pour vérifier les données

initialiser le nn

mettre les données dans un nn

lancer les données de sotries du nn sur un chart

afficher les points d'achat et de ventes

'''

# model.save_weights('./checkpoints/my_checkpoint')

# probability_model.save_weights('./checkpoints/my_checkpoint')

def add_pivots(pivots):
    a = 0
    b = 0
    for i in range(len(pivots)):
        pivots = pivots
        if pivots[i] == 1:
            if i > 2:
                pass
                # if pivots[i-1] != -1:
                #    pivots[i-1] = 1

                # if pivots[i-2] !=-1:
                #    pivots[i-2] = 1

                # if pivots[i-3] !=-1:
                #    pivots[i-3] = 1

                # if pivots[i-4] !=-1:
                #    pivots[i-4] = 1

        if pivots[i] == -1:
            if i > 3:
                pass
                # if pivots[i-1] !=1:
                #    pivots[i-1] = -1

                # if pivots[i-2] !=1:
                #    pivots[i-2] = -1

    for i in range(len(pivots)):
        if pivots[i] == -1:
            a += 1
        if pivots[i] == 1:
            b += 1
    print(len(pivots))
    print('ouverture', a)
    print('fermeture', b)
    print('ouverture + fermeture', a + b)
    return pivots


"""
def SetModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(1, 90)),
        tf.keras.layers.Dense(60),
        tf.keras.layers.Dense(40),
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
    return model, opt, loss_fn


def SetProbaModel(model, opt, loss_fn):
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    probability_model.compile(optimizer=opt,
                              loss=loss_fn,
                              metrics=['accuracy'])
    return probability_model
"""


def plot_pivots(X, pivots):
    plt.plot(np.arange(len(X)), X)
    plt.scatter(np.arange(len(X))[pivots == 1], X[pivots == 1], color='r')
    plt.scatter(np.arange(len(X))[pivots == -1], X[pivots == -1], color='g')


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def simple_bot_buy_sell(lst_v, lst_adx, lst_adx_pos, lst_adx_neg, _val, para01=0.18, para0=0.18, para1=0.19,
                        para2=0.18):
    global max_down
    benef = 0
    lst_point = []
    in_trade = False
    in_long = False
    in_short = False
    pas_f = False
    nb_benef = 0
    toto_benef = 0
    balance = [0]
    max_down = 999999999999999999999999999
    down_point=999999999999999
    up_point=-999999999999999

    # pas def et def open = dif

    lst_buy, lst_sell = [], []

    for i in range(len(lst_v)):
        a = 0
        # if i > val_period:
        b = balance[-1]
        if not in_trade:
            # if orange plus haut rapport au blue:
            if lst_adx_pos[i] > lst_adx_neg[i] and lst_adx_pos[i - 1] < lst_adx_neg[i - 1] and lst_adx[i] > para01:
                in_trade = True
                in_long = True
                # print('open long', lst_v[i])
                long = lst_v[i]
                a = -1
            if lst_adx_pos[i] < lst_adx_neg[i] and lst_adx_pos[i - 1] > lst_adx_neg[i - 1] and lst_adx[i] > para0:
                in_trade = True
                in_short = True
                # print('open short at ', lst_v[i])
                short = lst_v[i]
                a = 1
                # if croisement du orange à la baisse par rapport au blue:
                #    fermeture de trade
        else:
            """pas_red = lst_adx[i] - lst_adx[i - 1]
                if pas_f == 'azerty':
                    if pas_red <= pas_def:
                        # print(pas_red, pas_def)
                        # --
                        if in_long:
                            in_trade = False
                            in_short = False
                            # print('close long')
                            benef += (long - lst_v[i]) / 100 - 1
                            if (long - lst_v[i]) / 100 > 0:
                                nb_benef += 1
                            else:
                                nb_benef -= 1
                            # print('trade benef', (long-lst_v[i])/100)
                            toto_benef += 1
                            a = 1
                        else:
                            if in_short:
                                a = -1
                                in_trade = False
                                in_long = False
                                # print('close short')
                                benef += (short - lst_v[i]) / 100 - 1

                                if (short - lst_v[i]) / 100 > 0:
                                    nb_benef += 1
                                else:
                                    nb_benef -= 1
                                # print('trade benef', (short-lst_v[i])/100)
                                toto_benef += 1
                        # --
                else:
                    pass
                    #if pas_red > pas_def_open:
                    #    pas_f = True"""

            # if orange plus bas par rapport au blue:
            if in_long:
                if lst_adx_pos[i] < lst_adx_neg[i] and lst_adx_pos[i - 1] > lst_adx_neg[i - 1]:
                    # in_trade = False
                    in_long = False
                    if lst_adx[i] > para1:  # XRP : 18 ,ETH : 19
                        in_short = True
                        short = lst_v[i]
                    else:
                        in_trade = False

                    # print('close long')
                    b = balance[-1] + (100 / long * lst_v[i]) - 100 - _val
                    benef += (100 / long * lst_v[i]) - 100 - _val
                    # print('stop long benef', (100 / long * lst_v[i]) - 100)
                    # print('open short at ', lst_v[i])

                    if ((100 / long * lst_v[i]) - 100) > 0:
                        nb_benef += 1
                    else:
                        pass  # nb_benef -= 1
                    a = 1
                    toto_benef += 1
            if in_short:
                if lst_adx_pos[i] > lst_adx_neg[i] and lst_adx_pos[i - 1] < lst_adx_neg[i - 1]:
                    # in_trade = False
                    in_short = False
                    if lst_adx[i] > para2:  # XRP : 18 ,ETH : 18
                        in_long = True
                        long = lst_v[i]
                    else:
                        in_trade = False

                    # print('stop short benef', (short * 100 / lst_v[i]) - 100)
                    # print('open long', lst_v[i])
                    b = balance[-1] + (short * 100 / lst_v[i]) - 100 - _val
                    benef += (short * 100 / lst_v[i]) - 100 - _val
                    if (short * 100 / lst_v[i]) - 100 > 0:
                        nb_benef += 1
                    else:
                        pass  # nb_benef -= 1
                    a = -1
                    toto_benef += 1
                    # print('trade benef', (short-lst_v[i])/100)

                # if croisement du orange à la baisse par rapport au blue:
                #    fermeture de trade
        lst_point.append(a)
        balance.append(b)
        up_point=balance[-1]
        down_point=balance[-1]
        if len(balance)>0:
            if balance[-1] > up_point:
                up_point = balance[-1]
                down_point = balance[-1]
            if balance[-1] < down_point:
                down_point = balance[-1]
                if (down_point - up_point) < max_down:
                    max_down = down_point - up_point
    # print('benef final', benef, nb_benef)
    return np.array(lst_point), benef, nb_benef, toto_benef, balance, max_down


client = Client(api_key, api_secret)

print('client set')
# lst_x=[]
# for x in client.futures_symbol_ticker():
#        #print(any(i.isdigit() for i in str(x)))
#        #if not any(i.isdigit() for i in str(x)):
#        lst_x.append(x['symbol'])

# lst_x = ['BTCUSDT', 'BNBUSDT', 'LINKUSDT', 'ADAUSDT', 'DOGEUSDT', 'UNIUSDT', 'TLMUSDT', 'RAYUSDT', 'MASKUSDT',
#         'ATAUSDT', 'DOTUSDT']
#
lst_x = ['BTTUSDT', 'IOTXUSDT', 'XLMUSDT', 'OMGUSDT', 'BALUSDT', 'DOGEBUSD', 'XTZUSDT', 'ARUSDT', 'DENTUSDT',
         'GALAUSDT', 'ONTUSDT', 'ADABUSD', 'BNBUSDT', 'DODOUSDT', 'ATOMUSDT', 'CRVUSDT', 'ETHBUSD', 'RAYUSDT',
         'COTIUSDT', 'ALPHAUSDT', 'AXSUSDT', 'ATAUSDT', 'XRPBUSD', 'KNCUSDT', 'RENUSDT', 'C98USDT', 'DASHUSDT',
         'ZECUSDT', 'STMXUSDT', 'CHRUSDT', 'RLCUSDT', 'IOSTUSDT', 'RUNEUSDT', '1INCHUSDT', 'CELOUSDT', 'AAVEUSDT',
         'ANKRUSDT', 'CVCUSDT', 'XMRUSDT', 'LTCUSDT', 'BTCBUSD', 'BNBBUSD', 'FILUSDT', 'REEFUSDT', 'BCHUSDT', 'CTKUSDT',
         'ONEUSDT', 'TOMOUSDT', 'NEARUSDT', 'AKROUSDT', 'BTCUSDT', 'KAVAUSDT', 'LUNAUSDT', 'FTMUSDT', 'SCUSDT',
         'DYDXUSDT', 'SXPUSDT', 'HBARUSDT', 'ZRXUSDT', 'SOLUSDT', 'XRPUSDT', 'BELUSDT', 'ADAUSDT', 'KEEPUSDT',
         'FLMUSDT', 'SKLUSDT', 'BZRXUSDT', 'OCEANUSDT', 'ICPUSDT', 'ALICEUSDT', 'DOGEUSDT', 'CHZUSDT', 'VETUSDT',
         'BLZUSDT', 'MASKUSDT', 'RVNUSDT', 'SUSHIUSDT', 'ICXUSDT', 'STORJUSDT', 'ENJUSDT', 'HNTUSDT', 'DGBUSDT',
         'LINAUSDT', 'IOTAUSDT', 'NEOUSDT', 'ALGOUSDT', 'UNIUSDT', 'TRBUSDT', 'MTLUSDT', 'YFIIUSDT', 'EGLDUSDT',
         'TRXUSDT', 'NKNUSDT', 'RSRUSDT', 'DOTUSDT', 'AUDIOUSDT', 'BANDUSDT', 'ZILUSDT', 'LITUSDT', 'BAKEUSDT',
         'LINKUSDT', 'WAVESUSDT', 'SOLBUSD', 'BATUSDT', 'YFIUSDT', 'AVAXUSDT', 'GTCUSDT', 'MANAUSDT', 'ETCUSDT',
         'UNFIUSDT', 'MKRUSDT', 'COMPUSDT', 'XEMUSDT', 'ETHUSDT', 'SANDUSDT', 'GRTUSDT', 'QTUMUSDT', 'OGNUSDT',
         'KSMUSDT', 'CELRUSDT', 'BTSUSDT', 'ZENUSDT', 'LRCUSDT', 'THETAUSDT', 'EOSUSDT', 'SRMUSDT', 'HOTUSDT',
         'FTTBUSD', 'TLMUSDT', 'MATICUSDT', 'SNXUSDT', 'SFPUSDT']

best_crypto = []
print('azerty', lst_x)
print('azerty', len(lst_x))
aaa = 0
"""for iiii in lst_x:
    print('aaa', aaa)
    aaa+=1
    candlesticks = client.get_historical_klines(iiii, Client.KLINE_INTERVAL_2HOUR, '8 oct, 2020', '8 oct, 2021')
    #candlesticks = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_2HOUR, '25 mars, 2021', '25 sept, 2021')

    '''

    BENEF POUR 100 sur chaque trade : sans et avec les frais de 0.0750

    1min =  avec frais = met trop de temps à calculer
    3min =  avec frais = met trop de temps à calculer
    5min =  avec frais = -451.43
    15min =  avec frais = -81.47
    30min =  avec frais = -9.49

    1hour =  avec frais = 54.08
    2hour =  avec frais = 98.03
    4hour =  avec frais = 33.56
    6hour =  avec frais = 15.56
    8hour =  avec frais = 16.56
    12hour =  avec frais = 13.46

    1DAY = 92.83 avec frais = 92.45
    3DAY =  avec frais = -8.68

    pas pertinant
    1WEEK =  avec frais =
    1MONTH =  avec frais =

    2h -> fonctionnement pendant le bearmarket et pendant la correction
    daily -> fonctionnement pendant le bearmarket et pendant la correction mais moins que le 2h
    4h -> fonctionnement pendant le bearmarket et pendant la correction mais mieux que les autres timeframes


    '''

    data = candlesticks
    # np.save('btcusdt_6h', candlesticks)
    ##data = np.load('btcusdt_6h.npy')
    # pivots = peak_valley_pivots(df.close.values, 0.05, -0.1)
    # pivots = add_pivots(pivots)
    # rsi = ti.rsi(df.close.values, 14)
    # print('rsi', rsi)
    df = pd.DataFrame(data=data,
                      columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
                               "nb_volume", "taker_buy_base_asset_volume", "taker_buy_quote_asset_asset_volume", "ignored"],
                      dtype=float)
    df = df.copy()

    adx = ADXIndicator(pd.Series(df.high.values), pd.Series(df.low.values), pd.Series(df.close.values), 14)

    lst_adx = [x / 100 for x in list(adx.adx().values)]  # [27:len(list(adx.adx().values))]
    lst_adx_pos = [x / 100 for x in list(adx.adx_pos().values)]  # [27:len(list(adx.adx_pos().values))]
    lst_adx_neg = [x / 100 for x in list(adx.adx_neg().values)]  # [27:len(list(adx.adx_neg().values))]

    # uniformiser les données
    lst_price_open = df.open.values[27:]
    lst_price_high = df.high.values[27:]
    lst_price_low = df.low.values[27:]
    lst_price_close = df.close.values[27:]

    lst_adx = lst_adx[27:]
    lst_adx_pos = lst_adx_pos[27:]
    lst_adx_neg = lst_adx_neg[27:]
    # pivots = pivots[27:]
    """"""
    para_total = [0, 0, 0]
    max_benef_para = -9999999999999999999
    for i in np.arange(0, 0.4, 0.01):
        print('i', i, end='')
        for ii in np.arange(0, 0.5, 0.01):
            for iii in np.arange(0, 0.5, 0.01):
                lst_point_dmi, benef, nb_benef, toto_benef, balance, max_down = simple_bot_buy_sell(lst_price_close,
                                                                                                    lst_adx,
                                                                                                    lst_adx_pos,
                                                                                                    lst_adx_neg,
                                                                                                    0.0750 + 0.0750, i,
                                                                                                    ii, iii)
                if benef > max_benef_para:
                    max_benef_para = benef
                    para_total = [i, ii, iii]
    """"""
    print([iiii, max_benef_para, para_total])
    best_crypto.append([iiii, max_benef_para, para_total])
    time.sleep(0.1)

lst_ord=[]
l = len(best_crypto)
for i in range(l):
    temp_name=''
    temp_nb=-9999999999999
    temp_para=[]
    place = 0
    for ii in range(len(best_crypto)):
        if best_crypto[ii][1]>temp_nb:
            temp_name=best_crypto[ii][0]
            temp_nb=best_crypto[ii][1]
            temp_para=best_crypto[ii][2]
            place=ii
    lst_ord.append([temp_name, temp_nb, temp_para])
    del best_crypto[place]
print('lst_ord best para crypto: ', lst_ord)

"""

max_benef_para = 9999999999999999999


def test_bot(lst_price_close, lst_adx, lst_adx_pos, lst_adx_neg, iiii, i, ii, iii):
    global max_benef_para, para_total, k_interval
    lst_point_dmi, benef, nb_benef, toto_benef, balance, max_down = simple_bot_buy_sell(lst_price_close, lst_adx,
                                                                                        lst_adx_pos, lst_adx_neg,
                                                                                        0.0400, iiii, i, ii, iii)
    # if benef > max_benef_para:
    #     max_benef_para = benef
    #     k_interval = ai
    #     para_total = [iiii, i, ii, iii]

    if max_down == 0:
        if benef > max_benef_para:
            max_benef_para = benef
            k_interval = ai
            para_total = [iiii, i, ii, iii]
            print(f'nope down: benef:{max_benef_para}  ::  kline:{k_interval}  ::  para_total:{para_total}')


lst_active_thread = []

lst_best_crypto = []
k_interval = Client.KLINE_INTERVAL_1HOUR
# crypto pour le bot: doge akro trx
# XRP :
# Client.KLINE_INTERVAL_1MINUTE, Client.KLINE_INTERVAL_3MINUTE, Client.KLINE_INTERVAL_5MINUTE, Client.KLINE_INTERVAL_15MINUTE, Client.KLINE_INTERVAL_30MINUTE,

#lst_kliine = [Client.KLINE_INTERVAL_12HOUR, Client.KLINE_INTERVAL_8HOUR, Client.KLINE_INTERVAL_6HOUR,
#              Client.KLINE_INTERVAL_4HOUR, Client.KLINE_INTERVAL_2HOUR, Client.KLINE_INTERVAL_1HOUR]
up_point = -999999999999999
down_point = 999999999999999
max_down = 0

lst_kliine = [Client.KLINE_INTERVAL_1DAY]
for iaiao in ["XLMUSDT"]:

    max_benef_para = -9999999999999999999
    para_total = []

    for ai in lst_kliine:

        candlesticks = client.get_historical_klines(iaiao, ai, '26 janv, 2021', '14 oct, 2021')
        # candlesticks = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_2HOUR, '25 mars, 2021', '25 sept, 2021')

        '''

        BENEF POUR 100 sur chaque trade : sans et avec les frais de 0.0750

        1min =  avec frais = met trop de temps à calculer
        3min =  avec frais = met trop de temps à calculer
        5min =  avec frais = -451.43
        15min =  avec frais = -81.47
        30min =  avec frais = -9.49

        1hour =  avec frais = 54.08
        2hour =  avec frais = 98.03
        4hour =  avec frais = 33.56
        6hour =  avec frais = 15.56
        8hour =  avec frais = 16.56
        12hour =  avec frais = 13.46

        1DAY = 92.83 avec frais = 92.45
        3DAY =  avec frais = -8.68

        pas pertinant
        1WEEK =  avec frais =
        1MONTH =  avec frais =

        2h -> fonctionnement pendant le bearmarket et pendant la correction
        daily -> fonctionnement pendant le bearmarket et pendant la correction mais moins que le 2h
        4h -> fonctionnement pendant le bearmarket et pendant la correction mais mieux que les autres timeframes


        '''

        data = candlesticks
        # np.save('btcusdt_6h', candlesticks)
        ##data = np.load('btcusdt_6h.npy')
        # pivots = peak_valley_pivots(df.close.values, 0.05, -0.1)
        # pivots = add_pivots(pivots)
        # rsi = ti.rsi(df.close.values, 14)
        # print('rsi', rsi)
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

        # uniformiser les données
        lst_price_open = df.open.values[27:]
        lst_price_high = df.high.values[27:]
        lst_price_low = df.low.values[27:]
        lst_price_close = df.close.values[27:]
        lst_adx = lst_adx[27:]
        lst_adx_pos = lst_adx_pos[27:]
        lst_adx_neg = lst_adx_neg[27:]
        # pivots = pivots[27:]

        for iiii in np.arange(0, 0.3, 0.01):
            print('i', iiii, end=' ')
            for i in np.arange(0, 0.3, 0.01):
                for ii in np.arange(0, 0.3, 0.01):
                    for iii in np.arange(0, 0.3, 0.01):
                        t_temp = threading.Thread(target=test_bot, args=(
                            lst_price_close, lst_adx, lst_adx_pos, lst_adx_neg, iiii, i, ii, iii))
                        lst_active_thread.append(t_temp)
                        t_temp.start()

                        # lst_point_dmi, benef, nb_benef, toto_benef, balance, max_down = simple_bot_buy_sell(lst_price_close, lst_adx,
                        #                                                                                    lst_adx_pos, lst_adx_neg,
                        #                                                                                    0.0750 + 0.0750, 0.09,0.34,0.34)
                        #                          0.0750 + 0.0750, 0.09,0.34,0.34)

            print('stop all thread', end=' ')
            for iaa in lst_active_thread:
                iaa.join()
                lst_active_thread.remove(iaa)
            print('all thread are stoped', end=' ')
        print('best para', iaiao, str(ai), para_total, max_benef_para)
        print('lst_best_crypto', lst_best_crypto)
        print('')
        print('best para', iaiao, str(k_interval), para_total, max_benef_para)
        lst_best_crypto.append([str(iaiao), str(k_interval), para_total, max_benef_para])
        print('time', datetime.datetime.now())
print(lst_best_crypto)
print('fin')

#
# plt.figure(figsize=(40,25))
# plt.plot(np.arange(len(df.close.values)), df.close.values, 'blue')
# lst_point_dmi, benef, nb_benef, toto_benef, balance, max_down = simple_bot_buy_sell(lst_price_close, lst_adx,
#                                                                                    lst_adx_pos, lst_adx_neg,
#                                                                                    0.0400, para_total[0], para_total[1], para_total[2], para_total[3])

# plt.figure(figsize=(200,100))

# plot_pivots(df.close.values, pivots)
"""print('-- ' * 100)
print('max drawdown', max_down)
print('benef pour 100 dollar sans frais de financement', benef)
print('benef pour 100 dollar', benef - len(lst_price_close)/8*4*(0.02/100*100))
print('pourcentage de réussite', nb_benef/toto_benef)
print('nombre de benef', nb_benef)
print('nombre total', toto_benef)
print('-- ' * 100)

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

ax1.plot(np.arange(len(lst_price_close)), lst_price_close)
# ax1.scatter(np.arange(len(lst_price_close))[pivots == 1], lst_price_close[pivots == 1], color='r')
# ax1.scatter(np.arange(len(lst_price_close))[pivots == -1], lst_price_close[pivots == -1], color='g')

ax1.scatter(np.arange(len(lst_price_close))[lst_point_dmi == 1], lst_price_close[lst_point_dmi == 1], color='r')
ax1.scatter(np.arange(len(lst_price_close))[lst_point_dmi == -1], lst_price_close[lst_point_dmi == -1], color='g')

balance[-1] = balance[-1] - len(lst_price_close)/8*4*(0.02/100*100)
ax2.plot(np.arange(len(balance)), balance)
# ax2.scatter(np.arange(len(df.close.values))[pivots == 1], pivots[pivots == 1], color='r')
# ax2.scatter(np.arange(len(df.close.values))[pivots == -1], pivots[pivots == -1], color='g')

ax3.plot(np.arange(len(lst_adx)), lst_adx, color='r')
ax3.plot(np.arange(len(lst_adx_pos)), lst_adx_pos, color='y')
ax3.plot(np.arange(len(lst_adx_neg)), lst_adx_neg, color='b')

plt.show()

"""
"""
les croisements sont rentable
1/3 des trades sont rentable
mais ce tiers est largement rentable
"""

"""

#best crypto list 2h
#[['DOGEBUSD', 913.7335005432989], ['CHZUSDT', 886.0282689842279], ['AXSUSDT', 827.3877373526765], ['DOGEUSDT', 711.9097820696387], ['DENTUSDT', 660.9840638185618], ['BTTUSDT', 586.6666359835044], ['CHRUSDT', 531.0914022910903], ['STMXUSDT', 504.69612101940925], ['ONEUSDT', 493.3520738097945], ['RVNUSDT', 473.0774566371478], ['HOTUSDT', 453.4456524914712], ['FTMUSDT', 423.1284151520258], ['ETCUSDT', 417.7719878838337], ['NKNUSDT', 408.4727186144538], ['AKROUSDT', 384.6217711028299], ['LUNAUSDT', 383.3420163426143], ['FILUSDT', 382.5008987490965], ['AVAXUSDT', 352.79920403551347], ['SCUSDT', 339.68063195138046], ['AUDIOUSDT', 338.79746415793784], ['TRXUSDT', 318.18836234523553], ['XEMUSDT', 315.5666683483817], ['MATICUSDT', 310.7370424436875], ['OGNUSDT', 307.7632010713562], ['BNBUSDT', 302.9602924185487], ['BNBBUSD', 285.6166148271567], ['XRPUSDT', 284.59043647156426], ['IOSTUSDT', 275.57363186150855], ['IOTXUSDT', 271.15949677024287], ['EGLDUSDT', 269.88412697496545], ['OMGUSDT', 268.4300767168525], ['XRPBUSD', 264.49778537422793], ['TLMUSDT', 260.22279382753237], ['IOTAUSDT', 259.2094896084694], ['ATOMUSDT', 250.0272557258804], ['SXPUSDT', 248.0709626579002], ['UNFIUSDT', 246.73669727387443], ['ALICEUSDT', 241.62872391220432], ['DASHUSDT', 229.3421833252521], ['ALPHAUSDT', 209.74891760644988], ['ATAUSDT', 207.13315256512004], ['SANDUSDT', 206.7565832657345], ['BCHUSDT', 199.3754132895859], ['RUNEUSDT', 195.82241437910838], ['QTUMUSDT', 184.70367869654174], ['CELRUSDT', 177.82217562332679], ['RENUSDT', 173.7464291784164], ['BALUSDT', 170.57918147631474], ['ZECUSDT', 164.15407178476616], ['ANKRUSDT', 161.24882461014974], ['XTZUSDT', 152.58880578084072], ['ICPUSDT', 147.2388411759603], ['ETHBUSD', 144.66973245580252], ['ENJUSDT', 143.66369835892647], ['ETHUSDT', 143.56878224999144], ['CRVUSDT', 143.55718422562623], ['UNIUSDT', 141.39316608481806], ['C98USDT', 135.23293492286817], ['1INCHUSDT', 129.711056392783], ['GALAUSDT', 125.08117896500242], ['OCEANUSDT', 119.21277755749843], ['RAYUSDT', 117.04102407216361], ['LINAUSDT', 112.1591544666838], ['NEOUSDT', 109.79570990778325], ['VETUSDT', 109.12620118461861], ['SRMUSDT', 108.8250119330676], ['DODOUSDT', 107.65179538148766], ['GRTUSDT', 105.71334258042147], ['ZRXUSDT', 104.83583719592025], ['MTLUSDT', 103.39467289895627], ['LINKUSDT', 87.2172563165093], ['SFPUSDT', 84.88452977492562], ['ADAUSDT', 83.7760102166589], ['BTCUSDT', 82.43351450776703], ['SOLUSDT', 81.07569601921972], ['EOSUSDT', 74.30787379490108], ['BZRXUSDT', 73.9176105035667], ['BELUSDT', 72.09215508668602], ['ARUSDT', 69.93858522057963], ['WAVESUSDT', 67.89385787786668], ['STORJUSDT', 60.81392034642483], ['ADABUSD', 59.108094522546686], ['BTCBUSD', 59.02227371245837], ['AAVEUSDT', 57.4355622491786], ['LTCUSDT', 50.652828188111506], ['BATUSDT', 50.1270946286518], ['THETAUSDT', 48.16669767669676], ['ZENUSDT', 41.82917991566387], ['XLMUSDT', 41.37323324396515], ['KAVAUSDT', 39.965219159927045], ['GTCUSDT', 35.93823693472115], ['COMPUSDT', 33.44485767767898], ['CTKUSDT', 29.558021384137263], ['CVCUSDT', 26.23267503421824], ['RSRUSDT', 25.530326689304722], ['CELOUSDT', 18.271191459215284], ['YFIUSDT', 17.69856069321788], ['MANAUSDT', 14.67602920348549], ['MASKUSDT', 11.32536016897422], ['ZILUSDT', 10.81095324106753], ['LITUSDT', 9.260714298245269], ['DOTUSDT', 7.083266569449583], ['BAKEUSDT', 1.5743668656958594], ['BLZUSDT', -8.823347813706874], ['COTIUSDT', -10.458808129837115], ['SUSHIUSDT', -10.758112528288638], ['KSMUSDT', -27.880567782237563], ['DYDXUSDT', -30.87096760828085], ['FTTBUSD', -34.55391668731547], ['SOLBUSD', -35.577734511100665], ['YFIIUSDT', -43.18029442440477], ['RLCUSDT', -48.710388818001825], ['TRBUSDT', -55.278912448054776], ['NEARUSDT', -59.6276289604069], ['SKLUSDT', -59.843470198345194], ['LRCUSDT', -63.92897417102029], ['ONTUSDT', -67.26185508055161], ['MKRUSDT', -69.1317785130686], ['FLMUSDT', -89.20469306180277], ['ICXUSDT', -102.84587404236137], ['KEEPUSDT', -105.64269060708857], ['SNXUSDT', -111.20102424780686], ['REEFUSDT', -118.32314164715392], ['BTSUSDT', -129.24166392869273], ['ALGOUSDT', -130.23569044066411], ['DGBUSDT', -146.79456833111357], ['KNCUSDT', -161.91416995217511], ['XMRUSDT', -178.25468854231576], ['HBARUSDT', -180.76668194527804], ['BANDUSDT', -208.5869013063438], ['TOMOUSDT', -210.5458894054881], ['HNTUSDT', -249.680413622432]]

lst = [['DOGEBUSD', 913.7335005432989], ['CHZUSDT', 'long therme récession', 886.0282689842279], ['AXSUSDT', 'pas ouf resseceiontoin', 827.3877373526765], ['DOGEUSDT', 711.9097820696387], ['DENTUSDT', 660.9840638185618], ['BTTUSDT', 586.6666359835044], ['CHRUSDT', 531.0914022910903], ['STMXUSDT', 504.69612101940925], ['ONEUSDT', 493.3520738097945], ['RVNUSDT', 473.0774566371478], ['HOTUSDT', 453.4456524914712], ['FTMUSDT', 423.1284151520258], ['ETCUSDT', 417.7719878838337], ['NKNUSDT', 'pas rentable lors des baisse', 408.4727186144538], ['AKROUSDT', 384.6217711028299], ['LUNAUSDT', 383.3420163426143], ['FILUSDT', 382.5008987490965], ['AVAXUSDT', 352.79920403551347], ['SCUSDT', 339.68063195138046], ['AUDIOUSDT', 338.79746415793784], ['TRXUSDT', 318.18836234523553], ['XEMUSDT', 315.5666683483817], ['MATICUSDT', 310.7370424436875], ['OGNUSDT', 307.7632010713562], ['BNBUSDT', 302.9602924185487], ['BNBBUSD', 285.6166148271567], ['XRPUSDT', 284.59043647156426], ['IOSTUSDT', 275.57363186150855], ['IOTXUSDT', 271.15949677024287], ['EGLDUSDT', 269.88412697496545], ['OMGUSDT', 268.4300767168525], ['XRPBUSD', 264.49778537422793], ['TLMUSDT', 260.22279382753237], ['IOTAUSDT', 259.2094896084694], ['ATOMUSDT', 250.0272557258804], ['SXPUSDT', 248.0709626579002], ['UNFIUSDT', 246.73669727387443], ['ALICEUSDT', 241.62872391220432], ['DASHUSDT', 229.3421833252521], ['ALPHAUSDT', 209.74891760644988], ['ATAUSDT', 207.13315256512004], ['SANDUSDT', 206.7565832657345], ['BCHUSDT', 199.3754132895859], ['RUNEUSDT', 195.82241437910838], ['QTUMUSDT', 184.70367869654174], ['CELRUSDT', 177.82217562332679], ['RENUSDT', 173.7464291784164], ['BALUSDT', 170.57918147631474], ['ZECUSDT', 164.15407178476616], ['ANKRUSDT', 161.24882461014974], ['XTZUSDT', 152.58880578084072], ['ICPUSDT', 147.2388411759603], ['ETHBUSD', 144.66973245580252], ['ENJUSDT', 143.66369835892647], ['ETHUSDT', 143.56878224999144], ['CRVUSDT', 143.55718422562623], ['UNIUSDT', 141.39316608481806], ['C98USDT', 135.23293492286817], ['1INCHUSDT', 129.711056392783], ['GALAUSDT', 125.08117896500242], ['OCEANUSDT', 119.21277755749843], ['RAYUSDT', 117.04102407216361], ['LINAUSDT', 112.1591544666838], ['NEOUSDT', 109.79570990778325], ['VETUSDT', 109.12620118461861], ['SRMUSDT', 108.8250119330676], ['DODOUSDT', 107.65179538148766], ['GRTUSDT', 105.71334258042147], ['ZRXUSDT', 104.83583719592025], ['MTLUSDT', 103.39467289895627], ['LINKUSDT', 87.2172563165093], ['SFPUSDT', 84.88452977492562], ['ADAUSDT', 83.7760102166589], ['BTCUSDT', 82.43351450776703], ['SOLUSDT', 81.07569601921972], ['EOSUSDT', 74.30787379490108], ['BZRXUSDT', 73.9176105035667], ['BELUSDT', 72.09215508668602], ['ARUSDT', 69.93858522057963], ['WAVESUSDT', 67.89385787786668], ['STORJUSDT', 60.81392034642483], ['ADABUSD', 59.108094522546686], ['BTCBUSD', 59.02227371245837], ['AAVEUSDT', 57.4355622491786], ['LTCUSDT', 50.652828188111506], ['BATUSDT', 50.1270946286518], ['THETAUSDT', 48.16669767669676], ['ZENUSDT', 41.82917991566387], ['XLMUSDT', 41.37323324396515], ['KAVAUSDT', 39.965219159927045], ['GTCUSDT', 35.93823693472115], ['COMPUSDT', 33.44485767767898], ['CTKUSDT', 29.558021384137263], ['CVCUSDT', 26.23267503421824], ['RSRUSDT', 25.530326689304722], ['CELOUSDT', 18.271191459215284], ['YFIUSDT', 17.69856069321788], ['MANAUSDT', 14.67602920348549], ['MASKUSDT', 11.32536016897422], ['ZILUSDT', 10.81095324106753], ['LITUSDT', 9.260714298245269], ['DOTUSDT', 7.083266569449583], ['BAKEUSDT', 1.5743668656958594], ['BLZUSDT', -8.823347813706874], ['COTIUSDT', -10.458808129837115], ['SUSHIUSDT', -10.758112528288638], ['KSMUSDT', -27.880567782237563], ['DYDXUSDT', -30.87096760828085], ['FTTBUSD', -34.55391668731547], ['SOLBUSD', -35.577734511100665], ['YFIIUSDT', -43.18029442440477], ['RLCUSDT', -48.710388818001825], ['TRBUSDT', -55.278912448054776], ['NEARUSDT', -59.6276289604069], ['SKLUSDT', -59.843470198345194], ['LRCUSDT', -63.92897417102029], ['ONTUSDT', -67.26185508055161], ['MKRUSDT', -69.1317785130686], ['FLMUSDT', -89.20469306180277], ['ICXUSDT', -102.84587404236137], ['KEEPUSDT', -105.64269060708857], ['SNXUSDT', -111.20102424780686], ['REEFUSDT', -118.32314164715392], ['BTSUSDT', -129.24166392869273], ['ALGOUSDT', -130.23569044066411], ['DGBUSDT', -146.79456833111357], ['KNCUSDT', -161.91416995217511], ['XMRUSDT', -178.25468854231576], ['HBARUSDT', -180.76668194527804], ['BANDUSDT', -208.5869013063438], ['TOMOUSDT', -210.5458894054881], ['HNTUSDT', -249.680413622432]]
"""
"""
BEST PARA :
XRPUSDT
best para [0.09, 0.34, 0.34] 373.0206063894469


refaire la list avec l'amélioration (adx)
enregistrer la liste

"""
# de janv a oct
# ['BTCUSDT', 131.3597239255018, [0.08, 0.35, 0.38]]
# ['ETHUSDT', 196.6628074912079, [0.11, 0.26, 0.25]]
# ['DOTUSDT', 190.20198188464764, [0.12, 0.38, 0.34]]

# de oct 2020 à oct 2021
# [['DOTUSDT', 272.87417433659726, [0.12, 0.47, 0.34]], ['ETHUSDT', 231.666106079773, [0.1, 0.19, 0.0]], ['BTCUSDT', 150.25913913814276, [0.09, 0.36, 0.24]]]

# [['BTCUSDT', [0.21, 0.45, 0.43], 13.774735498777222], ['ETHUSDT', [0.28, 0.19, 0.26], 37.25021059991599], ['BNBUSDT', [0.24, 0.45, 0.45], 29.502153830416], ['LINKUSDT', [0.3, 0.25, 0.43], 24.723976586590638], ['ADAUSDT', [0.2, 0.27, 0.32], 3.411357485303644], ['DOGEUSDT', [0.15, 0.34, 0.2], 4.826203511115905], ['UNIUSDT', [0.23, 0.39, 0.35000000000000003], 26.92073760905557], ['TLMUSDT', [0.42, 0.0, 0.2], 33.14588512803642], ['RAYUSDT', [0.16, 0.3, 0.39], 71.58143478161188], ['MASKUSDT', [0.0, 0.33, 0.35000000000000003], 14.662592814520934], ['ATAUSDT', [0.2, 0.31, 0.28], 27.921951653214705], ['DOTUSDT', [0.4, 0.0, 0.38], 11.597032805241334]]

# ETHUSDT 4h [0.0, 0.26, 0.23, 0.29] 31.39251661024018

# du 14 janv au 14 oct
# ETHUSDT 4h [0.31, 0.19, 0.36, 0.16] 231.52807851705035


# du '26 juin, 2021' au '14 oct, 2021'
# ICPUSDT 4h [0.0, 0.0, 0.19, 0.23] 86.25636688503025


# ethusdt


# h4 = 186
# h2 =


# de janv à oct perfect crypto
# best para LTCUSDT 4h [0.0, 0.0, 0.2, 0.19] 265.53455628144195
