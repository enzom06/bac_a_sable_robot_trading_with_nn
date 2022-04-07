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

model = create_model()

probability_model = create_proba_model(model)

model.load_weights('./checkpoints/my_checkpoint_rsi_willr_nn_90_60_40_3')
probability_model.load_weights('./checkpoints/my_checkpoint_rsi_willr_nn_90_60_40_3_proba_model')

print('bot de test')
print('-- ' for i in range(100))
# best paramettre pour l'instant:
# [2, 4, 3, 4, 4, 0, 0, 4]
print('entrez les paramettres(nb, sépparé avec des virgules\n')
lst_best_para = str(input('liste de paramettres\n')).split(',')
lst_best_para = [float(i) for i in lst_best_para]
reserve = int(input('entrer la reserve:\n--> '))
part = int(input('entrer la part:\n--> '))
fee = (float(input('entrer les fee:\n--> '))) / 100
bot_fee_part = (float(input('entrer les frais du bot:\n--> '))) / 100
# sys compte maj par une def

from api_var import api_key, api_secret
client = Client(api_key, api_secret, {"timeout": 20})
# -- -- -- -- -- -- -- -- -- --

lst_buy = []
lst_sell = []
cpt = 0
nb_last_buy = 0
nb_last_sell = 0
# lst_pivots2=[]
sell_activate = 0
temp = 0

# -- -- -- -- --

max_reserve = reserve
all_fees = 0
nb_trade = 0

lst_total_balance = []
lst_total_hold = []
lst_total_bot_benef = []
temp_bot_benef = 0

# -- -- -- -- -- -- -- -- -- --
buy_part = 10
# -- --
first_price_hold = 0  # --------------------------------------------------------------------------------------------------------------------------


# -- -- -- -- -- -- -- -- -- --
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


def crea_lst_price_rsi_srsi_x_price(high, low, close):
    # -- -- -- -- -- -- -- -- -- -- rsi
    lst_price = np.array([])
    lst_price = close[- 30 - 14:]
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
    # -- -- -- -- -- -- -- -- -- -- stochrsi -> willr
    lst_price = np.array([])
    lst_price = close[- 30 - 14:]
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
    llst_price = close[- 30 - 14:]

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
    lst_price = close[-30 - 14:]
    rsi = ti.rsi(lst_price, 14)

    lst_price = list(lst_price)
    del lst_price[0:14]

    del lst_price[0:30]

    return lst_rsi, lst_srsi, lst_x_price, lst_price


def f_choix(lst_x):
    buy, stay, sell = probability_model.predict(lst_x)[0]
    if buy > sell:
        if buy > 0.0:
            return -1
    else:
        if sell > 0.0:
            return


def trade(_candle_high, _candle_low, _candle_close, highs, lows, closes):
    global reserve, part, bot_fee_part, lst_buy, lst_sell, sell_activate, temp, max_reserve, all_fees, nb_trade, lst_total_balance, lst_total_hold, lst_total_bot_benef, temp_bot_benef, buy_part, fee, lst_best_para, first_price_hold

    lst_rsi, lst_willr, lst_x_price, lst_price = crea_lst_price_rsi_srsi_x_price(highs, lows, closes)

    lst_rsi = indicator_mod(lst_rsi)
    lst_willr = indicator_mod(lst_willr)
    lst_x_price = indicator_mod(lst_x_price)

    lst_x_final = np.concatenate((lst_rsi, lst_willr, lst_x_price), axis=2)

    # print(lst_best_para)
    temp_i = f_choix(lst_x_final)

    # -- -- -- -- -- -- -- -- -- --
    # first choice test
    # f_test_choix_6(i, _candle_close)
    i = f_test_choix_8(temp_i, lst_best_para, _candle_close)

    # -- -- -- -- -- -- -- -- -- --
    # i loop

    if i == -1:
        if reserve > max_reserve:
            buy_part = part  # reserve*part/max_reserve
        else:
            buy_part = part
        if reserve >= buy_part:
            temp_fees = buy_part * fee
            nb_trade += 1
            lst_buy.append(buy_part / _candle_close)
            reserve -= buy_part
            reserve -= temp_fees
            all_fees += temp_fees

    elif i == 1:
        if len(lst_buy) > 0:
            temp_token_restant = 0
            for i in range(len(lst_buy)):
                temp_token_restant += lst_buy[i]
            if (reserve + _candle_close * temp_token_restant) >= max_reserve:
                if len(lst_total_balance) > 14:
                    if lst_total_balance[-1] - lst_total_balance[-14] > 0:
                        nb_trade += 1
                        temp_fees = lst_buy[0] * _candle_close * fee

                        if buy_part - lst_buy[0] * _candle_close > 0:
                            part_benef = buy_part - lst_buy[0] * _candle_close
                            reserve += lst_buy[0] * _candle_close - part_benef * bot_fee_part
                            temp_bot_benef = part_benef * bot_fee_part
                        else:
                            reserve += lst_buy[0] * _candle_close

                        reserve -= temp_fees
                        all_fees += temp_fees
                        lst_buy.pop(0)
                    else:
                        nb_trade += 1
                        temp_fees = lst_buy[0] * _candle_close * fee
                        reserve += lst_buy[0] * _candle_close
                        reserve -= temp_fees
                        all_fees += temp_fees
                        lst_buy.pop(0)
            else:
                nb_trade += 1
                temp_fees = lst_buy[0] * _candle_close * fee
                reserve += lst_buy[0] * _candle_close
                reserve -= temp_fees
                all_fees += temp_fees
                lst_buy.pop(0)
                # if len(lst_buy)>0:
                #    nb_trade+=1
                #    temp_fees = lst_buy[0]*lst_price[cpt]*fee
                #    reserve+=lst_buy[0]*lst_price[cpt]-temp_fees
                #    all_fees+=temp_fees
                #    lst_buy.pop(0)

    lst_total_hold.append(max_reserve / first_price_hold * _candle_close)
    temp_token_restant = 0
    for i in range(len(lst_buy)):
        temp_token_restant += lst_buy[i]

    if len(lst_total_bot_benef) > 0:
        lst_total_bot_benef.append(temp_bot_benef + lst_total_bot_benef[-1])
    else:
        lst_total_bot_benef.append(temp_bot_benef)
    lst_total_balance.append(reserve + _candle_close * temp_token_restant)

    with open('all_fees.txt', 'w') as fic:
        fic.write(str(all_fees))
    with open('lst_total_balance.txt', 'w') as fic:
        fic.write(str(lst_total_balance))
    with open('lst_total_bot_benef.txt', 'w') as fic:
        fic.write(str(lst_total_bot_benef))
    with open('lst_total_hold.txt', 'w') as fic:
        fic.write(str(lst_total_hold))
    with open('nb_trade.txt', 'w') as fic:
        fic.write(str(nb_trade))
    # -- -- -- --
    # print('-- '*20)
    # print('-- '*20)
    # print(f'paramettre: nb_last_sell {para1} : nb_last_buy {para2} : nb_last_buy {para3} : nb_last_sell {para4}')
    # print('-- '*20)
    # print('nb trade', nb_trade, 'fees', all_fees)
    """ changer lst_price[-1] -> prix en live
    if len(lst_buy) > 0:
        temp_token_restant = 0
        for i in range(len(lst_buy)):
            temp_token_restant += lst_buy[i]
        print('reserve final', reserve + lst_price[-1] * temp_token_restant)
        if reserve + lst_price[-1] * temp_token_restant > last_best_price:
            last_best_price = reserve + lst_price[-1] * temp_token_restant
        del temp_token_restant
    else:
        print('reserve', reserve)"""
    return [lst_total_balance, lst_total_bot_benef, lst_total_hold, temp_token_restant, all_fees]


def real_trade(_candle_high, _candle_low, _candle_close, highs, lows, closes):
    global reserve, part, bot_fee_part, lst_buy, lst_sell, sell_activate, temp, max_reserve, all_fees, nb_trade, lst_total_balance, lst_total_hold, lst_total_bot_benef, temp_bot_benef, buy_part, fee, lst_best_para, first_price_hold

    lst_rsi, lst_willr, lst_x_price, lst_price = crea_lst_price_rsi_srsi_x_price(highs, lows, closes)

    lst_rsi = indicator_mod(lst_rsi)
    lst_willr = indicator_mod(lst_willr)
    lst_x_price = indicator_mod(lst_x_price)

    lst_x_final = np.concatenate((lst_rsi, lst_willr, lst_x_price), axis=2)

    # print(lst_best_para)
    temp_i = f_choix(lst_x_final)

    # -- -- -- -- -- -- -- -- -- --

    # first choice test
    # f_test_choix_6(i, _candle_close)
    i = f_test_choix_8(temp_i, lst_best_para, _candle_close)

    # -- -- -- -- -- -- -- -- -- --
    # i loop

    if i == -1:
        if reserve > max_reserve:
            buy_part = part  # reserve*part/max_reserve
        else:
            buy_part = part
        if reserve >= buy_part:
            order1 = client.order_market_buy(symbol='BTCUSDT', quoteOrderQty=buy_part)
            # print(order1)
            nb_trade += 1

            lst_buy.append(float(order1['executedQty']))
            reserve -= float(order1['cummulativeQuoteQty'])
            all_fees += float(order1['fills'][0]['commission'])

            """

            temp_fees = buy_part * fee
            nb_trade += 1

            lst_buy.append(buy_part / _candle_close)
            reserve -= buy_part
            reserve -= temp_fees

            """

    elif i == 1:
        if len(lst_buy) > 0:
            temp_token_restant = 0
            for i in range(len(lst_buy)):
                temp_token_restant += lst_buy[i]
            if (reserve + _candle_close * temp_token_restant) >= max_reserve:
                if len(lst_total_balance) > 14:
                    if lst_total_balance[-1] - lst_total_balance[-14] > 0:
                        nb_trade += 1

                        order2 = client.order_market_sell(
                            symbol='BTCUSDT',
                            quantity=lst_buy[0])
                        reserve += float(order2['cummulativeQuoteQty'])
                        all_fees += float(order2['fills'][0]['commission'])

                        if (buy_part - float(order2['cummulativeQuoteQty'])) > 0:
                            temp_bot_benef = float(order2['cummulativeQuoteQty']) * fee

                        lst_buy.pop(0)

                    else:

                        order2 = client.order_market_sell(
                            symbol='BTCUSDT',
                            quantity=lst_buy[0])
                        reserve += float(order2['cummulativeQuoteQty'])
                        all_fees += float(order2['fills'][0]['commission'])

                        nb_trade += 1
                        lst_buy.pop(0)
            else:
                nb_trade += 1
                order2 = client.order_market_sell(
                    symbol='BTCUSDT',
                    quantity=lst_buy[0])
                reserve += float(order2['cummulativeQuoteQty'])
                all_fees += float(order2['fills'][0]['commission'])
                lst_buy.pop(0)
                # if len(lst_buy)>0:
                #    nb_trade+=1
                #    temp_fees = lst_buy[0]*lst_price[cpt]*fee
                #    reserve+=lst_buy[0]*lst_price[cpt]-temp_fees
                #    all_fees+=temp_fees
                #    lst_buy.pop(0)

    lst_total_hold.append(max_reserve / first_price_hold * _candle_close)
    temp_token_restant = 0
    for i in range(len(lst_buy)):
        temp_token_restant += lst_buy[i]

    if len(lst_total_bot_benef) > 0:
        lst_total_bot_benef.append(temp_bot_benef + lst_total_bot_benef[-1])
    else:
        lst_total_bot_benef.append(temp_bot_benef)
    lst_total_balance.append(reserve + _candle_close * temp_token_restant)

    with open('all_fees.txt', 'w') as fic:
        fic.write(str(all_fees))
    with open('lst_total_balance.txt', 'w') as fic:
        fic.write(str(lst_total_balance))
    with open('lst_total_bot_benef.txt', 'w') as fic:
        fic.write(str(lst_total_bot_benef))
    with open('lst_total_hold.txt', 'w') as fic:
        fic.write(str(lst_total_hold))
    with open('nb_trade.txt', 'w') as fic:
        fic.write(str(nb_trade))
    # -- -- -- --
    # print('-- '*20)
    # print('-- '*20)
    # print(f'paramettre: nb_last_sell {para1} : nb_last_buy {para2} : nb_last_buy {para3} : nb_last_sell {para4}')
    # print('-- '*20)
    # print('nb trade', nb_trade, 'fees', all_fees)
    """ changer lst_price[-1] -> prix en live
    if len(lst_buy) > 0:
        temp_token_restant = 0
        for i in range(len(lst_buy)):
            temp_token_restant += lst_buy[i]
        print('reserve final', reserve + lst_price[-1] * temp_token_restant)
        if reserve + lst_price[-1] * temp_token_restant > last_best_price:
            last_best_price = reserve + lst_price[-1] * temp_token_restant
        del temp_token_restant
    else:
        print('reserve', reserve)"""


def f_test_choix_6():
    pass


def f_test_choix_8(i, lst_best_para, _close):
    global nb_last_buy, nb_last_sell
    temp = 0
    if i == -1:
        # nb_last_sell=para1
        if nb_last_buy >= lst_best_para[0]:  # -----5
            temp = -1
            lst_buy.append(_close)
            # lst_x_buy.append(cpt)
            # nb_last_buy=0
            if nb_last_buy >= lst_best_para[1] + lst_best_para[0]:  # --------5
                nb_last_sell = -lst_best_para[2]  # --------10
            # else:
            #    nb_last_sell = -lst_best_para[3]  # --------10
        nb_last_buy += 1

    elif i == 1:
        # nb_last_buy=para3
        if nb_last_sell >= lst_best_para[3]:  # -------10
            temp = 1
            lst_sell.append(_close)
            # lst_x_sell.append(cpt)

            if nb_last_sell >= lst_best_para[4] + lst_best_para[3]:  # -------10
                nb_last_buy = -lst_best_para[5]  # --------5
            # else:
            #    nb_last_buy = -lst_best_para[6]  # --------5
            # nb_last_sell=0
            # pass
        # if nb_last_buy>0: #-------
        #    #nb_last_buy=-2#-------
        #    if nb_last_buy<0:#------
        #        #nb_last_buy=0#------
        #        pass
        nb_last_sell += 1
    # cpt += 1

    return temp
