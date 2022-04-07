# import time
import datetime as dt
# import asyncio
# from main import *
import websocket, json  # , pprint
from binance.client import Client
from binance.enums import *
# import binance
import tulipy as ti
import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from zigzag import *
import threading, time

# import plotly.express as px

from api_var import api_key, api_secret

# api_key = str(input('api key ->'))
# api_secret = str(input('api secret->'))
# -- -- -- -- -- -- -- -- -- --
RSI_PERIOD = 14
WILLR_PERIOD = 14
TIME_TO_TEST = 30
# TRADE_QUANTITY = 10

opens = []
highs = []
lows = []
closes = []

in_position = False  # pas utile
f = True
situation = []
last_situation = [[]]

# from pandas_datareader import get_data_yahoo

# -- -- -- -- -- -- -- -- -- --init des nn -- -- -- -- -- -- -- -- -- --

# def create_model():
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(1, 90)),
    tf.keras.layers.Dense(60),
    tf.keras.layers.Dense(40),
    tf.keras.layers.Dense(3)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    name="Adam"
)
# model.compile(optimizer='adam',loss='mean_squared_error')
model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['accuracy'])
#    return model
# def create_proba_model(model):
#    opt = tf.keras.optimizers.Adam(
#        learning_rate=0.001,
#        name="Adam"
#    )
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
#    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
probability_model.compile(optimizer=opt,
                          loss=loss_fn,
                          metrics=['accuracy'])
#    return probability_model

# -- -- -- -- -- -- -- -- -- -- : -- -- -- -- -- -- -- -- -- --

client = Client(api_key, api_secret)

Previous_Date = dt.datetime.today() - dt.timedelta(days=2)
date1 = Previous_Date.strftime("%d %m, %Y")

date2 = dt.datetime.now().strftime("%d %m, %Y")
TRADE_SYMBOL = str(input('entrer le symbole de l\'actif tradable en futures(ex:BTCUSDT):\n-->'))

candlesticks = client.get_historical_klines(TRADE_SYMBOL, Client.KLINE_INTERVAL_1HOUR, date1, date2)

for i in candlesticks:
    opens.append(float(i[1]))
    highs.append(float(i[2]))
    lows.append(float(i[3]))
    closes.append(float(i[4]))

lst_sell = []
oneone = False
short = 0

print('ohlc candles length', len(opens))
# client = Client(api_key, api_secret)
# model = create_model()
# probability_model = create_proba_model(model)
print('load models')
model.load_weights('./checkpoints/nn_rsi_willr_nn_90_60_40_3')
probability_model.load_weights('./checkpoints/nn_rsi_willr_nn_90_60_40_3_proba_model')
print('models loaded')
print('bot de test')
print('-- ' for i in range(100))
# best paramettre pour l'instant:
# [2, 4, 3, 4, 4, 0, 0, 4]
precision = int(input('entrer la précision(0.n):\n'))
lst_best_para = str(input('liste de paramettres(ex:0,1,2 --> [0,1,2])\n--> ')).split(',')
lst_best_para = [float(i) for i in lst_best_para]
reserve = int(input('entrer la reserve:\n--> '))
part = int(input('entrer la part(%):\n--> ')) / 100

SOCKET = f"wss://fstream.binance.com/ws/{TRADE_SYMBOL.lower()}@kline_1m"
# bot_fee_part = (float(input('entrer les frais du bot:\n--> '))) / 100
# sys compte maj par une def

# -- -- -- -- -- -- -- -- -- --

# lst_buy = []
# lst_sell = []
cpt = 0
nb_last_buy = 0
nb_last_sell = 0
# lst_pivots2=[]
sell_activate = 0
temp = 0

# -- -- -- -- --

max_reserve = reserve
nb_trade = 0

lst_total_balance = []
lst_total_hold = []
last_total_hold = 0
last_total_bot_benef = 0

in_short_trade = False
in_long_trade = False
# -- -- -- -- -- -- -- -- -- --
buy_part = part * 100
# -- --
first_price_hold = 0  # --------------------------------------------------------------------------------------------------------------------------
print('test trade')
ref_price = float(client.futures_mark_price(symbol=TRADE_SYMBOL)['markPrice'])
# client.futures_create_order(symbol=TRADE_SYMBOL, quantity=str(round(buy_part / ref_price, precision)), side='BUY',
#                            type='MARKET')
# client.futures_create_order(symbol=TRADE_SYMBOL, quantity=str(round(buy_part / ref_price, precision)), side='SELL',
#                            type='MARKET')
print('fin du test')


# -- -- -- -- -- -- -- -- -- --
# -- -- -- -- -- -- -- -- -- --
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
    # print(minn, maxx)

    for i in range(len(indic)):
        indic[i][0] = (indic[i][0] - minn) / (maxx - minn)

    m = (maxx + minn) / 2
    for i in range(len(indic)):
        indic[i][0] = indic[i][0] - m

    # print(minn, maxx)
    maxx = -9999
    minn = 9999
    for i in range(len(indic)):
        x = indic[i][0]
        for val in x:
            if val > maxx:
                maxx = val
            if val < minn:
                minn = val
    # print(minn, maxx)
    for i in range(len(indic)):
        indic[i][0] = (indic[i][0] - minn) / (maxx - minn)
    # print(minn, maxx)

    return indic


def crea_lst_price_rsi_srsi_x_price(high, low, close):
    # -- -- -- -- -- -- -- -- -- -- rsi
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    lst_price = np.array([])
    lst_price = close[-1 - 30 - 14:]
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
    lst_price = low[-1 - 30 - 14:]
    lst_high = np.array([])
    lst_high = high[-1 - 30 - 14:]
    lst_close = np.array([])
    lst_close = close[-1 - 30 - 14:]
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
    llst_price = close[-1 - 30 - 14:]

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
    lst_price = close[-1 - 30 - 14:]
    # rsi = ti.rsi(lst_price, 14)

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
            return 1


def trade(q, s):
    client.futures_create_order(symbol=TRADE_SYMBOL,
                                quantity=q,
                                side=s, type='MARKET')


# pas encore fait
def real_trade(_candle_high, _candle_low, _candle_close, highs, lows, closes):
    # global reserve, part, bot_fee_part, lst_buy, sell_activate, temp, max_reserve, all_fees, nb_trade, lst_total_balance, last_total_hold, last_total_bot_benef, buy_part, fee, lst_best_para, first_price_hold, lst_sell, oneone, short, in_short_trade, in_long_trade
    global reserve, part, nb_trade, lst_total_balance, last_total_hold, buy_part, lst_best_para, first_price_hold, oneone, in_short_trade, in_long_trade, ref_price, precision
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
        if in_short_trade:
            trade(str(round(buy_part / ref_price, precision)), 'BUY')
            nb_trade += 1
            in_short_trade = False

        else:

            if not in_long_trade:
                if reserve > buy_part:
                    ref_price = float(client.futures_mark_price(symbol=TRADE_SYMBOL)['markPrice'])
                    trade(str(round(buy_part / ref_price, precision)), 'BUY')
                    in_long_trade = True
                    nb_trade += 1


    elif i == 1:
        if in_long_trade:
            trade(str(round(buy_part / ref_price, precision)), 'SELL')
            nb_trade += 1
            in_long_trade = False

        else:
            if not in_short_trade:
                if reserve > buy_part:
                    ref_price = float(client.futures_mark_price(symbol=TRADE_SYMBOL)['markPrice'])
                    trade(str(round(buy_part / ref_price, precision)), 'SELL')
                    nb_trade += 1
                    in_short_trade = True
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    reserve = float(client.futures_account_balance()[1]['withdrawAvailable'])
    last_total_hold = max_reserve / first_price_hold * _candle_close

    # temp_token_restant = 0
    # for i in range(len(lst_buy)):
    #    temp_token_restant += lst_buy[i]

    lst_total_balance.append(client.futures_account_balance()[1]['balance'])

    if len(lst_total_balance) > 26:
        print('balance')
        # with open('lst_total_balance.txt', 'a')as f:
        #    f.write(f'{str(lst_total_balance[0])}:')
        lst_total_balance.pop(0)

    # with open('lst_total_hold.txt', 'a') as fic:
    #    fic.write(f':{str(last_total_hold)}')

    # with open('nb_trade.txt', 'w') as fic:
    #    fic.write(str(nb_trade))

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

    return 200


def f_test_choix_6():
    pass


def f_test_choix_8(i, lst_best_para, _close):
    global nb_last_buy, nb_last_sell
    temp = 0
    if i == -1:
        # nb_last_sell=para1
        if nb_last_buy >= lst_best_para[0]:  # -----5
            temp = -1
            # lst_buy.append(_close)
            # lst_x_buy.append(cpt)
            # nb_last_buy=0
            if nb_last_buy >= lst_best_para[1] + lst_best_para[0]:  # --------5
                nb_last_sell = -lst_best_para[2]  # --------10

        nb_last_buy += 1


    elif i == 1:
        if nb_last_sell >= lst_best_para[3]:  # -------10
            temp = 1

            if nb_last_sell >= lst_best_para[4] + lst_best_para[3]:  # -------10
                nb_last_buy = -lst_best_para[5]  # --------5

        nb_last_sell += 1
    # cpt += 1

    return temp


# -- -- -- -- -- -- -- -- -- --
# -- -- -- -- -- -- -- -- -- --
# -- -- -- -- -- -- -- -- -- --


def order(side, quantity, symbol, order_type=ORDER_TYPE_MARKET):
    try:
        print("sending order")
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
        print(order)
    except Exception as e:
        print("an exception occured - {}".format(e))
        return False

    return True


# return true/false order

def on_close(ws):
    print('disconnected from server')


def on_open(ws):
    print('connection established')


def on_message(ws, message):
    global opens, highs, lows, closes, in_position, f, reserve, lst_total_balance, last_total_hold, first_price_hold
    # print('received message')
    json_message = json.loads(message)
    # pprint.pprint(json_message)
    candle = json_message['k']

    is_candle_closed = candle['x']
    open = candle['o']
    high = candle['h']
    low = candle['l']
    close = candle['c']
    # time_candle = candle['t']

    if is_candle_closed:
        print("candle closed at {} : {}".format(close, dt.datetime.now()))
        opens.append(float(open))
        highs.append(float(high))
        lows.append(float(low))
        closes.append(float(close))
        # print("closes")
        # print(closes)
        # print('len closes1', len(closes))
        if len(closes) - 6 > RSI_PERIOD + TIME_TO_TEST:
            opens.pop(0)
            highs.pop(0)
            lows.pop(0)
            closes.pop(0)
        # print('len closes2', len(closes))
        if len(closes) > (RSI_PERIOD + TIME_TO_TEST):
            if f:
                first_price_hold = float(close)
                print(f"bot prêt à l'emploit {dt.datetime.now()}")
                f = False
            # np_closes = numpy.array(closes)
            # last_rsi = 0
            # rsi = talib.RSI(np_closes, RSI_PERIOD)
            # print("all rsis calculated so far")
            # print(rsi)
            # last_rsi = rsi[-1]
            # print("the current rsi is {}".format(last_rsi))
            """
            if last_rsi > 9*9*9:
                if in_position:
                    print("Overbought! Sell! Sell! Sell!")
                    # put binance sell logic here
                    order_succeeded = order(SIDE_SELL, TRADE_QUANTITY, TRADE_SYMBOL)
                    if order_succeeded:
                        in_position = False
                else:
                    print("It is overbought, but we don't own any. Nothing to do.")

            if last_rsi < -(9*9*9):
                if in_position:
                    print("It is oversold, but you already own it, nothing to do.")
                else:
                    print("Oversold! Buy! Buy! Buy!")
                    # put binance buy order logic here
                    order_succeeded = order(SIDE_BUY, TRADE_QUANTITY, TRADE_SYMBOL)
                    if order_succeeded:
                        in_position = True
            """
            real_trade(float(high), float(low), float(close), highs, lows, closes)
            print('-- ' * 20)
            print(f'reserve:       {str(reserve)}')
            if len(lst_total_balance) > 0:
                print(f'total balance: {str(lst_total_balance[-1])}')
            else:
                print(f'total balance: {str(lst_total_balance)}')
            print(f'total hold:    {str(last_total_hold)}')
            print('-- ' * 20)
            # ap real trade


ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
# ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
# ws.run_forever() #ping_timeout=60
# mettre dans un def pr lancer async
if __name__ == "__main__":
    print('-- -- -- -- -- -- -- -- -- -')
    print('lancement du websocket')
    print('-- -- -- -- -- -- -- -- -- --')
    while True:
        try:
            ws.run_forever()
        except:
            pass

    print('fin')
    print('-- -- -- -- -- -- -- -- -- --')



"""

La reserve de bot ne prends pas en compte la déperdition dut à la prise de benef du bot
les benef du bot sont encore dans la reserve

mettre les try/except dans la fonction trade pour plus de simplicité et de lisilibilité
ajouter dans la fonction try catch et in_long_trade / in_short_trade + nb_trades

"""
