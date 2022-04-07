'''#faire un bot loop qui récupère la data
#https://www.cryptodatadownload.com/data/binance/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from zigzag import *

import plotly.express as px


import tensorflow as tf
import tulipy as ti
import time


#met just dans

#lst des nouv data
#lst=['BTC', 'ETH', 'BNB']
#for i in lst:
#    df = pd.read_csv(f'./Binance_{i}USDT_1h.csv')
#    df = df[::-1]
#    df.to_csv(f'./Binance_{i}USDT_1h.csv')
#

from binance import ThreadedWebsocketManager
data_frame = pd.read_csv('btc_usdt_1h_live.csv')


def live_update(_symbole):
    symbol = _symbole

    twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
    twm.start()

    def handle_socket_message(msg):
        print(msg)
        # voir ce que renvoie le kline et mettre dans la liste
        with open('data_temp_test.csv', 'a') as f:
            pd.DataFrame([[['Raphael'], ['red'], ['bo staff']]]).to_csv(f, header=False)

    twm.start_kline_socket(callback=handle_socket_message, symbol=symbol, '1h')

    twm.join()
symbole='BTCUSDT'
live_update(symbole)'''


import websocket

def on_close(ws):
    # print('disconnected from server')
    print("Retry : %s" % time.ctime())
    time.sleep(10)
    connect_websocket()  # retry per 10 seconds


def on_open(ws):
    print('connection established')


def on_message(ws, message):
    print('message', message)
SOCKET = "wss://stream.binance.com:9443/ws/ethusdt@kline_1m"
ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
ws.run_forever()