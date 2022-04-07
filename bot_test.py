import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ta.trend import ADXIndicator
from zigzag import *

# from pandas_datareader import get_data_yahoo
import plotly.express as px

import tensorflow as tf
import tulipy as ti
import datetime as dt

from api_var import api_key, api_secret

import time
import binance
from binance.client import Client

client = Client(api_key, api_secret)


# lst_x = [x['symbol'] for x in client.futures_symbol_ticker()]

# print(client.get_asset_balance('BTC'))
# print(float(client.get_asset_balance('BTC')['free']))

# print('price', client.get_symbol_ticker(symbol='BTCUSDT'))
# client.order_market_sell(symbol='BUSDUSDT', quantity=15)

# client.order_market_buy(symbol='BUSDUSDT', quantity=10)


def trade(q, s):
    client.create_order(symbol='XRPUSDT',
                        quantity=q,
                        side=s, type='MARKET')


# print(trade(str(13), 'BUY'))
# print(trade(str(13), 'SELL'))
# trade(str(round(buy_part / ref_price, precision)), 'SELL')
print(client.get_symbol_ticker(symbol='XRPUSDT')['price'])
