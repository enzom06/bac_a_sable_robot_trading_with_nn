# import tulipy as ti
"""

df long
def dif_data = 14

dx1
dx2 -> ne pas borner les data
dx3


lst_x = [24 - ->+] à partir des points donner dans x + 20 et ajouter pour chaque points
lst_y = pivots


"""
# import numpy as np
# print(type(np.array([])))
# import datetime as dt
# import time

# import pandas as pd
# from ta.trend import ADXIndicator
# import pandas as pd
# a = ADXIndicator(pd.Series(2), pd.Series(0), pd.Series(1), 14)

# print(a.adx().values.shape[0])
# a.adx_pos()
# a.adx_neg()


"""def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

print('has number', has_numbers('ZZFZ'))
"""

from binance.client import Client

from api_var import api_key, api_secret


# client = Client(api_key, api_secret)

from threading import Thread as t

import time

# !/usr/bin/python

import time

"""

import threading

def cube(n):
    print(f"Le cube: {n * n * n}")
    threading.Thread.join(t1)

def carre(n):
    print(f"Le carré: {n * n}")

# création de thread
t1 = threading.Thread(target=carre, args=(3,))
t2 = threading.Thread(target=cube, args=(3,))

lst_tread = []
lst_tread.append(t1)


# démarrer le thread t1
t1.start()
#lst_tread.append(t2)
# démarrer le thread t2
t2.start()

for i in lst_tread:
    i.join()

# les deux thread sont exécutés
print("C'est fini!")


print("Exiting Main Thread")"""

# b = client.futures_account_balance()
# print('blalance')
# for i in b:
#    if i ['asset'] == 'USDT':
#        print(i['balance'])
#        print(i['withdrawAvailable'])
# print(b)
# print(b[1]['balance'])
# print(b[1]['withdrawAvailable'])
# client.futures_create_order(symbol='ETHUSDT', quantity=str(0.002), side='BUY',
#                            type='MARKET')

# client.futures_create_order(symbol='ETHUSDT', quantity=str(0.002), side='SELL',
#                            type='MARKET')


# print(time.time())
# time.sleep(1)

# print(time.time()-60*60*2)
# print(dt.timedelta(hours=1))
# print(dt.datetime.today() - dt.timedelta(hours=1))
# Previous_Date = dt.datetime.today() - dt.timedelta(hours=1)
# date1 = Previous_Date.strftime("%d %m, %Y")
# print('date1', date1)

# candlesticks = client.get_historical_klines(TRADE_SYMBOL, Client.KLINE_INTERVAL_1MINUTE, date1, date2)

"""import pickle

lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

with open('lst_nn.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(lst, filehandle)

lst = []

with open('lst_nn.data', 'rb') as filehandle:
    # read the data as binary data stream
    l = pickle.load(filehandle)

print('l', l)"""


#l = [[50,100,200],[50,150,200]]
#for i in l[0]:
#    print('i', (i)/(200))

#reserve = 1000
#print('reserve', reserve)
#reserve -= 100 + (100 * 0.0750 / 100)
#print('reserve', reserve)
#position = round(100 / 10, 5)
#reserve += (position * 5) - (position * 10 * 0.0750 / 100)
#print('reserve', reserve)
#print('position', position)
