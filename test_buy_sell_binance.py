from binance.client import Client
from api_var import api_key, api_secret


client = Client(api_key, api_secret, {"timeout": 20})


print(client.get_asset_balance(asset='USDT'))
order1 = client.order_market_buy(
    symbol='BTCUSDT',
    quoteOrderQty=20)
print(order1)
print(client.get_asset_balance(asset='USDT'))
order2 = client.order_market_sell(
    symbol='BTCUSDT',
    quantity=order1['executedQty'])
print(order2)


#{'asset': 'USDT', 'free': '30.91886496', 'locked': '43.60500000'}


#{'asset': 'USDT', 'free': '20.95517649', 'locked': '43.60500000'}

#{'symbol': 'BTCUSDT', 'orderId': 6943860577, 'orderListId': -1, 'clientOrderId': '079VcSXAYo537atn44SWAQ', 'transactTime': 1627334821742, 'price': '0.00000000', 'origQty': '0.00053200', 'executedQty': '0.00053200', 'cummulativeQuoteQty': '19.98373944', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'MARKET', 'side': 'SELL', 'fills': [{'price': '37563.42000000', 'qty': '0.00053200', 'commission': '0.00004886', 'commissionAsset': 'BNB', 'tradeId': 970981827}]}


#-- -- -- -- -- -- -- -- -- --



#o = {'symbol': 'BTCUSDT', 'orderId': 6941426098, 'orderListId': -1, 'clientOrderId': 'RYunbdwTs5ZHUAdmwCINI1', 'transactTime': 1627326230163, 'price': '0.00000000', 'origQty': '0.00025100', 'executedQty': '0.00025100', 'cummulativeQuoteQty': '9.96368847', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'MARKET', 'side': 'BUY', 'fills': [{'price': '39695.97000000', 'qty': '0.00025100', 'commission': '0.00002325', 'commissionAsset': 'BNB', 'tradeId': 970404567}]}


#btc_recup = o['executedQty'] # ordre executé
#all_fees= o['fills'][0]['commission'] # frais en bnb
