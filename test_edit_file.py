"""def abc(lst_best_para):
    last_best_price = 0
    lst_buy = []
    lst_sell = []
    lst_x_buy = []
    lst_x_sell = []
    cpt = 0
    nb_last_buy = 0
    short = 0
    nb_last_sell = 0
    lst_pivots2 = []
    sell_activate = 0
    temp = 0
    print('lst best para', lst_best_para)
    for i in lst_pivots:
        temp = 0
        if i == -1:
            # nb_last_sell=para1
            if nb_last_buy >= lst_best_para[-1][0]:  # -----5
                temp = -1
                lst_buy.append(lst_price[cpt])
                lst_x_buy.append(cpt)
                # nb_last_buy=0
                if nb_last_buy >= lst_best_para[-1][1] + lst_best_para[-1][0]:  # --------5
                    nb_last_sell = -lst_best_para[-1][2]  # --------10
                # else:
                #    nb_last_sell=-lst_best_para[0][3]#--------10
            nb_last_buy += 1

        elif i == 1:
            # nb_last_buy=para3
            if nb_last_sell >= lst_best_para[-1][3]:  # -------10
                temp = 1
                lst_sell.append(lst_price[cpt])
                lst_x_sell.append(cpt)

                if nb_last_sell >= lst_best_para[-1][4] + lst_best_para[-1][3]:  # -------10
                    nb_last_buy = -lst_best_para[-1][5]  # --------5
                # else:
                #    nb_last_buy=-lst_best_para[0][7] #--------5
                # nb_last_sell=0
                # pass
            # if nb_last_buy>0: #-------
            #    #nb_last_buy=-2#-------
            #    if nb_last_buy<0:#------
            #        #nb_last_buy=0#------
            #        pass
            nb_last_sell += 1
        lst_pivots2.append(temp)
        cpt += 1

    # -- -- -- -- -- -- -- -- -- --
    buy_part = 10
    cpt = 0

    reserve = 100  # 100
    part = 40  # 10
    bot_fee_part = 0.0
    max_reserve = reserve
    all_fees = 0
    fee = 0.040 / 100
    nb_trade = 0
    lst_buy = []
    lst_total_balance = []
    lst_total_hold = []
    lst_total_bot_benef = []
    temp_bot_benef = 0
    oneone = False
    lst_sell = []
    the_fee = 0.0100 / 100
    cpt_fee = 0
    for i in lst_pivots2:
        temp_bot_benef = 0

        if i == -1:
            if len(lst_sell) > 0:
                nb_trade += 1
                # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
                # price lst_sell
                short += 1
                temp_token_restant = 0
                for i in range(len(lst_sell)):
                    temp_token_restant += buy_part + (buy_part - (lst_sell[i] * lst_price[cpt]))

                if (reserve + temp_token_restant) >= max_reserve:
                    if len(lst_total_balance) > 24:
                        if lst_total_balance[-1] - lst_total_balance[-24] > 0:
                            temp_fees = lst_sell[0] * lst_price[cpt] * fee

                            if (buy_part + buy_part - (lst_sell[0] * lst_price[cpt])) - buy_part > 0:
                                part_benef = buy_part - (lst_sell[0] * lst_price[cpt])
                                reserve += buy_part + buy_part - (
                                            lst_sell[0] * lst_price[cpt]) - part_benef * bot_fee_part
                                temp_bot_benef = part_benef * bot_fee_part
                            else:
                                reserve += buy_part + buy_part - (lst_sell[0] * lst_price[cpt])

                            reserve -= temp_fees
                            all_fees += temp_fees

                            lst_sell.pop(0)
                        else:
                            temp_fees = lst_sell[0] * lst_price[cpt] * fee
                            reserve += lst_sell[0] * lst_price[cpt]
                            reserve -= temp_fees
                            all_fees += temp_fees
                            lst_sell.pop(0)
                    else:
                        temp_fees = lst_sell[0] * lst_price[cpt] * fee
                        reserve += lst_sell[0] * lst_price[cpt]
                        reserve -= temp_fees
                        all_fees += temp_fees
                        lst_sell.pop(0)
                else:
                    temp_fees = lst_sell[0] * lst_price[cpt] * fee
                    reserve += lst_sell[0] * lst_price[cpt]
                    reserve -= temp_fees
                    all_fees += temp_fees
                    lst_sell.pop(0)
                    # if len(lst_buy)>0:
                    #    nb_trade+=1
                    #    temp_fees = lst_buy[0]*lst_price[cpt]*fee
                    #    reserve+=lst_buy[0]*lst_price[cpt]-temp_fees
                    #    all_fees+=temp_fees
                    #    lst_buy.pop(0)
                oneone = False
                # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

            else:
                if oneone == False:
                    buy_part = part
                    if reserve >= part:
                        temp_fees = buy_part * fee
                        nb_trade += 1
                        lst_buy.append(buy_part / lst_price[cpt])
                        reserve -= buy_part
                        reserve -= temp_fees
                        all_fees += temp_fees
                        oneone = True

        elif i == 1:
            if len(lst_buy) > 0:
                temp_token_restant = 0
                for i in range(len(lst_buy)):
                    temp_token_restant += lst_buy[i]
                if (reserve + lst_price[cpt] * temp_token_restant) >= max_reserve:
                    if len(lst_total_balance) > 24:
                        if lst_total_balance[-1] - lst_total_balance[-24] > 0:
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
                oneone = False
            else:
                # -- -- -- -- -- -- -- -- -- --
                if oneone == False:
                    buy_part = part
                    if reserve >= part:
                        temp_fees = buy_part * fee
                        nb_trade += 1
                        lst_sell.append(buy_part / lst_price[cpt])  # price of token if in the production bot
                        reserve -= buy_part
                        reserve -= temp_fees
                        all_fees += temp_fees
                        oneone = True

                # -- -- -- -- -- -- -- -- -- --
                pass

        lst_total_hold.append(max_reserve / lst_price[0] * lst_price[cpt])
        temp_token_restant = 0
        for i in range(len(lst_buy)):
            temp_token_restant += lst_buy[i]

        token_restant = 0
        lst_temp_lst_sell = []
        for i in range(len(lst_sell)):
            lst_temp_lst_sell.append(buy_part + (buy_part - (lst_sell[i] * lst_price[cpt])))
            token_restant += buy_part + (buy_part - (lst_sell[i] * lst_price[cpt]))

        # print('price', lst_price[cpt])
        # print('lst_sell', lst_sell)
        # print('lst_temp_lst_sell', lst_temp_lst_sell)

        if cpt_fee >= 7:
            cpt_fee = 0
            for i in range(len(lst_sell)):
                lst_sell[i] -= lst_sell[i] * the_fee

        else:
            cpt_fee += 1

        if len(lst_total_bot_benef) > 0:
            lst_total_bot_benef.append(temp_bot_benef + lst_total_bot_benef[-1])
        else:
            lst_total_bot_benef.append(temp_bot_benef)
        lst_total_balance.append(reserve + lst_price[cpt] * temp_token_restant + token_restant)
        cpt += 1
    # -- -- -- --
    # print('-- '*20)
    # print('-- '*20)
    # print(f'paramettre: nb_last_sell {para1} : nb_last_buy {para2} : nb_last_buy {para3} : nb_last_sell {para4}')
    # print('-- '*20)
    print('nb trade', nb_trade, 'fees', all_fees)
    print('balance hold', lst_total_hold[-1], 'coef', lst_total_hold[-1] / max_reserve)
    print('balance', lst_total_balance[-1], 'coef', lst_total_balance[-1] / max_reserve)
    print('coef', lst_total_balance[-1] / max_reserve - lst_total_hold[-1] / max_reserve)

    # -- -- -- -- -- -- -- -- -- --
    plt.figure(figsize=(50, 40))
    # plt.yscale("log")
    plt.plot(np.arange(len(lst_total_hold)), lst_total_hold, c='orange')  # hold
    plt.plot(np.arange(len(lst_total_balance)), lst_total_balance, c='blue')  # balance
    ema = ti.ema(np.array(lst_total_balance), 2000)
    plt.plot(np.arange(len(ema)), ema, c='green')  # ema
    ema = ti.ema(np.array(lst_total_hold), 2000)
    plt.plot(np.arange(len(ema)), ema, c='red')  # ema
    plt.plot(np.arange(len(ema)), [max_reserve for i in range(len(ema))])  # reserve
    plt.show()
    plt.figure(figsize=(50, 40))
    plt.plot(np.arange(len(lst_total_bot_benef)), lst_total_bot_benef, c='red')  # benef bot
    plt.plot(np.arange(len(ema)), [part for i in range(len(ema))])  # la part du bot
    plt.plot(np.arange(len(ema)), [max_reserve for i in range(len(ema))])  # reserve
    plt.show()
"""
from binance.client import Client
import datetime
from api_var import api_key, api_secret


client = Client(api_key, api_secret)

Previous_Date = datetime.datetime.today() - datetime.timedelta(days=2)
date1 = Previous_Date.strftime("%d %m, %Y")

date2 = datetime.datetime.now().strftime("%d %m, %Y")

print(date1)
print(date2)


candlesticks = client.get_historical_klines('MKRUSDT', Client.KLINE_INTERVAL_1HOUR, date1, date2)
#ohlc 1, 2, 3, 4
print(len(candlesticks))
print('candlesticks', candlesticks[0])
print('candlesticks', candlesticks[0][1])


#print(client.futures_symbol_ticker(symbol='MKRUSDT'))

#print('balance', client.futures_account_balance()[1]['balance'])
#print('withdraw available', client.futures_account_balance()[1]['withdrawAvailable'])
#{'symbol': 'BTCUSDT', 'markPrice': '44048.20000000', 'indexPrice': '44039.22743253', 'estimatedSettlePrice': '44135.13965055', 'lastFundingRate': '0.00010000', 'interestRate': '0.00010000', 'nextFundingTime': 1628812800000, 'time': 1628807667003}

#price = float(client.futures_mark_price(symbol='MKRUSDT')['markPrice'])
#print(price)
#l_k = client.futures_stream_get_listen_key()
#print(l_k)
#print(client.futures_stream_keepalive(l_k)=={})

#orderid = client.futures_create_order(symbol='MKRUSDT', quantity=str(round(10/price, 3)), side='BUY', type='MARKET')['orderId']

#print(client.futures_create_order(symbol='MKRUSDT', quantity=str(round(10/price, 3)), side='BUY', type='MARKET'))


#print(client.futures_cancel_all_open_orders(symbol='MKRUSDT'))
#print(client.futures_cancel_orders(symbol='MKRUSDT'))
#print(client.futures_cancel_order(symbol='MKRUSDT', orderId=orderid))
#availableBalance


#print(client.futures_symbol_ticker(symbol='MKRUSDT'))

#print(client.futures_account_balance())#[1]['balance'] #withdrawAvailable

#print(client.futures_create_order(symbol='MKRUSDT', quantity=str(round(10/price, 3)), side='SELL', type='MARKET'))


#{'symbol': 'MKRUSDT', 'price': '3615.30', 'time': 1628873118811}
#[{'accountAlias': 'FzsRFzTiXqSgfW', 'asset': 'BNB', 'balance': '0.00000195', 'withdrawAvailable': '0.00000195', 'updateTime': 1628809254000}, {'accountAlias': 'FzsRFzTiXqSgfW', 'asset': 'USDT', 'balance': '26.43606709', 'withdrawAvailable': '26.43606709', 'updateTime': 1628812075748}, {'accountAlias': 'FzsRFzTiXqSgfW', 'asset': 'BUSD', 'balance': '0.00000000', 'withdrawAvailable': '0.00000000', 'updateTime': 0}]
#{'orderId': 2490938944, 'symbol': 'MKRUSDT', 'status': 'NEW', 'clientOrderId': 'RzJmaVgzhxqy7qT6mF0heN', 'price': '0', 'avgPrice': '0.00000', 'origQty': '0.003', 'executedQty': '0', 'cumQty': '0', 'cumQuote': '0', 'timeInForce': 'GTC', 'type': 'MARKET', 'reduceOnly': False, 'closePosition': False, 'side': 'BUY', 'positionSide': 'BOTH', 'stopPrice': '0', 'workingType': 'CONTRACT_PRICE', 'priceProtect': False, 'origType': 'MARKET', 'updateTime': 1628873120850}
#{'symbol': 'MKRUSDT', 'price': '3615.30', 'time': 1628873118811}
#[{'accountAlias': 'FzsRFzTiXqSgfW', 'asset': 'BNB', 'balance': '0.00000195', 'withdrawAvailable': '0.00000195', 'updateTime': 1628809254000}, {'accountAlias': 'FzsRFzTiXqSgfW', 'asset': 'USDT', 'balance': '26.43172862', 'withdrawAvailable': '25.88717013', 'updateTime': 1628873120850}, {'accountAlias': 'FzsRFzTiXqSgfW', 'asset': 'BUSD', 'balance': '0.00000000', 'withdrawAvailable': '0.00000000', 'updateTime': 0}]
#{'orderId': 2490938985, 'symbol': 'MKRUSDT', 'status': 'NEW', 'clientOrderId': '4V07L0eKKRFuKXTnyILe4I', 'price': '0', 'avgPrice': '0.00000', 'origQty': '0.003', 'executedQty': '0', 'cumQty': '0', 'cumQuote': '0', 'timeInForce': 'GTC', 'type': 'MARKET', 'reduceOnly': False, 'closePosition': False, 'side': 'SELL', 'positionSide': 'BOTH', 'stopPrice': '0', 'workingType': 'CONTRACT_PRICE', 'priceProtect': False, 'origType': 'MARKET', 'updateTime': 1628873122254}
