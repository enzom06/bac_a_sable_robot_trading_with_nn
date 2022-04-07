from tradingview_ta import TA_Handler, Interval, Exchange
import pickle, time, os
tesla = TA_Handler(
    symbol="TSLA",
    screener="america",
    exchange="NASDAQ",
    interval=Interval.INTERVAL_1_DAY,
    # proxies={'http': 'http://example.com:8080'} # Uncomment to enable proxy (replace the URL).
)

#looop wait 1s
#if time in range 5s wit the range want then pass elese wait and retest

oscillators = tesla.get_analysis().oscillators
moving_averages = tesla.get_analysis().moving_averages
t = time.time()
with open('list_dico_data_tradingview.data', 'rb') as fic:
    lst_temp = pickle.load(fic)

lst_temp.append([oscillators, moving_averages, t])

with open('list_dico_data_tradingview.data', 'wb') as fic:
    pickle.dump(lst_temp, fic)

# Example output: {"RECOMMENDATION": "BUY", "BUY": 8, "NEUTRAL": 6, "SELL": 3}


