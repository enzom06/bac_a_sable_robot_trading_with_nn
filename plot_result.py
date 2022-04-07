"""import matplotlib.pyplot as plt
import numpy as np
import tulipy as ti
with open('lst_total_hold_liive', 'r') as fic:
    lst_total_hold = list(fic)

with open('lst_total_hold_liive', 'r') as fic:
    lst_total_balance = list(fic)

with open('lst_total_hold_liive', 'r') as fic:
    lst_total_bot_benef = list(fic)

part = int(input('taille de la part'))
max_reserve=int(input('max reserver'))
plt.figure(figsize=(50, 40))
plt.plot(np.arange(len(lst_total_hold)), lst_total_hold, c='orange')  # hold
plt.plot(np.arange(len(lst_total_balance)), lst_total_balance, c='blue')  # balance
plt.plot(np.arange(len(lst_total_bot_benef)), lst_total_bot_benef, c='red')  # benef bot
ema = ti.ema(np.array(lst_total_balance), 2000)
plt.plot(np.arange(len(ema)), ema, c='green')  # ema
plt.plot(np.arange(len(ema)), [part for i in range(len(ema))])
plt.plot(np.arange(len(ema)), [max_reserve for i in range(len(ema))])
plt.show()"""

var1 = 0

print(var1)
def change_var1():
    global var1
    print(f'{str(var1)}')
    var1 +=1
    print(f'{str(var1)}')

change_var1()
print(var1)