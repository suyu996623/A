# 布林通道

from pandas_datareader import data as pdr
import yfinance as yf
from datetime import datetime

yf.pdr_override()
#y_symbols = ['2330.TW','00738U.TW','BTC-USD','SI=F','BZ=F']
y_symbols = ['00655L.TW']
start_date = datetime(2023,1,1)
end_date = datetime(2023,11,7)
df = pdr.get_data_yahoo(y_symbols,start=start_date,end=end_date)

import matplotlib.pyplot as plt
import mpl_finance as mpf
from talib import abstract  # 技術指標用
import numpy as np
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')

# sma_5 = talib.SMA(np.array(df['Close']),5)
# sma_20 = talib.SMA(np.array(df['Close']),20)
# sma_60 = talib.SMA(np.array(df['Close']),60)

df_bbnds = abstract.BBANDS(np.array(df['Close']), timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)

df.index = df.index.format(formatter=lambda x : x.strftime('%Y-%m-%d'))

# recttuple (left, bottom, width, height)
# The dimensions (left, bottom, width, height) of the new Axes.
# All quantities are in fractions of figure width and height.

fig = plt.figure(figsize=(16,7), layout='constrained')
ax1 = fig.add_axes((0.05,0.4,0.9,0.55))
ax1.set_xticks(range(0,len(df.index),30))
ax1.set_xticklabels(df.index[::30])
mpf.candlestick2_ochl(ax1, df['Open'], df['Close'], df['High'], df['Low'], width=0.8, colorup='r', colordown='k', alpha=1)
# ax.plot(sma_5,label='5日均線',alpha=0.9)
# ax.plot(sma_20,label='20日均線',alpha=0.9)
# ax.plot(sma_60,label='60日均線',alpha=0.9)
# ax.legend()
ax1.plot(df_bbnds[0], label='upperband',alpha=0.9)
ax1.plot(df_bbnds[1], label='middleband',alpha=0.9)
ax1.plot(df_bbnds[2], label='lowerband',alpha=0.9)
ax1.legend(loc=3) 

ax2 = fig.add_axes((0.05,0.05,0.9,0.25))
ax2.set_xticks(range(0,len(df.index),30))
ax2.set_xticklabels(df.index[::30])
mpf.volume_overlay(ax2, df['Open'], df['Close'], df['Volume'], colorup='r', colordown='g', width=0.5, alpha=0.8)

plt.show()
