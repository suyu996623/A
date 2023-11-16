# 所有通道

from pandas_datareader import data as pdr
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import mpl_finance as mpf
from talib import abstract  # 技術指標用
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')

yf.pdr_override()
y_symbols = ['2330.TW']
start_date = datetime(2023,1,1)
end_date = datetime(2023,11,7)
df = pdr.get_data_yahoo(y_symbols,start=start_date,end=end_date)

df['ma5'] = abstract.SMA(np.array(df['Close']),5)
df['ma10'] = abstract.SMA(np.array(df['Close']),10)
df['ma20'] = abstract.SMA(np.array(df['Close']),20)
df['ma60'] = abstract.SMA(np.array(df['Close']),60)

df_bbnds = abstract.BBANDS(np.array(df['Close']), timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)

# 根據你的需求計算KD線、MACD和RSI
low_list = df['Low'].rolling(9, min_periods=9).min()
high_list = df['High'].rolling(9, min_periods=9).max()
rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
df["RSV"]=rsv
# 計算K, D, J值
df['%K'] = pd.DataFrame(rsv).ewm(com=2).mean()
df['%D'] = df['%K'].ewm(com=2).mean()
df['%J'] = 3*df['%D'] - 2*df['%K']
#df_kd = abstract.STOCH(df['High'], df['Low'], df['Close'], fastk_period=9, slowk_period=3, slowd_period=3)
#df['%K'] = df_kd[0]
#df['%D'] = df_kd[1]
#df['%J'] = 3*df_kd[1]-2*df_kd[0]
#df['3K-2D'] = 3*df_kd[0]-2*df_kd[1]
#df['RSV'] = (df['Close']-df['Close'].rolling(window=9).min())/(df['Close'].rolling(window=9).max()-df['Close'].rolling(window=9).min())*100
#技術指標-WILLR值
df['%R'] = abstract.WILLR(df['High'], df['Low'], df['Close'], timeperiod=9)

df_macd = abstract.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['MACD'] = df_macd[0]
df['MACD Signal'] = df_macd[1]
df['MACD Histogram'] = df_macd[2]

df['RSI5'] = abstract.RSI(df['Close'],timeperiod=5)
df['RSI10'] = abstract.RSI(df['Close'],timeperiod=10)

df['BIAS10'] = (df['Close']- df['ma10'])/df['ma10']
df['BIAS20'] = (df['Close']- df['ma20'])/df['ma20']
df['BIAS10_20'] = df['BIAS10'] - df['BIAS20']

df.index = df.index.format(formatter=lambda x : x.strftime('%y-%m-%d'))
fig = plt.figure(figsize=(16,18), layout='constrained')
ax1 = fig.add_subplot(9,1,(1,3))  # 共七列一行  我們要占用一二列
ax1.set_xticks(range(0,len(df.index),30))
ax1.set_xticklabels(df.index[::30])
mpf.candlestick2_ochl(ax1, df['Open'], df['Close'], df['High'], df['Low'], width=0.8, colorup='r', colordown='k', alpha=1)
ax1.plot(df['ma5'],label='5日均線',alpha=0.9)
ax1.plot(df['ma20'],label='20日均線',alpha=0.9,color='r')
ax1.plot(df['ma60'],label='60日均線',alpha=0.9,color='purple')
ax1.plot(df_bbnds[0], label='upperband',alpha=0.9,color='g')
ax1.plot(df_bbnds[1], label='middleband',alpha=0.9)
ax1.plot(df_bbnds[2], label='lowerband',alpha=0.9,color='g')
ax1.legend(loc=2)

ax2 = fig.add_subplot(9,1,4)  # 共七列一行  我們要占用第四列
ax2.set_xticks(range(0,len(df.index),30))
ax2.set_xticklabels(df.index[::30])
title_bbox = dict(boxstyle='square', facecolor='white', edgecolor='blue', alpha=0.7)
ax2.set_title('成交量',bbox=title_bbox, x=0.04, y=0.5)
mpf.volume_overlay(ax2, df['Open'], df['Close'], df['Volume'], colorup='r', colordown='g', width=0.5, alpha=0.8)

# 新增一個子圖來顯示KD線
ax3 = fig.add_subplot(9,1,5)
ax3.plot(df['%K'], label='K line',color='b')
ax3.plot(df['%D'], label='D line',color='r')
ax3.plot(df['%J'], label='J line',linestyle='--')
ax3.set_xticks(range(0,len(df.index),30))
ax3.set_xticklabels(df.index[::30])
ax3.legend(loc=2)

# 新增一個子圖來顯示MACD
ax4 = fig.add_subplot(9,1,6)
ax4.plot(df['MACD'], label='MACD',color='orange')
ax4.plot(df['MACD Signal'], label='MACD Signal',color='blue',alpha=0.8)
macd_colors = np.where(df['MACD Histogram'] >= 0, 'r', 'g')
ax4.bar(df.index, df['MACD Histogram'], label='MACD Hist', color=macd_colors)
ax4.set_xticks(range(0,len(df.index),30))
ax4.set_xticklabels(df.index[::30])
ax4.legend(loc=2)

# 新增一個子圖來顯示RSI
ax5 = fig.add_subplot(9,1,7)
ax5.plot(df['RSI5'], label='RSI5')
ax5.plot(df['RSI10'], label='RSI10')
ax5.set_xticks(range(0,len(df.index),30))
ax5.set_xticklabels(df.index[::30])
ax5.legend(loc=2)

# 新增一個子圖來顯示BIAS
ax6 = fig.add_subplot(9,1,8)
ax6.plot(df['BIAS10'], label='BIAS10',color= 'blue',alpha=0.8)
ax6.plot(df['BIAS20'], label='BIAS20',color= 'orange')
bias_colors = np.where(df['BIAS10_20'] >= 0, 'r', 'g')
ax6.bar(df.index, df['BIAS10_20'], label='BIAS10-20 Hist', color=bias_colors)
ax6.set_xticks(range(0,len(df.index),30))
ax6.set_xticklabels(df.index[::30])
ax6.legend(loc=2)
# 新增一個子圖來顯示WILLR值
ax7 = fig.add_subplot(9,1,9)
ax7.plot(df['%R'], label='WILLR 9')
ax7.set_xticks(range(0,len(df.index),30))
ax7.set_xticklabels(df.index[::30])
ax7.legend(loc='lower left')
plt.show()