
import pandas as pd
import os
import talib
import numpy as np

path = "C:/Users/suyu/Desktop/suv/2330/"
filename = os.listdir(path)
pathfile = [path+item for item in filename]
pathfile.sort()
print(pathfile)

list1 = []
for item in pathfile:
    df = pd.read_csv(item, encoding='big5')  # 讀取 CSV 檔
    df = df.reset_index()  # 重新設置列索引  原本的列索引會全部變成欄位
    title = df.columns[-1]
    df = df.drop(columns=[title])  # 刪除指定行索引的資料行
    df.columns = df.loc[0, :]
    df = df.drop([0])
    df = df[df['日期'].apply(lambda x: x.split('/')[0].isdigit())]
    df = df.drop(['成交金額', '漲跌價差', '成交筆數'], axis=1)  # 刪除指定行索引的資料行
    df['成交股數'] = df['成交股數'].str.split(',').str.join('').astype(int)  # 將一個欄位轉型
    df[['開盤價', '最高價', '最低價', '收盤價']] = df[['開盤價', '最高價', '最低價', '收盤價']].astype(float)  # 將數個欄位轉型
    list1.append(df)

df1 = pd.concat(list1, ignore_index=True)

df1['5_SMA'] = talib.SMA(np.array(df1['收盤價']), 5)      # <class 'numpy.ndarray'>
df1['20_SMA'] = talib.SMA(np.array(df1['收盤價']), 20)    # <class 'numpy.ndarray'>
df1['60_SMA'] = talib.SMA(np.array(df1['收盤價']), 60)    # <class 'numpy.ndarray'>

df1.to_excel("2330.xlsx")

