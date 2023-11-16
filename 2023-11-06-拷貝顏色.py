# 2023-11-06-copyColorName.py

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
import pandas as pd
import pyperclip

colorClick = []

def copyClipboard(event):
    clip = event.widget["text"]
    if clip not in colorClick:
        colorClick.append("'"+clip+"'")
        msg = ', '.join(colorClick)
        pyperclip.copy("["+msg+"]")


def clearClipboard(event):
    global colorClick
    colorClick=[]
    pyperclip.copy('')
    messagebox.showinfo('提醒', '已清除剪貼簿內容！')

df1=pd.read_excel(r"tk-colours.xlsx") # Path of the file.
Color_Name = list(df1['Color Name'])
Color_RGB = list(df1['rgb'])
color_dict = dict(zip(Color_Name,Color_RGB))

print(len(color_dict)) # 556
print(color_dict)

def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return round(h,4), round(s,4),round(v,4)

for x in color_dict :
    t1 = tuple(color_dict[x].strip('RGB()').split(','))
    t1 = ( int(t1[0]), int(t1[1]), int(t1[2]) )
    t1 = t1 + rgb2hsv(*t1)
    color_dict[x] = t1      # (240, 248, 255, 208.0, 0.0588, 1.0)

print(len(color_dict)) # 556
print(color_dict)

# setup
window = tk.Tk()
window.title('buttons')
window.geometry('1800x950')

unknown_color = []
for key in color_dict :
    try:
        label = ttk.Label(master=window,background=key)
    except Exception as e:
        str1 = str(e).strip("unknown color name ").strip('"')
        if str1 != 'ist index out of rang':
            unknown_color.append(str1)
print(len(unknown_color)) # 44
print(unknown_color)

for key in unknown_color :
    color_dict.pop(key)
print(len(color_dict)) # 512

sort_list = sorted(color_dict.items(), key = lambda kv:(kv[1][3], kv[1][4], kv[1][5]))
print(sort_list)

for n,item in enumerate(sort_list) : # [ ... ('brown', (165, 42, 42, 0.0, 0.7455, 0.6471))...]

    try:
        if item[1][5] < 0.2  or ( 239 < item[1][3] < 241 and item[1][4] > 0.1) :
            fg = 'white'
        else :
            fg = 'black'
        label = ttk.Label(master=window,
                          text=item[0],
                          background=item[0],
                          foreground=fg,
                          width=16,
                          font=("Arial", 12))
        row = n // 12
        column = n % 12
        label.grid(row=row, column=column)
        label.bind('<Button-1>', copyClipboard)
        label.bind('<Button-3>', clearClipboard)
    except:
        break  # 處理最後一列不足的部分

# run
window.mainloop()


'''

R, G, B是 [0, 255]  H 是[0, 360]   S, V 是 [0, 1]
背景知識：

RGB大家都很熟悉，是紅，綠，藍三色的值。

HSV是什麼呢？ HSV(Hue, Saturation, Value)是根據顏色的直覺特性由A. R. Smith在1978年創建的一種顏色空間, 也稱六角錐體模型(Hexcone Model)。
這個模型中顏色的參數分別是：色調（H），飽和度（S），亮度（V）。
HSV顏色空間模型 
色調H：以角度度量，取值範圍為0°～360°，從紅色開始以逆時針方向計算，紅色為0°，綠色為120°,藍色為240°。
它們的補色是：黃色為60°，青色為180°,品紅為300°；
飽和度S：取值範圍為0.0～1.0；
亮度V：取值範圍為0.0(黑色)～1.0(白色)。 
RGB和CMY顏色模型都是以硬體為導向的，而HSV（Hue Saturation Value）色彩模型是使用者導向的。
HSV模型的三維表示從RGB立方體演化而來。 
設想從RGB沿著立方體對角線的白色頂點向黑色頂點觀察，就可以看到立方體的六角形外形。
六邊形邊界表示色彩，水平軸表示純度，明度沿垂直軸測量。

['olive', 'yellow4', 'yellow3', 'yellow2', 'yellow1']

'''