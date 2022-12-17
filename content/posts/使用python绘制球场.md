---
title: 使用python绘制传球事件和统计图像
date: 2020-10-13 21:49
updated: 2020-10-13 23:11
---

## 1.绘制一个标准足球场

### 	1.工具：matplotlib库

​		matplotlib库为我们在python中提供了一套绘图的工具。

​		**基本概念**：

​		figure:用于绘制的**面板**（一个窗口）

​		axis:子图（在窗口内指定位置的绘图区域）

### 	2.绘制思路

####	1. 初始化创建面板及子图

```python 
*# \*创建面板*
fig = plt.figure(num=1, dpi=200)
# *添加子图(a*b子图第c位置)
ax = fig.add_subplot(1, 1, 1, facecolor='#33691E') # *facecolor取自Material Design颜色
```

   *初始使用plt.figure时发现最终得到的面板大小不是固定的，阅读doc发现存在参数dpi可以调整显示大小

   ![image-20200928231732962.png](https://www.z4a.net/images/2020/10/09/image-20200928231732962.png)

   *测试后调整为200以在1080p分辨率获得相对合适的窗口大小*

####	2. 两个矩形的绘制   

```python
plt.plot([0, 0], [0, 90],
    color="white", linewidth=2)
plt.plot([0, 130], [90, 90],
    color="white", linewidth=2)
plt.plot([130, 130], [90, 0],
    color="white", linewidth=2)
plt.plot([130, 0], [0, 0],
    color="white", linewidth=2)
plt.plot([65, 65], [0, 90],
    color="white", linewidth=2)
```

   *这里调用plot函数绘制连续点（即线），color控制线条颜色，为了美观设定线条宽度为2*

   绘制效果如图：

<img src="https://www.z4a.net/images/2020/10/08/image-20200928235057831.png" alt="image-20200928235057831.png" style="zoom:33%;" />



####	3. 绘制足球场正中的圆形区域和罚球点

```python
	# *中间的圆
    circle_center = plt.Circle((65, 45), 9.15,
                               color="white", fill=False, linewidth=2)
    circle_spot = plt.Circle((65, 45), 0.8,
                             color="white", linewidth=2)
    ax.add_patch(circle_spot)
    ax.add_patch(circle_center)
```

   *这里使用plt.circle函数直接绘制圆形，设定坐标为球场中点后确定半径使用参数**fill=False**避免填充，罚球点则默认填充*

   *然后使用add_patch函数将两个圆形对象添加到子图ax中*

效果如图：

<img src="https://www.z4a.net/images/2020/10/08/image-20200928235208969.png" alt="image-20200928235208969.png" style="zoom:33%;" />

#### 4. 绘制左右侧禁区及圆弧

```python
    # *左侧禁区
    plt.plot([16.5, 16.5], [65, 25],
             color="white", linewidth=2)
    plt.plot([0, 16.5], [65, 65],
             color="white", linewidth=2)
    plt.plot([16.5, 0], [25, 25],
             color="white", linewidth=2)
    # *右侧禁区
    plt.plot([130, 113.5], [65, 65],
             color="white", linewidth=2)
    plt.plot([113.5, 113.5], [65, 25],
             color="white", linewidth=2)
    plt.plot([113.5, 130], [25, 25],
             color="white", linewidth=2)
    # *左侧圆弧
    left_Arc = Arc((11, 45), height=18.3, width=18.3,
                   angle=0, theta1=310, theta2=50, color="white", linewidth=2)
    ax.add_patch(left_Arc)
    # *右侧圆弧
    right_Arc = Arc((119, 45), height=18.3, width=18.3,
                    angle=0, theta1=130, theta2=230, color="white", linewidth=2)
    ax.add_patch(right_Arc)
    # *左侧小禁区
    plt.plot([0, 5.5], [54, 54],
             color="white", linewidth=2)
    plt.plot([5.5, 5.5], [54, 36],
             color="white", linewidth=2)
    plt.plot([5.5, 0.5], [36, 36],
             color="white", linewidth=2)
    # *右侧小禁区
    plt.plot([130, 124.5], [54, 54],
             color="white", linewidth=2)
    plt.plot([124.5, 124.5], [54, 36],
             color="white", linewidth=2)
    plt.plot([124.5, 130], [36, 36],
             color="white", linewidth=2)
```

绘制圆弧时用到arc函数，传入圆心位置（不实际存在）等参数后，同样通过add_patch添加到子图中，其他区域的绘制与前面相似

绘制结果如图：

<img src="https://www.z4a.net/images/2020/10/08/image-20200928235632502.png" alt="image-20200928235632502.png" style="zoom:33%;" />

#### 5.绘制完成后发现的问题

绘制结束后发现拖动窗口大小会导致坐标轴比例不一致，在阅读doc并利用搜索引擎后，找到了解决办法：

```python
# *调整xy轴比例相等
plt.axis('equal')
```

在初始化子图时加入此参数将默认保持比例相等到结束

## 2.用matplotlib画出所有传球事件并生成视频

### 	1.工具：matplotlib库&pandas库

​	matplotlib自带的animation库可以满足绘制动画的需求，尝试使用python自带的csv库后发现pandas提供的csv文件读取功能更加强大，于是使用pandas

### 	2.读取passingevents.csv的数据

首先分析我们需要用到的数据：

| **传球球员的ID **        | **球员传球时所在的x，y坐标** |
| ------------------------ | ---------------------------- |
| **接球球员的ID **        | **球员接球时所在的x，y坐标** |
| **当前比赛的场次ID**     | **当前事件发生的时间 **      |
| **当前时间所在上下半场** | **传球者团队ID**             |

****

```python
# *读取文件
data = pandas.read_csv(
    "D:/下载/data/2020_Problem_D_DATA/passingevents.csv", index_col=None)
ID = data.MatchID.values
Teamid = data.TeamID.values
OPI = data.OriginPlayerID.values
DPI = data.DestinationPlayerID.values
MP = data.MatchPeriod.values
Time = data.EventTime.values
Type = data.EventSubType.values
Ox = data.EventOrigin_x.values
Oy = data.EventOrigin_y.values
Dx = data.EventDestination_x.values
Dy = data.EventDestination_y.values
```

使用**pandas.read_csv**，添加参数**index_col=None**取消以第一列（MatchID）作为索引，读取到的各列数据以数组形式返回

### 3.使用matplotlib.animation.FuncAnimation绘制视频

#### 	1.函数认知

<img src="https://www.z4a.net/images/2020/10/08/image-20200929003352415.png" alt="image-20200929003352415.png" style="zoom:80%;" />

阅读matplotlib的doc可以看到大量参数，其中**关键的参数**有：

1. <img src="https://www.z4a.net/images/2020/10/08/image-20200929003553779.png" alt="image-20200929003553779.png" style="zoom:80%;" />

   **fig**，用于传入要使用的面板

2. <img src="https://www.z4a.net/images/2020/10/08/image-20200929003728142.png" alt="image-20200929003728142.png" style="zoom:80%;" />

   **func**，用于绘制每一帧画面，**FuncAnimation**将在执行过程中反复调用该函数直到结束以绘制每一帧图像，也是逐帧**绘图的核心函数**

3. <img src="https://www.z4a.net/images/2020/10/08/image-20200929004322865.png" alt="image-20200929004322865.png" style="zoom:80%;" />

   **init_func**，绘制第一帧画面，如果没有该参数，将会以**func**中的第一帧作为初始化帧（在搜索引擎得知，有时不启用此参数可能会导致第一帧无法刷新）

**其他的参数在初始考虑时并没有涉及到，故等到使用时再作解释*

#### 2.伟大的第一步，试图绘制第一帧！

第一帧的绘制其实相当简单，整个过程的图形建立在一个足球场的背景下，所以我们使用之前绘制的标准足球场的代码直接建立init_func()函数：

```python
def init_func():
    plt.plot([0, 0], [0, 90],
                color="white", linewidth=2)
    plt.plot([0, 130], [90, 90],
                color="white", linewidth=2)
    plt.plot([130, 130], [90, 0],
                color="white", linewidth=2)
    plt.plot([130, 0], [0, 0],
                color="white", linewidth=2)
    plt.plot([65, 65], [0, 90],
                color="white", linewidth=2)
    # *中间的圆
    circle_center = plt.Circle((65, 45), 9.15,
                               color="white", fill=False, linewidth=2)
    circle_spot = plt.Circle((65, 45), 0.8,
                             color="white", linewidth=2)
    ax.add_patch(circle_spot)
    ax.add_patch(circle_center)
    # *左侧禁区
    plt.plot([16.5, 16.5], [65, 25],
             color="white", linewidth=2)
    plt.plot([0, 16.5], [65, 65],
             color="white", linewidth=2)
    plt.plot([16.5, 0], [25, 25],
             color="white", linewidth=2)
    # *右侧禁区
    plt.plot([130, 113.5], [65, 65],
             color="white", linewidth=2)
    plt.plot([113.5, 113.5], [65, 25],
             color="white", linewidth=2)
    plt.plot([113.5, 130], [25, 25],
             color="white", linewidth=2)
    # *左侧圆弧
    left_Arc = Arc((11, 45), height=18.3, width=18.3,
                   angle=0, theta1=310, theta2=50, color="white", linewidth=2)
    ax.add_patch(left_Arc)
    # *右侧圆弧
    right_Arc = Arc((119, 45), height=18.3, width=18.3,
                    angle=0, theta1=130, theta2=230, color="white", linewidth=2)
    ax.add_patch(right_Arc)
    # *左侧小禁区
    plt.plot([0, 5.5], [54, 54],
             color="white", linewidth=2)
    plt.plot([5.5, 5.5], [54, 36],
             color="white", linewidth=2)
    plt.plot([5.5, 0.5], [36, 36],
             color="white", linewidth=2)
    # *右侧小禁区
    plt.plot([130, 124.5], [54, 54],
             color="white", linewidth=2)
    plt.plot([124.5, 124.5], [54, 36],
             color="white", linewidth=2)
    plt.plot([124.5, 130], [36, 36],
             color="white", linewidth=2)
```

#### 3.更加伟大的第二步，依据表格数据与绘制建立联系绘制每一帧

一开始我的绘制思路是，在每一次**func**调用中绘制出传球者、接球者的位置（用圆点表示）

通过对csv数据的读取，我们很容易得到了每一次传球事件两个球员的坐标等数据，利用plot函数，首先初始化各对象（最终要以元组的方式返回）：

```python
# *初始化各参数
pos_oriplayer, = plt.plot(Ox[0], Oy[0],
                          "ro", color='red')  # *初始化设置传球者
pos_desplayer, = plt.plot(Dx[0], Dy[0],
                          "ro", color='blue')  # *初始化设置接球者
OPI_name = plt.text(0, 0, '', ha='center', va='top',
                    fontsize=5, color='red')  # *初始化设置传球者ID
DPI_name = plt.text(0, 0, '', ha='center', va='top',
                    fontsize=5, color='blue')  # *初始化设置接球者ID
Time_MatchPeriod = plt.text(0, 0, '',
                            fontsize=10, color='white')  # *初始化设置比赛时间&比赛阶段
text_matchid = plt.text(0, 0, '',
                        fontsize=12, color='blue')  # *初始化设置比赛场次ID
```

**update_frames(num)**传入的参数**num**是可迭代的，在

```python
ani = animation.FuncAnimation(fig, update_frames, interval=150, blit=True, 
                              repeat=False, save_count=23430 * 3, 
                              frames=23430 * 3, init_func=init_func())
```

中，**interval**作为两帧之间的间隔，单位为毫秒(ms)，**blit=True**确定了每一帧的刷新方式，即只绘制变化的内容，这样可以避免已绘制的帧仍然留在画面上的问题（并一定程度减轻性能消耗），**repeat**参数确定在播放完所有帧后是否重复播放。

然后依据获得的数据，调用之前读取的数组进行绘制：

```python
def update_frames(num):
    # *绘制各项参数
    pos_oriplayer.set_data(Ox[num], Oy[num])
    pos_desplayer.set_data(Dx[num], Dy[num])
    OPI_name.set_position((Ox[num], Oy[num]-2))
    OPI_name.set_text("%s" % (OPI[num]))
    DPI_name.set_position((Dx[num], Dy[num]-3))
    DPI_name.set_text("%s" % (DPI[num]))
    Time_MatchPeriod.set_position((80, 95))
    Time_MatchPeriod.set_text("Time:%d  MatchPeriod:%s" % (Time[num], MP[num]))
    text_matchid.set_position((-5, 95))
    text_matchid.set_text("MatchID: %d" % (ID[num]))
    return pos_oriplayer, OPI_name, pos_desplayer,DPI_name, Time_MatchPeriod, text_matchid, 
```

最初版本绘制的效果其实很不理想，我们观看到的动画非常僵硬地逐帧显示了两个不明所以的附带名称的圆点以及位于画面左右上角的时间等数据，以下是我依据回忆复刻代码的100帧gif实例：

<img src="https://www.z4a.net/images/2020/10/08/plswork.gif" alt="plswork.gif" style="zoom:50%;" />

在这里我发现，球员的位置似乎总是在左半场运动，在启用**blit=False**参数后，我看到了每一帧存留的记录，显示结果确实如此，回看csv表格，才发现x，y坐标均在(0,100)，此时才想起来修改**Ox*1.3**，**Oy*0.9**(与我绘制的球场尺寸匹配)

#### 4.优化实现结果

按照题目的要求，此时已经算得上“绘制出所有的传球事件”

但是这样的视频显然缺乏可读性，所以，还要继续优化这个结果。

考虑到事件发生的非连续性，直接绘制运动的球员显然不太显示，但是可以让图像显示的顺序来暗示球传递的方向，依据这个原理，我们可以把原先的n帧画面拆分成3n帧画面：

在3n帧画面中，可分为n组，每一组由3帧组成，依照***传球球员、传球轨迹、接球球员***的顺序逐帧刷新，同时保留该组之前的帧，在结果看起来，就可以产生传球的方向感：

实现起来也很简单（为了更加易读将函数尽可能分块，但应该还有更高效的方法）：

```python
# *获得线段XY范围
def cal_line(num):
    x = np.linspace(Ox[num]*1.3, Dx[num]*1.3)
    y = np.linspace(Oy[num]*0.9, Dy[num]*0.9)
    return x, y

# *绘制帧1
def draw_frame1(num):
    Time_MatchPeriod.set_text(
        "Time:%d  MatchPeriod:%s" % (Time[num], MP[num]))
    text_matchid.set_text("MatchID: %d" % (ID[num]))
    pos_oriplayer.set_data(Ox[num]*1.3, Oy[num]*0.9)
    OPI_name.set_position((Ox[num]*1.3, Oy[num]*0.9-2))
    OPI_name.set_text("%s" % (OPI[num]))
    return pos_oriplayer, OPI_name, Time_MatchPeriod, text_matchid,

# *绘制帧2
def draw_frame2(num):
    time.sleep(0.3)
    Time_MatchPeriod.set_text(
        "Time:%d  MatchPeriod:%s" % (Time[num-1], MP[num-1]))
    text_matchid.set_text("MatchID: %d" % (ID[num-1]))
    text_type.set_position(
        ((Ox[num-1]*1.3+Dx[num-1]*1.3)/2, (Dy[num-1]*0.9+Oy[num-1]*0.9)/2))
    text_type.set_text(Type[num-1])
    pos_oriplayer.set_data(Ox[num-1]*1.3, Oy[num-1]*0.9)
    OPI_name.set_position((Ox[num-1]*1.3, Oy[num-1]*0.9-2))
    OPI_name.set_text("%s" % (OPI[num-1]))
    x, y = cal_line(num-1)
    line.set_data(x, y)
    return pos_oriplayer, OPI_name, Time_MatchPeriod, text_matchid, text_type, line,

# *绘制帧3
def draw_frame3(num):
    Time_MatchPeriod.set_text(
        "Time:%d  MatchPeriod:%s" % (Time[num-2], MP[num-2]))
    text_matchid.set_text("MatchID: %d" % (ID[num-2]))
    pos_desplayer.set_data(Dx[num-2]*1.3, Dy[num-2]*0.9)
    DPI_name.set_position((Dx[num-2]*1.3, Dy[num-2]*0.9-3))
    DPI_name.set_text("%s" % (DPI[num-2]))
    pos_oriplayer.set_data(Ox[num-2]*1.3, Oy[num-2]*0.9)
    OPI_name.set_position((Ox[num-2]*1.3, Oy[num-2]*0.9-2))
    OPI_name.set_text("%s" % (OPI[num-2]))
    x, y = cal_line(num-2)
    line.set_data(x, y)
    return pos_desplayer, DPI_name, pos_oriplayer, OPI_name, Time_MatchPeriod, text_matchid, text_type, line,

# *逐帧绘制的主函数
def draw_one(num):
    if num % 3 == 0:
        return draw_frame1(num)
    elif num % 3 == 1:
        return draw_frame2(num)
    elif num % 3 == 2:
        return draw_frame3(num)

def update_frames(num):
    Time_MatchPeriod.set_position((80, 95))
    text_matchid.set_position((-5, 95))
    return draw_one(num)
```

*其中还添加了一段线段**line**用以更加清晰地表示传球路径，并在**line**中点处添加注释表示传球类型

#### 5.完整代码及结果实例

```python
import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import matplotlib.animation as animation
import time
# *调整字体以保证中文正常输出
font = {'family': 'Microsoft YaHei'}
# *创建面板
fig = plt.figure(num=1, dpi=200)
# *添加子图(a*b子图第c位置)
ax = fig.add_subplot(1, 1, 1, facecolor='#33691E')
# *调整xy轴比例相等
plt.axis('equal')

# *读取文件

data = pandas.read_csv(
    "D:/下载/data/2020_Problem_D_DATA/passingevents.csv", index_col=None)
ID = data.MatchID.values
Teamid = data.TeamID.values
OPI = data.OriginPlayerID.values
DPI = data.DestinationPlayerID.values
MP = data.MatchPeriod.values
Time = data.EventTime.values
Type = data.EventSubType.values
Ox = data.EventOrigin_x.values
Oy = data.EventOrigin_y.values
Dx = data.EventDestination_x.values
Dy = data.EventDestination_y.values


# *绘制基本背景，作为第一帧画面

def init_func():
    plt.plot([0, 0], [0, 90],
                color="white", linewidth=2)
    plt.plot([0, 130], [90, 90],
                color="white", linewidth=2)
    plt.plot([130, 130], [90, 0],
                color="white", linewidth=2)
    plt.plot([130, 0], [0, 0],
                color="white", linewidth=2)
    plt.plot([65, 65], [0, 90],
                color="white", linewidth=2)
    # *中间的圆
    circle_center = plt.Circle((65, 45), 9.15,
                               color="white", fill=False, linewidth=2)
    circle_spot = plt.Circle((65, 45), 0.8,
                             color="white", linewidth=2)
    ax.add_patch(circle_spot)
    ax.add_patch(circle_center)
    # *左侧禁区
    plt.plot([16.5, 16.5], [65, 25],
             color="white", linewidth=2)
    plt.plot([0, 16.5], [65, 65],
             color="white", linewidth=2)
    plt.plot([16.5, 0], [25, 25],
             color="white", linewidth=2)
    # *右侧禁区
    plt.plot([130, 113.5], [65, 65],
             color="white", linewidth=2)
    plt.plot([113.5, 113.5], [65, 25],
             color="white", linewidth=2)
    plt.plot([113.5, 130], [25, 25],
             color="white", linewidth=2)
    # *左侧圆弧
    left_Arc = Arc((11, 45), height=18.3, width=18.3,
                   angle=0, theta1=310, theta2=50, color="white", linewidth=2)
    ax.add_patch(left_Arc)
    # *右侧圆弧
    right_Arc = Arc((119, 45), height=18.3, width=18.3,
                    angle=0, theta1=130, theta2=230, color="white", linewidth=2)
    ax.add_patch(right_Arc)
    # *左侧小禁区
    plt.plot([0, 5.5], [54, 54],
             color="white", linewidth=2)
    plt.plot([5.5, 5.5], [54, 36],
             color="white", linewidth=2)
    plt.plot([5.5, 0.5], [36, 36],
             color="white", linewidth=2)
    # *右侧小禁区
    plt.plot([130, 124.5], [54, 54],
             color="white", linewidth=2)
    plt.plot([124.5, 124.5], [54, 36],
             color="white", linewidth=2)
    plt.plot([124.5, 130], [36, 36],
             color="white", linewidth=2)

# *获得线段XY范围
def cal_line(num):
    x = np.linspace(Ox[num]*1.3, Dx[num]*1.3)
    y = np.linspace(Oy[num]*0.9, Dy[num]*0.9)
    return x, y

# *绘制帧1

def draw_frame1(num):
    Time_MatchPeriod.set_text(
        "Time:%d  MatchPeriod:%s" % (Time[num], MP[num]))
    text_matchid.set_text("MatchID: %d" % (ID[num]))
    pos_oriplayer.set_data(Ox[num]*1.3, Oy[num]*0.9)
    OPI_name.set_position((Ox[num]*1.3, Oy[num]*0.9-2))
    OPI_name.set_text("%s" % (OPI[num]))
    return pos_oriplayer, OPI_name, Time_MatchPeriod, text_matchid,

# *绘制帧2

def draw_frame2(num):
    time.sleep(0.3)
    Time_MatchPeriod.set_text(
        "Time:%d  MatchPeriod:%s" % (Time[num-1], MP[num-1]))
    text_matchid.set_text("MatchID: %d" % (ID[num-1]))
    text_type.set_position(
        ((Ox[num-1]*1.3+Dx[num-1]*1.3)/2, (Dy[num-1]*0.9+Oy[num-1]*0.9)/2))
    text_type.set_text(Type[num-1])
    pos_oriplayer.set_data(Ox[num-1]*1.3, Oy[num-1]*0.9)
    OPI_name.set_position((Ox[num-1]*1.3, Oy[num-1]*0.9-2))
    OPI_name.set_text("%s" % (OPI[num-1]))
    x, y = cal_line(num-1)
    line.set_data(x, y)
    return pos_oriplayer, OPI_name, Time_MatchPeriod, text_matchid, text_type, line,

# *绘制帧3

def draw_frame3(num):
    Time_MatchPeriod.set_text(
        "Time:%d  MatchPeriod:%s" % (Time[num-2], MP[num-2]))
    text_matchid.set_text("MatchID: %d" % (ID[num-2]))
    pos_desplayer.set_data(Dx[num-2]*1.3, Dy[num-2]*0.9)
    DPI_name.set_position((Dx[num-2]*1.3, Dy[num-2]*0.9-3))
    DPI_name.set_text("%s" % (DPI[num-2]))
    pos_oriplayer.set_data(Ox[num-2]*1.3, Oy[num-2]*0.9)
    OPI_name.set_position((Ox[num-2]*1.3, Oy[num-2]*0.9-2))
    OPI_name.set_text("%s" % (OPI[num-2]))
    x, y = cal_line(num-2)
    line.set_data(x, y)
    return pos_desplayer, DPI_name, pos_oriplayer, OPI_name, Time_MatchPeriod, text_matchid, text_type, line,

# *逐帧绘制主函数

def draw_one(num):
    if num % 3 == 0:
        return draw_frame1(num)
    elif num % 3 == 1:
        return draw_frame2(num)
    elif num % 3 == 2:
        return draw_frame3(num)


def update_frames(num):
    Time_MatchPeriod.set_position((80, 95))
    text_matchid.set_position((-5, 95))
    return draw_one(num)
# *初始化位置

line, = plt.plot(0, 0, color='#FFE0B2')  # *初始化线段

pos_oriplayer, = plt.plot(0, 0,
                          "ro", color='#F4511E')  # *初始化设置传球者位置
pos_desplayer, = plt.plot(0, 0,
                          "ro", color='#03A9F4')  # *初始化设置接球者位置
OPI_name = plt.text(0, 0, '', ha='center', va='top',
                    fontsize=5, color='#F4511E')  # *初始化设置传球者ID位置
DPI_name = plt.text(0, 0, '', ha='center', va='top',
                    fontsize=5, color='#03A9F4')  # *初始化设置接球者ID位置
Time_MatchPeriod = plt.text(0, 0, '',
                            fontsize=10, color='white')  # *初始化设置比赛时间&比赛阶段位置
text_matchid = plt.text(0, 0, '',
                        fontsize=10, color='#4DD0E1')  # *初始化设置比赛场次ID位置
text_type = plt.text(0, 0, '', ha='center', va='top',
                     fontsize=6, color='#D81B60')  # *初始化设置传球类型位置

ani = animation.FuncAnimation(
    fig, update_frames, interval=150, blit=True, repeat=False, save_count=500, frames=500, init_func=init_func())  # *绘制动画
ani.save('plswork.gif', writer="ffmpeg", progress_callback=lambda i, n: print(f'Saving frame {i/n*100}%'))  # *利用ffmpeg编码保存动画为mp4,并以百分比形式返回当前进度
plt.show()
```

*因为数据量较大，在保存过程中输出保存进度百分比以了解保存状态

实例(取500帧，间隔150ms，速度较快)：

<img src="https://www.z4a.net/images/2020/10/08/plswork-2.gif" alt="plswork-2.gif" style="zoom: 50%;" />

#### 6.优化更新

采用每一帧不自动擦除的方式绘制，减少了代码量，也提高了可读性，从177行缩减到了120行（含注释空格）

```python
import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import matplotlib.animation as animation
import time

font = {'family': 'Microsoft YaHei'}
# *创建面板
fig = plt.figure(num=1, dpi=200)
# *添加子图(a*b子图第c位置)
ax = fig.add_subplot(1, 1, 1, facecolor='#33691E')
# *调整xy轴比例相等
plt.axis('equal')
# *读取文件
data = pandas.read_csv(
    "D:/下载/data/2020_Problem_D_DATA/passingevents.csv", index_col=None)
ID = data.MatchID.values  # *读取比赛ID
OPI = data.OriginPlayerID.values  # *读取传球者ID
DPI = data.DestinationPlayerID.values  # *读取接球者ID
MP = data.MatchPeriod.values  # *读取比赛阶段(上下半场)
Time = data.EventTime.values  # *读取传球事件时间
Type = data.EventSubType.values  # *读取传球类型
Ox = data.EventOrigin_x.values
Oy = data.EventOrigin_y.values  # *读取传球者x,y坐标
Dx = data.EventDestination_x.values
Dy = data.EventDestination_y.values  # *读取接球者x,y坐标


# *绘制基本背景，作为第一帧画面


def init_func():
    plt.plot([0, 0], [0, 90],
             color="white", linewidth=2)
    plt.plot([0, 130], [90, 90],
             color="white", linewidth=2)
    plt.plot([130, 130], [90, 0],
             color="white", linewidth=2)
    plt.plot([130, 0], [0, 0],
             color="white", linewidth=2)
    plt.plot([65, 65], [0, 90],
             color="white", linewidth=2)
    # *中间的圆
    circle_center = plt.Circle((65, 45), 9.15,
                               color="white", fill=False, linewidth=2)
    circle_spot = plt.Circle((65, 45), 0.8,
                             color="white", linewidth=2)
    ax.add_patch(circle_spot)
    ax.add_patch(circle_center)
    # *左侧禁区
    plt.plot([16.5, 16.5], [65, 25],
             color="white", linewidth=2)
    plt.plot([0, 16.5], [65, 65],
             color="white", linewidth=2)
    plt.plot([16.5, 0], [25, 25],
             color="white", linewidth=2)
    # *右侧禁区
    plt.plot([130, 113.5], [65, 65],
             color="white", linewidth=2)
    plt.plot([113.5, 113.5], [65, 25],
             color="white", linewidth=2)
    plt.plot([113.5, 130], [25, 25],
             color="white", linewidth=2)
    # *左侧圆弧
    left_arc = Arc((11, 45), height=18.3, width=18.3,
                   angle=0, theta1=310, theta2=50, color="white", linewidth=2)
    ax.add_patch(left_arc)
    # *右侧圆弧
    right_arc = Arc((119, 45), height=18.3, width=18.3,
                    angle=0, theta1=130, theta2=230, color="white", linewidth=2)
    ax.add_patch(right_arc)
    # *左侧小禁区
    plt.plot([0, 5.5], [54, 54],
             color="white", linewidth=2)
    plt.plot([5.5, 5.5], [54, 36],
             color="white", linewidth=2)
    plt.plot([5.5, 0.5], [36, 36],
             color="white", linewidth=2)
    # *右侧小禁区
    plt.plot([130, 124.5], [54, 54],
             color="white", linewidth=2)
    plt.plot([124.5, 124.5], [54, 36],
             color="white", linewidth=2)
    plt.plot([124.5, 130], [36, 36],
             color="white", linewidth=2)


def cal_line(num):
    x = np.linspace(Ox[num]*1.3, Dx[num]*1.3)
    y = np.linspace(Oy[num]*0.9, Dy[num]*0.9)
    return x, y


def update_frames(num):
    if num % 3 == 0:
        ax.clear()
        init_func()
        plt.text(80, 95, ("Time:%d  MatchPeriod:%s" % (
            Time[num], MP[num])), fontsize=10, color='white')  # *绘制比赛时间&比赛阶段
        plt.text(-5, 95, ("MatchID: %d" %
                          (ID[num])), fontsize=10, color='#4DD0E1')  # *绘制比赛场次ID
        plt.plot(Ox[num]*1.3, Oy[num]*0.9, "ro", color='#F4511E')  # *绘制传球者
        plt.text(Ox[num]*1.3, Oy[num]*0.9-2, ("%s" % (OPI[num])),
                 ha='center', va='top', fontsize=5, color='#F4511E')  # *绘制传球者ID
    elif num % 3 == 1:
        time.sleep(0.3)
        plt.text((Ox[num-1]*1.3+Dx[num-1]*1.3)/2, (Dy[num-1]*0.9+Oy[num-1]*0.9)/2,
                 Type[num-1], ha='center', va='top', fontsize=6, color='#D81B60')  # *绘制传球类型
        x, y = cal_line(num-1)
        plt.plot(x, y, color='#FFE0B2')  # *绘制线段
    elif num % 3 == 2:
        plt.plot(Dx[num-2]*1.3, Dy[num-2]*0.9, "ro", color='#03A9F4')  # *绘制接球者
        plt.text(Dx[num-2]*1.3, Dy[num-2]*0.9-3, ("%s" % (DPI[num-2])),
                 ha='center', va='top', fontsize=5, color='#03A9F4')  # *绘制接球者ID


ani = animation.FuncAnimation(fig, update_frames, interval=150,
                              blit=False, repeat=False, save_count=500, frames=500)  # *绘制动画
plt.show()

```

可以做到完全相同的效果,而且似乎降低了占用：

<img src="https://i.loli.net/2020/10/14/WeDxtJi7rFspPu6.gif" alt="abc" style="zoom:67%;" />

### 3.绘制所有球员的所有信息统计

#### 1.确定球员所有的信息

翻看readme可以发现，球员具有的事件信息有：

![](https://s1.ax1x.com/2020/10/08/0wsLNj.png)

于是想到用条形图来绘制所有的事件，又因为球员数量较大，所以还是可以使用逐帧绘制动画的方法来完成。

#### 2.绘制每个球员的事件

首先将事件信息存储起来便于直接使用：

```python
Type_name = ['Free Kick', 'Duel', 'Pass', 'Others on the ball', 'Foul', 'Goalkeeper leaving line',
             'Offside', 'Save attempt', 'Shot', 'Substitution', 'Interruption']
```

读取文件：

```python
# *读取文件
data = pd.read_csv(
    "D:/下载/data/2020_Problem_D_DATA/fullevents.csv", index_col=None)
ID = data.MatchID.values
Teamid = data.TeamID.values
OPI = data.OriginPlayerID.values
Type = data.EventType.values
opname = data.Playerid.values # *所有球员的ID(不重复)
```

每一帧的绘制函数：

```python
def update_frames(num):
    ax.clear()
    formatter = FuncFormatter(nums) # *格式化
    ax.yaxis.set_major_formatter(formatter) # *绘制y坐标
    plt.xticks(x, (Type_name)) # *绘制事件类型x轴
    count = [0]*11 # *初始化count用于事件数目存储
    for i in range(len(ID)):
        if OPI[i] == opname[num]:
            for j in range(0, 11):
                if Type[i] == Type_name[j]:
                    count[j] = count[j] + 1
    for a, b in zip(x, count):
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va='bottom', fontsize=20) # *创建数据标签
    plt.bar(x, count)
    plt.title("Player: %s" % opname[num], fontsize=20)
    plt.box(False)
```

绘制动画：

```python
ani = animation.FuncAnimation(
    fig, update_frames, interval=500, blit=False, repeat=False, frames=5000)
```

最终的效果如图：

![status.gif](https://www.z4a.net/images/2020/10/08/status.gif)

**完整代码如下：**

```python
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation

plt.rcParams['font.family'] = 'Microsoft YaHei'

x = np.arange(11)
fig = plt.figure(num=1, dpi=100, figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1)
# *读取文件
data = pd.read_csv(
    "D:/下载/data/2020_Problem_D_DATA/fullevents.csv", index_col=None)
ID = data.MatchID.values
Teamid = data.TeamID.values
OPI = data.OriginPlayerID.values
Type = data.EventType.values
opname = data.Playerid.values

Type_name = ['Free Kick', 'Duel', 'Pass', 'Others on the ball', 'Foul', 'Goalkeeper leaving line',
             'Offside', 'Save attempt', 'Shot', 'Substitution', 'Interruption']




def nums(x, pos):
    'The two args are the value and tick position'
    return x


def update_frames(num):
    ax.clear()
    formatter = FuncFormatter(nums)
    ax.yaxis.set_major_formatter(formatter)
    plt.xticks(x, (Type_name))
    count = [0]*11
    for i in range(len(ID)):
        if OPI[i] == opname[num]:
            for j in range(0, 11):
                if Type[i] == Type_name[j]:
                    count[j] = count[j] + 1
    for a, b in zip(x, count):
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va='bottom', fontsize=20)
    plt.bar(x, count)
    plt.title("Player: %s" % opname[num], fontsize=20)
    plt.box(False)


ani = animation.FuncAnimation(
    fig, update_frames, interval=500, blit=False, repeat=False)
ani.save('status.gif', writer="ffmpeg", progress_callback=lambda i,
         n: print(f'Saving frame {i/n*100}%'))
plt.show()

```

