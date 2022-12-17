---
title: Python滤镜和图像风格迁移
date: 2020-10-14 10:58:04
updated: 2020-10-18 14:08:04

---

# Python滤镜和图像风格迁移

<img src="https://i.loli.net/2020/10/14/DohYj4NOkUwauXQ.png" alt="来自Jotang的题目" style="zoom:50%;" />

## 任务一

### 1.了解滤波器

在百度百科中寻找“图像滤波器”会得到一段看起来**非常让人困惑**的文字：

> 由于成像系统、传输介质和记录设备等的不完善，数字图像在其形成、传输记录过程中往往会受到多种噪声的污染。另外，在图像处理的某些环节当输入的像对象并不如预想时也会在结果图像中引入噪声。这些噪声在图像上常表现为一引起较强视觉效果的孤立像素点或像素块。一般，噪声信号与要研究的对象不相关它以无用的信息形式出现，扰乱图像的可观测信息。对于[数字图像](https://baike.baidu.com/item/数字图像)信号，噪声表为或大或小的极值，这些极值通过加减作用于图像像素的真实[灰度值](https://baike.baidu.com/item/灰度值)上，对图像造成亮、暗点干扰，极大降低了图像质量，影响[图像复原](https://baike.baidu.com/item/图像复原)、分割、[特征提取](https://baike.baidu.com/item/特征提取)、图像识别等后继工作的进行。要构造一种有效抑制噪声的滤波器必须考虑两个基本问题：能有效地去除目标和背景中的噪声;同时，能很好地保护图像目标的形状、大小及特定的几何和拓扑结构特征  。

理解图像滤波器概念，其实质有二：

#### 一、**图像本质上就是各种色彩波的叠加**

图像在计算机里是按照每个位置的像素值存储的，每个像素的颜色，可以用红、绿、蓝、透明度四个值描述，大小范围都是`0 ～ 255`，比如黑色是`[0, 0, 0, 255]`，白色是`[255, 255, 255, 255]`。(也就是我们常用的rgba)

把每一行的像素的rgb值以折线形式绘制出来，就会得到一段图像：

<img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201014114224.png" alt="图源来自网络" style="zoom:67%;" />

**所以可以理解为：图像就是色彩的波动：波动大，就是色彩急剧变化；波动小，就是色彩平滑过渡。**

而滤波器的功能，就是将这些波动的变化进行削弱或者放大，例如物理中的：

> - [低通滤波器](https://baike.baidu.com/item/低通滤波)（lowpass）：减弱或阻隔高频信号，保留低频信号
>
> - [高通滤波器](https://baike.baidu.com/item/高通滤波)（highpass）：减弱或阻隔低频信号，保留高频信号

低通滤波器过滤高频信号，曲线将变得平滑；高通滤波器放大了高频信号，曲线保留下曲折尖锐的部分

在图像中的表现则是：

- 低通滤波器：图像变得模糊（锐度下降）

  <img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201014115512.jpeg" alt="图像对比1" style="zoom:67%;" />

- 高通滤波器：图像只剩下锐度极高的部分，其他部分的色彩丢失

  <img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201014115520.jpeg" alt="图像对比2" style="zoom:67%;" />

  虽然实际应用的滤镜比单纯的高通低通滤波器复杂，但本质上应该也是附加规则的高低通滤波器的组合。

#### 二、理解卷积算法

**PIL**库中的滤镜算法主要涉及到卷积滤镜，即在数字图像的像素矩阵中使用一个n*n的矩阵来滤波(该矩阵即卷积核kernal)，以这个矩阵为单位对图像像素进行遍历，每个输出的像素都是区域像素按照一定权重组合计算出的结果，遍历之后输出的图像就是输出的图像。(即依据“规则”通过每个像素点附近的像素值来修改当前像素点的值，遍历修改后就完成了滤波)

![实现原理图(来自网络)](https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201015142211.gif)

`这张Gif很好的描述了卷积算法的过程，所以被我偷了下来，嘻嘻`

### 2.尝试使用现成的滤波器



阅读Doc容易发现，在**python**的**PIL**库中，**ImageFilter**类下有许多滤波器可以使用：

> **• BLUR**：模糊滤波
>
> **• CONTOUR**：轮廓滤波
>
> **• DETAIL**：细节滤波
>
> **• EDGE_ENHANCE**：边界增强滤波
>
> **• EDGE_ENHANCE_MORE**：边界增强滤波（程度更深）
>
> **• EMBOSS**：浮雕滤波
>
> **• FIND_EDGES**：寻找边界滤波
>
> **• SMOOTH**：平滑滤波
>
> **• SMOOTH_MORE**：平滑滤波（程度更深）
>
> **• SHARPEN**：锐化滤波
>
> **• GaussianBlur(radius=2)**：高斯模糊
>
> ​	\>radius指定平滑半径。
>
> **• UnsharpMask(radius=2, percent=150, threshold=3)**：反锐化掩码滤波
>
> ​	\>radius指定模糊半径；
>
> ​	\>percent指定反锐化强度（百分比）;
>
> ​	\>threshold控制被锐化的最小亮度变化。
>
> **• Kernel(size, kernel, scale=None, offset=0)**：核滤波
>
> ​	当前版本只支持核大小为3x3和5x5的核大小，且图像格式为“L”和“RGB”的图像。
>
> ​	\>size指定核大小（width, height）；
>
> ​	\>kernel指定核权值的序列；
>
> ​	\>scale指定缩放因子；
>
> ​	\>offset指定偏移量，如果使用，则将该值加到缩放后的结果上。
>
> **• RankFilter(size, rank)**：排序滤波
>
> ​	\>size指定滤波核的大小；
>
> ​	\>rank指定选取排在第rank位的像素，若大小为0，则为最小值滤波；若大小为size * size / 2则为中值滤波；若大小为size * size - 1则为最大值滤波。
>
> **• MedianFilter(size=3)**：中值滤波
>
> ​	\>size指定核的大小
>
> **• MinFilter(size=3)**：最小值滤波器
>
> ​	\>size指定核的大小
>
> **•** **MaxFilter(size=3)**：最大值滤波器
>
> ​	\>size指定核的大小
>
> **•** ModeFilter(size=3)**：波形滤波器
>
> ​	选取核内出现频次最高的像素值作为该点像素值，仅出现一次或两次的像素将被忽略，若没有像素出现两次以上，则保留原像素值。
>
> ​	\>size指定核的大小

一段简单的代码可以测试两个滤波器：

```python
from PIL import ImageFilter
from PIL import Image

im = Image.open("test.png")
im_blur = im.filter(ImageFilter.GaussianBlur(radius=5)) # *高斯模糊
im_contour = im.filter(ImageFilter.CONTOUR) # *轮廓滤波
im.show()
im_blur.show()
im_contour.show()
```

可以看到不同滤波器带来的不同的改变：

> <img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201014133724.png" alt="原图" style="zoom: 25%;" /><img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201014133359.png" alt="高斯模糊" style="zoom: 25%;" /><img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201014133834.png" alt="轮廓滤波" style="zoom: 25%;" />

### 3.自己实现卷积滤波器

#### 1.伟大的第一步，确定算法基本思路

1. **初始化设置卷积核**
2. **读取图像二维列表形式存储的像素值**
3. **执行遍历获取新的像素值并存储到新的列表中**

#### 2.尝试实现

初始化设置卷积核：

```python
core = np.array([[1, 1, 0],
               [1, 1, -1],
               [0, -1, -1]])  # *预设值的卷积核
```

读取图像二维列表形式存储的像素值，并分离三通道：

```python
img = plt.imread("t1.jpg")
core_line = core.shape[0] // 2  # *获取卷积核行数
core_row = core.shape[1] // 2  # *获取卷积核列数
# *分别在行前后添加i行,在列前后添加j列,第三维不填充,填充值为0(不会影响原图像)
img = np.pad(img, ((core_line, core_line), (core_row, core_row), (0, 0)), 'constant')
channel_r = img[:, :, 0]  # *R通道像素值
channel_g = img[:, :, 1]  # *G通道像素值
channel_b = img[:, :, 2]  # *B通道像素值
```

执行遍历获取新的像素值并存储到新的列表中：

```python
def calculate(img, core):
    result = (img * core).sum()  # *矩阵乘法获得结果像素值
    if(result < 0):  # *过滤无效像素值
        result = 0
    elif result > 255:
        result = 255
    return result

channel_line = img.shape[0] - core_line + 1  # *获取图像像素点列数
channel_row = img.shape[1] - core_row + 1  # *获取图像像素点行数

new_img = np.zeros((channel_line, channel_row),
                   dtype='uint8')  # *初始化一个和原图像大小相同的用0填充的图像矩阵

for i in trange(channel_line):
    for j in range(channel_row):
        # *调用calculate函数完成每个像素点的滤波计算并赋值给新图像
        new_img[i][j] = calculate(img[i:i+core_line, j:j+core_row], core)
```

最后把三通道合并，将函数分块获得最终代码：

```python
import matplotlib.pyplot as plt
import pylab
import numpy as np
from tqdm import trange


def collect_channel(img, core):
    core_line = core.shape[0] // 2  # *获取卷积核行数
    core_row = core.shape[1] // 2  # *获取卷积核列数
    
    # *分别在行前后添加i行,在列前后添加j列,第三维不填充,填充值为0(不会影响原图像)
    img = np.pad(img, ((core_line, core_line), (core_row, core_row), (0, 0)), 'constant')
    channel_r = convolution(img[:, :, 0], core)  # *提取R通道数据并执行卷积
    channel_g = convolution(img[:, :, 1], core)  # *提取G通道数据并执行卷积
    channel_b = convolution(img[:, :, 2], core)  # *提取B通道数据并执行卷积

    dstack = np.dstack([channel_r, channel_g, channel_b])
    return dstack     # *合并三个颜色通道


def convolution(img, core):

    core_line = core.shape[0]  # *获取卷积核行数
    core_row = core.shape[1]  # *获取卷积核列数

    channel_line = img.shape[0] - core_line + 1  # *获取图像像素点列数
    channel_row = img.shape[1] - core_row + 1  # *获取图像像素点行数

    new_img = np.zeros((channel_line, channel_row),
                       dtype='uint8')  # *初始化一个和原图像大小相同的用0填充的图像矩阵

    for i in trange(channel_line):
        for j in range(channel_row):
            # *调用calculate函数完成每个像素点的滤波计算并赋值给新图像
            new_img[i][j] = calculate(img[i:i+core_line, j:j+core_row], core)
    return new_img


def calculate(img, core):
    result = (img * core).sum()  # *矩阵乘法获得结果像素值
    if(result < 0):  # *过滤无效像素值
        result = 0
    elif result > 255:
        result = 255
    return result


img = plt.imread("t1.jpg")

core = np.array([[1, 1, 0],
               [1, 1, -1],
               [0, -1, -1]])  # *预设的卷积核

result = collect_channel(img, core)
plt.imshow(result)
plt.imsave("D:/0aJotang/#6/results.jpg", result)
pylab.show()
```

对一个憨憨表情包处理后的结果如下：

<img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201015163709.jpg" alt="原图" style="zoom:150%;" /><img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201015163919.jpg" alt="[1, 1, 0],[1, 0, -1],[0, -1, -1]" style="zoom:150%;" /><img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201015164304.jpg" alt="[1, 1, 0],[1, 0, 1],[0, 1, -1]" style="zoom:150%;" /><img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201015164002.jpg" alt="[1, 5, 0],[1, 0, -1],[0, -1, -1]" style="zoom:150%;" />



#### 3.均值模糊/高斯模糊

要使用滤镜达到模糊的效果，我们可以理解为“图像细节的丢失”，但这种丢失不是简单的丢失了像素点，而是像素点和附近像素点的像素值差降低了，也就是更加“平滑”了，这一点和前面提到的“低通滤波器”比较类似。

要降低像素差值，我们可以对每个像素取附近像素点的平均值，这样每个像素值之间的差值就相应减少了。

- **均值模糊**

  ​	直接在卷积遍历过程中求整个矩阵的平均值并赋值给对应像素点：

  `修改求和函数即可做到`

  **效果如图:**

  ```python
  def calculate(img, core):
      result = (img * core).sum() / 9  # *矩阵乘法获得结果像素值并求平均值
      if(result < 0):  # *过滤无效像素值
          result = 0
      elif result > 255:
          result = 255
      return result
  ```

  效果举例：

  <img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201015195410.jpg" alt="原图" style="zoom: 50%;" /><img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201015195841.jpg" alt="3*3平均值卷积内核" style="zoom: 50%;" /><img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201015195443.jpg" alt="5*7平均值卷积内核" style="zoom: 50%;" />

- 高斯模糊

  ​	因为图像像素点分布实际上不是简单分布，每个像素点附近的像素点存在一定的连续性，距离越远，连续性就越不明显，这样的分布特点和正态分布一致，于是有使用正态分布(高斯函数)的方式来模糊处理图像的算法，这样的模糊方法因为过渡更加符合现实情况，在实拍的图片中使用效果会显得更加真实。

  ​	因为是二维的图像，所以需要使用到二维高斯函数：


$$
G(x,y)=\frac1{2\mathrm{πσ}^2}e^\frac{ { }^{-(x^2+y^2)} } {2\sigma^2}
$$

> ​	其中，**σ为模糊量**，因为在正态分布中**σ为方差**，其值越大曲线越扁平，相应的模糊过渡越平滑，所以模糊量越大
>
> ​	此外，卷积核的半径与模糊程度也呈现**正相关关系**

实现起来也并不难，只需要计算出高斯函数的值并转化sum(core)=1就完成了：

```python
import matplotlib.pyplot as plt
from matplotlib.pyplot import pause
import pylab
import numpy as np
from tqdm import trange

# *设置全局变量
def public():
    global core_line, core_row, sigma
    core_line = 10
    core_row = 10
    sigma = 5
    
    
# *计算二维高斯函数值
def gaussian(sigma, x, y):
    z = 1/(2 * np.pi * (sigma**2)) * np.exp(-(x**2+y**2)/(2 * sigma**2))
    return z


def collect_channel(img, core):
    public()
    # *分别在行前后添加i行,在列前后添加j列,第三维不填充,填充值为0(不会影响原图像)
    img = np.pad(img, ((core_line, core_line),
                       (core_row, core_row), (0, 0)), 'constant')
    channel_r = convolution(img[:, :, 0], core)  # *提取R通道数据并执行卷积
    channel_g = convolution(img[:, :, 1], core)  # *提取G通道数据并执行卷积
    channel_b = convolution(img[:, :, 2], core)  # *提取B通道数据并执行卷积

    dstack = np.dstack([channel_r, channel_g, channel_b])
    return dstack     # *合并三个颜色通道


def convolution(img, core):
    public()

    channel_line = img.shape[0] - core_line + 1  # *获取图像像素点列数
    channel_row = img.shape[1] - core_row + 1  # *获取图像像素点行数

    new_img = np.zeros((channel_line, channel_row),
                       dtype='uint8')  # *初始化一个和原图像大小相同的用0填充的图像矩阵

    for i in trange(channel_line):
        for j in range(channel_row):
            # *调用calculate函数完成每个像素点的滤波计算并赋值给新图像
            new_img[i][j] = calculate(img[i:i+core_line, j:j+core_row], core)
    return new_img


def calculate(img, core):

    result = (img * core).sum()  # *矩阵乘法获得结果像素值
    if(result < 0):  # *过滤无效像素值
        result = 0
    elif result > 255:
        result = 255
    return result


def gaussian_cal():
    public()
    # *初始化卷积核
    core = [[1.0 for i in range(core_line)] for j in range(core_row)]
    sums = 0
    # *计算卷积核半径
    for i in range(core_line):
        for j in range(core_row):
            x = abs(j - (core_row // 2))
            y = abs(i - (core_line // 2))
            core[i][j] = gaussian(sigma, x, y)
            sums = sums + core[i][j]
    core = core / sums  # *保证卷积核元素总和为1
    return core


img = plt.imread("D:/0aJotang/#6/nice.jpg")
plt.imshow(img)
result = collect_channel(img, gaussian_cal())
plt.imshow(result)
plt.imsave("D:/0aJotang/#6/results1.jpg", result)
pylab.show()
```

**效果如图:**

<img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201015195410.jpg" alt="原图" style="zoom: 50%;" /><img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201016002018.jpg" alt="sigma=5,core:5*5" style="zoom:50%;" /><img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201016002348.jpg" alt="sigma:20 core:15*15" style="zoom:50%;" />

#### 4.其他滤镜

锐化类滤镜：主要通过强化中心像素值(即赋予高权重)的方式来增强边缘区域的特征

- 浮雕滤镜

  <img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201016004730.jpg" alt="原图" style="zoom: 67%;" /><img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201016004747.jpg" alt="浮雕效果" style="zoom: 67%;" />
  $$
  \begin{array} {ccc}-1&0&0\\0&1&0\\0&0&0\end{array}
  $$

- 轮廓提取

  <img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201016004730.jpg" alt="原图" style="zoom: 67%;" /><img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201016005057.jpg" alt="轮廓提取" style="zoom:67%;" />
  $$
  \begin{array} {ccc}-1&-1&-1\\-1&8&-1\\-1&-1&-1\end{array}
  $$

- 更多滤镜，魔改卷积核······

## 任务二

### 1.认识图片风格迁移

**所谓图片风格迁移，是指利用程序算法学习特定图片的风格，然后再把这种风格应用到另外一张图片上的技术。**

传统的方法是分析某类风格的图像，对其图像特征进行建模，再通过这个模型来应用到目标图像上，缺点是只能针对每一类图像单独建模，而且不同风格的图像建模的方法差异也很大：

<img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201016123905.jpeg" alt="传统风格迁移" style="zoom: 67%;" />

后来出现了基于神经网络学习的风格迁移算法，让程序使用任意一张图片的风格进行风格迁移成为可能：

<img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201016123834.jpeg" alt="神经网络风格迁移算法" style="zoom:50%;" />

基于神经网络的风格迁移算法，大致可以描述为：

> 定义两个表示距离的变量,一个表示输入图片和内容图片的距离(Dc),一个表示输入图片和样式图片的距离(Ds).即Dc测量输入和内容图片的内容差异的距离,Ds则测量输入和样式图片之间样式的差异距离.优化Dc和Ds使之最小,即完成图像风格转移

### 2.使用pytorch实现图片风格迁移

#### 1.核心思想

> 使用CNN(卷积神经网络)提取内容图片的内容和风格图片的风格，然后将这两项特征输入到一张新的图像中。对输入的图像提取出内容和风格与CNN提取的内容和风格进行Loss计算，用MSE度量，然后逐步对Loss进行优化，使Loss值达到最理想，将被优化的参数进行输出，这样输出的图片就达到了风格迁移的目的。

- 计算风格损失和内容损失，并逐步降低梯度优化损失，最后优化参数输出
- 通过预训练的卷积网络提取出更高纬度的图片内容和风格，最后通过定义内容损失函数和风格损失函数进行反向传播更新参数

#### 2.功能分步实现

##### 1.加载图片

```python
# *输出图像大小
imsize = 512
loader = transforms.Compose([
    transforms.Resize(imsize),  # *拉伸调整输入图像尺寸
    transforms.ToTensor()])  # *将图像转换为torch张量

# *图像载入

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0) # *添加伪维数以满足神经网络要求的输入维数
    return image.to(device, torch.float)

# *转换并绘制图像

def imshow(tensor, title=None):
    unloader = transforms.ToPILImage() 
    image = tensor.cpu().clone()  # *clone张量
    image = image.squeeze(0)      # *移除添加的伪维
    image = unloader(image)  # *转换回PIL图像
    plt.imshow(image)  # *绘制图像
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # *暂停使plot子图更新
```

##### 2.计算损失

1. 内容损失
   $$
   L_{content}\left(\overset\rightharpoonup p,\overset\rightharpoonup x,l\right)=\frac12\sum_{i,j} {(F_{i,j}^l-P_{i,j}^l)}^2\\
   $$


   ```python
   # *内容损失
   
   class ContentLoss(nn.Module):
       def __init__(self, target,):
           super(ContentLoss, self).__init__()
           self.target = target.detach()
   
       def forward(self, input):
           self.loss = F.mse_loss(input, self.target)  # *调用mse_loss计算矩阵均方损失
           return input
   
   ```

2. 风格损失

   style loss取自原始image和生成的image在神经网络中的Gram matrix的MSE(Gram矩阵可以在一定程度上反映原始图像的“风格”):

   ![img](https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201018013104.webp)

   `因为公式字母太多写LaTex太麻烦于是就用了图片XD`

   ```python
   # *计算Gram矩阵
   
   def gram_matrix(input):
       a, b, c, d = input.size()  # a=batch size(=1)
       # *特征映射 b=number
       # *(c,d)=dimensions of a f. map (N=c*d)
   
       features = input.view(a * b, c * d)  # *将矩阵F_XL重塑为\hat F_XL
   
       G = torch.mm(features, features.t())  # *计算gram积
   
       # *归一化gram矩阵的值.
       return G.div(a * b * c * d)
   
   # *风格损失计算(与内容损失计算类似)
   
   class StyleLoss(nn.Module):
   
       def __init__(self, target_feature):
           super(StyleLoss, self).__init__()
           self.target = gram_matrix(target_feature).detach()
   
       def forward(self, input):
           G = gram_matrix(input)
           self.loss = F.mse_loss(G, self.target)
           return input
   ```


##### 3.降低梯度

```python
   def get_input_optimizer(input_img):
       optimizer = optim.LBFGS([input_img.requires_grad_()])
       return optimizer
```

##### 4.规范化输入图像以导入nn.Sequential

```python
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # *计算均值、均方差以归一化矩阵
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # *归一化矩阵
        return (img - self.mean) / self.std
```

##### 5.获得风格模型和损失量

```python
# *获得风格模型和损失量


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # *规范化模块
    normalization = Normalization(
        normalization_mean, normalization_std).to(device)
    # *只是为了拥有可迭代的访问权限或列出内容/系统损失
    content_losses = []
    style_losses = []

    # *创建一个新的nn.Sequential来放入按序激活的模块
    model = nn.Sequential(normalization)

    i = 0  # *每次检视层的增量
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # *在下面插入的ContentLoss和StyleLoss的本地版本不能很好地发挥作用。所以在这里替换不合适的
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(
                layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # *加入内容损失:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # *加入风格损失:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # *移除添加的损失层
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses
```

#### 3.完整示例代码

```python
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
# *指定计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# *输出图像大小
imsize = 512
loader = transforms.Compose([
    transforms.Resize(imsize),  # *拉伸调整输入图像尺寸
    transforms.ToTensor()])  # *将图像转换为torch张量

# *图像载入


def image_loader(image_name):
    image = Image.open(image_name)
    # *添加伪维数以满足神经网络要求的输入维数
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader("D:/0aJotang/#6/py_fi/input_style/style.jpg")  # *读取风格图像
content_img = image_loader("D:/0aJotang/#6/py_fi/input_content/content.jpg")  # *读取内容图像
# *确认传入内容和风格图像尺寸一致
assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

unloader = transforms.ToPILImage()  # *将图像转换回PIL—image以在plot绘制

plt.ion()

# *用于转换并绘制图像


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # *clone张量
    image = image.squeeze(0)      # *移除添加的伪维
    image = unloader(image)  # *重新转换回PIL图像
    plt.imshow(image)  # *绘制图像
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # *暂停使plot子图更新


# *内容损失


class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)  # *计算损失
        return input


# *计算gram矩阵

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # *特征映射 b=number
    # *(c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # *将矩阵F_XL重塑为\hat F_XL

    G = torch.mm(features, features.t())  # *计算gram积

    # *归一化gram矩阵的值.
    return G.div(a * b * c * d)


# *风格损失计算

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# *导入预训练模型
cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# *规范化输入图像以导入nn.Sequential


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # *计算均值、均方差以归一化矩阵
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # *归一化矩阵
        return (img - self.mean) / self.std


# *计算样式/内容损失的期望深度层：
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# *获得风格模型和损失量


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # *规范化模块
    normalization = Normalization(
        normalization_mean, normalization_std).to(device)
    # *只是为了拥有可迭代的访问权限或列出内容/系统损失
    content_losses = []
    style_losses = []

    # *创建一个新的nn.Sequential来放入按序激活的模块
    model = nn.Sequential(normalization)

    i = 0  # *每次检视层的增量
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # *对于我们在下面插入的ContentLoss和StyleLoss，
            # *本地版本不能很好地发挥作用。所以我们在这里替换不合适的
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(
                layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # *加入内容损失:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # *加入风格损失:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # *移除添加的损失层
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


input_img = content_img.clone()  # *读取并clone内容图片


# *降低梯度

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

# *运行风格迁移
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # *修正更新的输入图像的值
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # *修正张量为(0,1)
    input_img.data.clamp_(0, 1)

    return input_img


output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')
plt.savefig("D:/0aJotang/#6/py_fi/output/output.png")
plt.ioff()
plt.show()
```

输入图片效果示例：

![风格图像](https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201018020705.png)![内容图像](https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201018020731.png)![输出图像](https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201018020742.png)<img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201018142208.png" alt="input_style" style="zoom: 50%;" /><img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201018142149.png" alt="input_content" style="zoom: 50%;" /><img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201018142219.png" alt="output" style="zoom: 50%;" />
<img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201019110100.jpg" alt="style" style="zoom:67%;" /><img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201019110134.jpg" alt="content-2" style="zoom:67%;" /><img src="https://kevinmatt-1303917904.cos.ap-chengdu.myqcloud.com/20201019110153.png" alt="output" style="zoom:67%;" />



