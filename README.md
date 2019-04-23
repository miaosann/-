##  魔镜魔镜告诉你（Mirror-mirror-tells-you）
> **A keras-based "Face Value" scoring system!**
>
> 代码注释详细，model和数据集过大未能上传，详情邮箱联系

**演示视频：** https://www.bilibili.com/video/av50221836

### 1、Training/Testing Set

**训练/测试数据集：** SCUT-FBP5500、SCUT-FBP-500 【其中5500张图片只使用了AF类2000张】

### 2、Benchmark Evaluation

> “We evaluate three different CNN models on SCUT-FBP5500 dataset for facial beauty prediction using two kinds of experimental settings, respectively. These CNNs are trained by initializing parameters with the models pretrained on ImageNet. Three different evaluation metrics are used in our experiments, including: Pearson correlation (PC), maximum absolute error (MAE), root mean square error (RMSE). More experimental details are in our paper. ”
> ![image](https://raw.githubusercontent.com/miaosann/Mirror-mirror-tells-you/master/images/Results%20of%205-folds%20cross%20validations.png)
> ![image](https://raw.githubusercontent.com/miaosann/Mirror-mirror-tells-you/master/images/Results%20of%20the%20split%20of%2060%25%20training%20and%2040%25%20testing.png)



### 3、Model Summary

- **the first Model using CNN** 

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 128, 128, 32)      896       
_________________________________________________________________
activation_1 (Activation)    (None, 128, 128, 32)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 126, 126, 32)      9248      
_________________________________________________________________
activation_2 (Activation)    (None, 126, 126, 32)      0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 63, 63, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 63, 63, 32)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 127008)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               16257152  
_________________________________________________________________
activation_3 (Activation)    (None, 128)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 129       
=================================================================
Total params: 16,267,425
Trainable params: 16,267,425
Non-trainable params: 0
_________________________________________________________________
```
##### 实现代码：

```python
def make_network():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(128, 128, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    return model
```

##### 模型结构图：

![Net](https://raw.githubusercontent.com/miaosann/Mirror-mirror-tells-you/master/images/Net.jpg)



- **the second model based ResNet50**

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
resnet50 (Model)             (None, 2048)              23587712  
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 2049      
=================================================================
Total params: 23,589,761
Trainable params: 23,536,641
Non-trainable params: 53,120
_________________________________________________________________
```

##### 实现代码：

```python
def make_network():
    model = ResNet50(include_top=False, pooling='avg')
    new_model = Sequential()
    new_model.add(model)
    new_model.add(Dense(1, ))
    new_model.summary()

    return new_model
```

##### 模型结构图：

![ResNet50](https://raw.githubusercontent.com/miaosann/Mirror-mirror-tells-you/master/images/ResNet50.png)



### 4、Here are outputs

![zdy](https://raw.githubusercontent.com/miaosann/Mirror-mirror-tells-you/master/images/zdy.jpg)

![wsf](https://raw.githubusercontent.com/miaosann/Mirror-mirror-tells-you/master/images/wsf.jpg)

> ​    在进行后台模型和Django整合阶段会出现问题：Django 初始化加载模型的过程没有问题，但是一旦调用函数使用模型时执行到 `model.predict` 就会报错`ValueError: Tensor Tensor("dense_2/Softmax:0", shape=(?, 8), dtype=float32) is not an element of this graph.`
>
> ​    网上大神们给出解决方案：在初始化加载模型之后，就随便生成一个向量让 `model` 执行一次 `predict` 函数，之后再使用就不会有问题了。至于原因嘛？欸，机器玄学吧！！

### 5、The Web UI

![UI1](https://raw.githubusercontent.com/miaosann/Mirror-mirror-tells-you/master/images/UI1.png)

![UI2](https://raw.githubusercontent.com/miaosann/Mirror-mirror-tells-you/master/images/UI2.jpg)



### 6、Take a photo and scoring

![new1](https://raw.githubusercontent.com/miaosann/Mirror-mirror-tells-you/master/images/new1.jpg)

![new2](https://raw.githubusercontent.com/miaosann/Mirror-mirror-tells-you/master/images/new2.png)

![new3](https://raw.githubusercontent.com/miaosann/Mirror-mirror-tells-you/master/images/new3.png)

**虽然为美女颜值打分，但是实在是找不到女同学检测，无奈只好打开摄像头亲自上阵**


第四部分中界面UI为用户上传图片进行评分，但是为了进一步加强人机交互体验，所以今天我对其进行改进。使用`webcam.js`调用笔记本电脑（手机亦可）自带摄像头，对用户进行拍照，并将照片显示在右侧的`Canvas`画板中，供用户观看照片效果，然后跳转至结果界面。

> ​    在进行前端代码编写阶段会出现报错：webcam.capture is not a function
>
> ​    这是因为测试html文件必须使用http请求方式打开，不可以通过本地file://请求直接打开，所以我们使用   Python3自带的简易服务器，先打开`powershell`进入html文件目录，然后使用命令`python -m http.server 8888 `启动服务器。



### 7、The Evaluation of Model1 and Model2

- **model1**

  此模型一般打分会偏高，美女分高实属正常，但是一些颜值很差得人分也容易出现80+甚至90+【宋小宝雨露均沾女装：84，王云：92】，所以认为此模型效果略有欠缺，但是勉强可用。

  总结为该模型：`喜欢美女，但是考虑丑女感受`，属于左右逢源型。

- **model2**

  此模型是在model1觉得有所欠缺后，针对论文中提到的`ResNet50`网络而进行训练的，经过实际使用，效果明显好于model1，所以我用其给高中和大学一部分女同学进行评分，效果近似于我心中的评分。因此，我自己使用的模型是model2。但是偶尔也会存在事与愿违，比如一个我觉得长相甜美可爱的女同学，分数过于低了，除了她以外对于大多数人效果还是不错的。

  总结为该模型：`实是求是，喜欢美女，但是不放过丑女`，属于铁面无私型。



