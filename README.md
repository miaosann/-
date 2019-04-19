##  魔镜魔镜告诉你（Mirror-mirror-tells-you）
> **A keras-based "Face Value" scoring system!**
>
> 代码注释详细，model和数据集过大未能上传，详情邮箱联系

### 1、Training/Testing Set

**训练/测试数据集：** SCUT-FBP5500、SCUT-FBP-500 【其中5500张图片只使用了AF类2000张】

### 2、Benchmark Evaluation

> “We evaluate three different CNN models on SCUT-FBP5500 dataset for facial beauty prediction using two kinds of experimental settings, respectively. These CNNs are trained by initializing parameters with the models pretrained on ImageNet. Three different evaluation metrics are used in our experiments, including: Pearson correlation (PC), maximum absolute error (MAE), root mean square error (RMSE). More experimental details are in our paper. ”
> ![image](https://github.com/miaosann/Mirror-mirror-tells-you/blob/master/images/Results%20of%205-folds%20cross%20validations.png)
> ![image](https://github.com/miaosann/Mirror-mirror-tells-you/blob/master/images/Results%20of%20the%20split%20of%2060%25%20training%20and%2040%25%20testing.png)



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

![ResNet50](https://github.com/miaosann/Mirror-mirror-tells-you/blob/master/images/Net.jpg)



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

![ResNet50](https://github.com/miaosann/Mirror-mirror-tells-you/blob/master/images/ResNet50.png)



### 4、Here are outputs

![zdy](https://github.com/miaosann/Mirror-mirror-tells-you/blob/master/images/zdy.jpg)

![wsf](https://github.com/miaosann/Mirror-mirror-tells-you/blob/master/images/wsf.jpg)



### 5、The Web UI

![UI1](https://github.com/miaosann/Mirror-mirror-tells-you/blob/master/images/UI1.png)

![UI2](https://github.com/miaosann/Mirror-mirror-tells-you/blob/master/images/UI2.jpg)
