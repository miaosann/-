from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
import os
import numpy as np
import csv
from keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.layers import Input, Convolution2D, BatchNormalization, Activation
from tensorflow.python.keras import Model

def shape_of_array(arr):
    '''
    求出数组的形状
    :param arr:
    :return:
    '''
    array = np.array(arr)
    return array.shape


def get_label(num):
    '''
    从csv中获取每张图片对应的label,并通过输入num返回lable
    :param num：SCUT-FBP-500中num为图片编号，SCUT-FBP5500中num为名字即可
    :return：lable的具体数值
    '''
    with open('./girl_labels.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        #label
        # for row in reader:
        #     if row['Image'] == str(num):
        #         return float(row['Attractiveness label'])
        for row in reader:
            if row['Image'] == num:
                return float(row['Attractiveness label'])


def load_image_data(filedir):
    '''
    载入图片数据集
    :param filedir：图片数据集所在路径
    :return：img_data为归一化后的图片矩阵list，label为该图片对应的lable的list
    '''
    label = []
    image_data_list = []
    train_image_list = os.listdir(filedir)
    # train_image_list.remove('.DS_Store')
    for img in train_image_list:
        #print(img)
        url = os.path.join(filedir + img)
        image = cv2.imread(url)
        image = cv2.resize(image, (128, 128))
        image_data_list.append(image)
        #按照照片名字切割SCUT-FBP-1，得出1
        #img_num = img.split(".")[0].split("-")[2]

        # 按照照片名字查找label
        img_num = img;
        att_label = get_label(img_num) / 5.0
        print(img_num, '  ', att_label)
        label.append(att_label)

    img_data = np.array(image_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    return img_data, label


def make_network():
    '''
    搭建网络模型
    :return:
    '''
    #the second model based ResNet50
    model = ResNet50(include_top=False, pooling='avg')
    new_model = Sequential()
    new_model.add(model)
    new_model.add(Dense(1, ))
    new_model.summary()

    #the first Model using CNN
    # model = Sequential()
    # model.add(Conv2D(32, (3, 3), padding='same', input_shape=(128, 128, 3)))
    # model.add(Activation('relu'))
    # model.add(Conv2D(32, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    #
    # model.add(Flatten())
    # model.add(Dense(128))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1))

    return new_model


def main():
    train_x, train_y = load_image_data('./cache/girls/')
    model = make_network()

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(batch_size=32, x=train_x, y=train_y, epochs=30)

    #model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    #hist = model.fit(train_x, train_y, batch_size=100, epochs=100, verbose=1)

    model.evaluate(train_x, train_y)
    model.save('model/faceRank.h5')


if __name__ == '__main__':
    main()
