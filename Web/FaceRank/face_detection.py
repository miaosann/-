# -*- coding: utf-8 -*-
import cv2

def show(img):
    '''
    openCV显示图片操作
    :param img：read进来的图片矩阵
    :return:
    '''
    cv2.imshow('image', img)
    cv2.waitKey(0)

def get_face_image(img, margin_extend_rate=0.3):
    '''
    识别图片并检测到脸部后返回
    :param img：imread函数进来的图片矩阵
    :param margin_extend_rate：得到人脸的图片大小
    :return: faces：为得到人脸的矩阵数组，list中可有一个至多个人脸矩阵；
             coordinate：为人脸的中心坐标
    '''
    faces = []
    coordinates = []
    faces_coordinate_ = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5)
    )
    # x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
    for (x, y, w, h) in faces_coordinate_:
        x_extend = int(w * margin_extend_rate)
        y_extend = int(h * margin_extend_rate)
        if y-y_extend > 0:
            y_min = y-y_extend
        else:
            y_min = 0

        if y+h+y_extend > img.shape[0]:
            y_max = img.shape[0]
        else:
            y_max = y+h+y_extend

        if x-x_extend > 0:
            x_min = x-x_extend
        else:
            x_min = 0

        if x+w+x_extend > img.shape[1]:
            x_max = img.shape[1]
        else:
            x_max = x+w+x_extend

        roi = img[y_min:y_max, x_min:x_max]
        faces.append(roi)
        coordinates.append((x, y))
        print('FaceDetected')

    return faces, coordinates


def draw_faces(img):
    '''
    给图片中检测到的人脸使用矩形框括起来
    :param img：imread函数进来的图片矩阵
    :return:
    '''
    faces = []
    image = img
    faces_coordinate_ = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5)
    )
    # x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
    for (x, y, w, h) in faces_coordinate_:
        '''
        第一个参数：img是原图
        第二个参数：（x，y）是矩阵的左上点坐标
        第三个参数：（x+w，y+h）是矩阵的右下点坐标 
        第四个参数：（0,255,0）是画线对应的rgb颜色
        第五个参数：2是所画的线的宽度
        '''
        cv2.rectangle(image, (x, y), (x + w, y + w), (0, 255, 0), 2)
    return image

face_cascade = cv2.CascadeClassifier(r'C:/Users/miaohualin/Desktop/Web/FaceRank/haarcascade_frontalface_default.xml')

if __name__ == '__main__':
    # faces,coordinates = get_face_image(cv2.imread('girls.jpg'))
    # print(faces)
    # print("*********************************")
    # print(faces[0].shape)
    # print(faces[1].shape)
    # print("---------------------------------")
    # print(coordinates)
    # for img in faces:
    #    show(img)
    img = draw_faces(cv2.imread('girls.jpg'))
    print(img.shape)
    show(img)
