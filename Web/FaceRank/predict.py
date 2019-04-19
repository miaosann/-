from __future__ import print_function
from keras.models import load_model
import numpy as np
import sys
import os
import cv2
import csv
from scipy.stats import norm
import FaceRank.face_detection as fd


def save_predict_img(img_file, save_path):
    '''
    对图片进行预测，并将预测后的图片进行保存
    :param img_file：待预测图片路径
    :param save_path：预测后的图片输出路径
    :return：score、percentage、AQ（主要用于Django）
    '''
    img = cv2.imread(img_file)
    img_drawed = fd.draw_faces(img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    faces, coordinates = fd.get_face_image(img)
    score_para = None
    for i in range(len(faces)):
        score = predict_cv_img(faces[i])
        score_para = score[0][0]
        #各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        cv2.putText(img_drawed, str(get_AQ(score)), coordinates[i], font, 0.8, (255, 0, 0), 2)
    cv2.imwrite(save_path, img_drawed)
    return return_AQ(score_para)


def get_percentage(score):
    '''
    得到该图片评分位于训练集中的平均位置,即人群中颜值所处的前百分之多少
    :param score：分数如 2.36751
    :return：前百分之多少
    '''
    for i in range(len(list)):
        if score < float(list[i]):
            return (i + 1.0) / 2000.0


def get_AQ(score):
    '''
    得到颜值（AQ） appearance quantity
    :param score：分数如 2.36751
    :return：appearance quantity
    '''
    score = float(score)
    print("predict: ",score)
    percentage = get_percentage(score)
    print("percentage: ",percentage)
    z_score = norm.ppf(percentage) #累积分布函数的反函数
    return int(100 + (z_score * 24))
    #return z_score

def return_AQ(score):
    '''
    返回颜值（AQ） appearance quantity，该函数用于Django中
    :param score：分数如 2.36751
    :return：score、percentage、AQ
    '''
    score = float(score)
    print("predict: ",score)
    percentage = get_percentage(score)
    print("percentage: ",percentage)
    z_score = norm.ppf(percentage) #累积分布函数的反函数
    return score,percentage,int(100 + (z_score * 24))
    #return z_score


def load_image(file):
    '''
    载入图片
    :param file：图片
    :return：返回expand_dims后的图片矩阵
    '''
    image = cv2.imread(file)
    image = cv2.resize(image, (128, 128))
    image = image / 255  #128*128
    image = np.expand_dims(image, axis=0)  #1*128*128
    return image


def predict_cv_img(img):
    '''
    归一化后预测图片的score（分数如 2.36751）
    :param img：图片矩阵
    :return：score（分数如 2.36751）
    '''
    img = cv2.resize(img, (128, 128))
    img = img / 255
    img = np.expand_dims(img, axis=0)
    return predict(img)


def predict(img):
    '''
    用模型进行预测分数
    :param img:
    :return:
    '''
    score = model.predict(img)
    #print(score)
    return score * 5.0


def training_test():
    filelist = os.listdir('./cache/data/')
    for i in filelist:
        print(i, '  ', predict(load_image('./cache/data/' + i)))


def main():
    for i in sys.argv:
        if i.find('.jpg') != -1:
            print(predict(load_image(i)))


model = load_model('C:/Users/miaohualin/Desktop/Web/FaceRank/model/faceRank.h5')
list = []
with open('C:/Users/miaohualin/Desktop/Web/FaceRank/girl_labels.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        list.append(row['Attractiveness label'])
list.sort()

if __name__ == '__main__':
    #print(get_AQ(1))
    #main()
    # for i in range(1,14):
    # img_file = "C:/Users/miaohualin/Desktop/Web/static/upload/dlrb.jpg"
    # save_path = "C:/Users/miaohualin/Desktop/Web/static/result/dlrb.jpg"
    # a,b,c = save_predict_img(img_file=img_file, save_path=save_path)
    # print("a:",a,"b:",b,"c:",c)
    img = load_image("C:/Users/miaohualin/Desktop/Web/static/upload/dlrb.jpg")
    print(img)
    #img = np.random.rand(1,128,128)
    a = model.predict(img)
    print(a)
