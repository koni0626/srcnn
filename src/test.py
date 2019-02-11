# coding:UTF-8

from keras.models import Sequential
from keras.layers import Conv2D, Input, BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import SGD, Adam
import numpy as np
import math
import conf
import glob
import os
import cv2
import train


def test_data():
    file_list = glob.glob(os.path.join(conf.TEST_DATA_PATH, "*"))
    x = []
    y = []
    for file_name in file_list:
        try:
            src_img = cv2.imread(file_name)
            src_img = cv2.resize(src_img, conf.IMG_SIZE)
            height = src_img.shape[0]
            width = src_img.shape[1]
            x_data = cv2.resize(src_img,(int(width/2.), int(height/2.)))
            x_data = cv2.resize(x_data, (width, height))/255.
            x.append(x_data)
            y.append(src_img/255.)
        except:
            print(file_name)
    
    return np.array(x), np.array(y)
        


if __name__ == '__main__':
    model = train.srcnn()
    model.load_weights("../weight/srcnn.197-0.00-0.87.hdf5")
    
    test_list = glob.glob(conf.TEST_DATA_PATH+"/*")
    test_list.sort()
    for i, file_name in enumerate(test_list):
        dst_file_name = file_name.split("/")[-1]
        img = cv2.imread(file_name)/255.
        img = np.array([img])
        result = model.predict(img)
        output_filename = "{path}/{file_name}".format(path=conf.RESULT_DATA_PATH, file_name=dst_file_name)
        if not os.path.exists(conf.RESULT_DATA_PATH):
            os.makedirs(conf.RESULT_DATA_PATH)
        print(result[0])
        cv2.imwrite(output_filename, result[0]*255)
        
        
        
        
