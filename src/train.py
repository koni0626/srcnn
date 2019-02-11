# coding:UTF-8


from keras.models import Sequential
from keras.layers import Conv2D, Input, BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import SGD, Adam
from keras import backend as K
import numpy as np
import math
import conf
import glob
import os
import cv2


def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

def psnr(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                         " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                   str(y_pred.shape))

    return -10. * np.log10(np.mean(np.square(y_pred - y_true)))
    
    
def srcnn():
    SRCNN = Sequential()

    SRCNN.add(Conv2D(nb_filter=128, nb_row=9, nb_col=9, init='he_normal',
                     activation='relu', border_mode='same',input_shape=conf.INPUT_SIZE))

    SRCNN.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='he_normal',
                     activation='relu', border_mode='same'))
                     
    SRCNN.add(Conv2D(nb_filter=32, nb_row=1, nb_col=1, init='he_normal',
                     activation='relu', border_mode='same'))

    SRCNN.add(Conv2D(nb_filter=3, nb_row=5, nb_col=5, init='he_normal',
                     activation='linear', border_mode='same'))
                     
    adam = Adam(lr=1e-5)
    SRCNN.compile(optimizer=adam, loss='mse', metrics=['acc'])
    return SRCNN


def train_data():
    file_list = glob.glob(os.path.join(conf.TRAIN_DATA_PATH, "*"))
    x = []
    y = []
    for file_name in file_list:
        try:
            src_img = cv2.imread(file_name)
            #src_img = cv2.resize(src_img, conf.IMG_SIZE)
            height = src_img.shape[0]
            width = src_img.shape[1]
            x_data = cv2.resize(src_img,(int(width/2.), int(height/2.)))
            x_data = cv2.resize(x_data, (width, height))/255.
            x.append(x_data)
            src_img = src_img/255.
            y.append(src_img)
        except:
            print(file_name)
    
    return np.array(x), np.array(y)
        

if __name__ == '__main__':
    model = srcnn()
    print(model.summary())
    cp_cb = ModelCheckpoint(filepath=conf.SAVE_WEIGHT_FILE, monitor='acc', verbose=1, save_best_only=True, mode='auto')
    cp_his = CSVLogger(conf.LOG_FILE)
    x, y = train_data()
    model.fit(x, y, batch_size=32,
              callbacks=[cp_cb, cp_his], shuffle=True, nb_epoch=200, verbose=1)
    
