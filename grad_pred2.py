import os, sys
import tensorflow as tf
import importlib.util
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, SeparableConv2D
from tensorflow.keras.layers import ELU, LeakyReLU, ReLU, concatenate, multiply, Input, Lambda
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pylab as plt
import cv2
from glob import glob

img_width, img_height = 96, 96
validation_data_dir = 'image'
nb_validation_samples = 1

input_shape = (img_width, img_height, 3) 
og_dir = os.getcwd()
os.chdir(validation_data_dir)
saveImage = []
check = False
for file in (glob('*.png')+ glob('*.jpeg')+ glob('*.jpg')):
    saveImage = cv2.resize(cv2.imread(file), (img_width, img_height))
    temp_image = np.expand_dims(cv2.resize(cv2.imread(file), (img_width, img_height)), axis=0)
    if check == True:
        predict_list = np.concatenate ((predict_list, temp_image))
    else:
        predict_list = np.copy(temp_image)
        check = True
predict_list = np.copy(predict_list/255.0)
os.chdir(og_dir)

def showimage():
    global saveImage
    last_conv_layer = model.get_layer("conv2d_5")
    grads = K.gradients(model.output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output])
    pooled_grads_value, conv_layer_output_value = iterate([img_data])
    for j in range(128):
        conv_layer_output_value[:, :, :, j] *= pooled_grads_value[j]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    img = np.copy(predict_list[i])
    heatmap = cv2.resize(np.float32(heatmap), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite("imageTemp1.png", saveImage)
    cv2.imwrite("imageTemp2.png", heatmap)
    src1 = cv2.imread('imageTemp1.png')
    src2 = cv2.imread('imageTemp2.png')
    superimposed_img = cv2.addWeighted(src1, 0.6, src2, 0.4, 0)
    cv2.imwrite("result.jpg",superimposed_img)
    cv2.imshow("GradCam", cv2.resize(superimposed_img, (512, 512)))
    
model_dir='covid_1.model'
model=load_model(model_dir)
img_data = 0
for i in range (0, nb_validation_samples):
    img_data=np.expand_dims(predict_list[i], 0)
    showimage()
