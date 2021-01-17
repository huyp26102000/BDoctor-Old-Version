import os, sys
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
print ('importing')
import tensorflow as tf
import importlib.util
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
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
from PIL import Image
tf.keras.backend.set_learning_phase(0)



# image = cv2.imread(args["image"])
# output = imutils.resize(image, width=400)
 
# # pre-process the image for classification
# image = cv2.resize(image, (256, 256))
# image = image.astype("float") / 255.0
# image = img_to_array(image)
# image = np.expand_dims(image, axis=0)



img_width, img_height = 96, 96
validation_data_dir = 'image'

nb_validation_samples = 1
batch_size = 1


input_shape = (img_width, img_height, 3) #BLACK WHITE IMAGE
test_datagen = ImageDataGenerator(rescale=1. / 255)
predict_list = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='binary',
    # classes = 
    shuffle=False)
test_X, test_Y = next(predict_list)

def showimage ():
    #last_conv_layer = model.get_layer("multiply")
    last_conv_layer = model.get_layer("conv2d_5")
    grads = K.gradients(model.output, last_conv_layer.output)[0]
    print (grads,' grads')
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output])
    print(iterate)
    pooled_grads_value, conv_layer_output_value = iterate([img_data])
    print (pooled_grads_value.shape,' pooled_grads')
    print (conv_layer_output_value.shape,' conv_layer_output_value')
    for j in range(128):
        conv_layer_output_value[:, :, :, j] *= pooled_grads_value[j]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    #print (heatmap.shape)
    heatmap = np.squeeze(heatmap)
    #print (heatmap.shape,' squeezed')
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    #print(heatmap)
    #print (validation_data_dir+'/'+predict_list.filenames[i])
    img = cv2.imread(validation_data_dir+'/'+predict_list.filenames[i])
    #print(heatmap.dtype)
    #print (heatmap.shape,' truoc resize')
    heatmap = cv2.resize(np.float32(heatmap), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    #print(heatmap.dtype)
    #print (heatmap.shape)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imshow(predict_list.filenames[i] ,cv2.resize(img, (512, 512)))
    cv2.imshow("GradCam", cv2.resize(superimposed_img, (512, 512)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    



#model_dir='Dataset_Categorized/Chinaset/smoll_Model/MobilenetV2_weights_best.h5'
model_dir='covid_1.model'
#model_dir='Dataset_Categorized/Chinaset/Model/Big_Model_weights_best.h5'
#model_dir='Dataset_Categorized/Chinaset/Model/kaggle_vgg16.h5'



model=load_model(model_dir)

print (model.summary())
# ggghhey
"""
for attn_layer in model.layers:
    c_shape = attn_layer.get_output_shape_at(0)
    if len(c_shape)==4:
        if c_shape[-1]==1:
            print(attn_layer)
            break

rand_idx = np.random.choice(range(len(test_X)), size = 6)
attn_func = K.function(inputs = [model.get_input_at(0), K.learning_phase()],
           outputs = [attn_layer.get_output_at(0)]
          )
fig, m_axs = plt.subplots(len(rand_idx), 2, figsize = (8, 4*len(rand_idx)))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for c_idx, (img_ax, attn_ax) in zip(rand_idx, m_axs):
    cur_img = test_X[c_idx:(c_idx+1)]
    attn_img = attn_func([cur_img, 0])[0]
    img_ax.imshow(cur_img[0,:,:,0], cmap = 'bone')
    attn_ax.imshow(attn_img[0, :, :, 0], cmap = 'viridis', 
                   vmin = 0, vmax = 1, 
                   interpolation = 'lanczos')
    real_label = test_Y[c_idx]
    img_ax.set_title('TB\nClass:%s' % (real_label))
    pred_confidence = model.predict(cur_img)[0]
    attn_ax.set_title('Attention Map\nPred:%s' % (pred_confidence[0]))
fig.savefig('attention_map.png', dpi = 300)"""

#print (model.evaluate(predict_list, verbose=0))

#print (predict_list.filenames)
#print (model.predict_generator(predict_list [1], steps=1)
"""
for i in range (0, 2):
    img_data=predict_list.next()
    print (img_data.ndim)
""""""
print (predict_list)
"""
#gggaaayyy

for i in range (0, nb_validation_samples-1):
    img_data=predict_list.next()
    #print (img_data[0].shape,' test')
# proba = model.predict(image)[0]
# idxs = np.argsort(proba)[::-1][:2]
    showimage()
    #print (i,' ',predict_list.filenames[i],' ',model.predict (img_data) [0][1])
    # if result [0][0] >= 0.8:
    #    print (predict_list.filenames[i],' ,',result [0][0],' ,binh thuong cao')
    # elif result [0][0] <= 0.2:
    #    print (predict_list.filenames[i],' ,',result [0][0],' ,lao phoi cao')
    # elif (result [0][0] < 0.8) and (result [0][0] >0.5):
    #    print (predict_list.filenames[i],' ,',result [0][0],' ,co the binh thuong')
    # elif (result [0][0] > 0.2) and (result [0][0] <= 0.5):
    #    print (predict_list.filenames[i],' ,',result [0][0],' ,co the lao phoi')
    #gayyy
cv2.waitKey(0)
cv2.destroyAllWindows()
