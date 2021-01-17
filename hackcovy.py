import os
import sys
import datetime
import random
import torch
import cv2
import time
import argparse
import imutils
import pickle
import shutil
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import tensorflow as tf
import importlib.util
from glob import glob
from tkinter import *
from tkinter.ttk import *
from PIL import ImageTk, Image
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from tkcalendar import Calendar, DateEntry
from keras.preprocessing.image import img_to_array
from keras.models import load_model
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

model = load_model("covid_new.model")
mlb = pickle.loads(open("covid_1.pkl", "rb").read())
vectorFileDir = []
vectorFileName = []
vectorCheck = []
dateAxis = np.array([])
covidCountStack = np.array([])
normalCountStack = np.array([])
tupleDate = []
tupleName = []
valueAxis = np.row_stack((covidCountStack, normalCountStack))
normalCount = 0
covidCount = 0

def convertHeatMap(heatmapImage):
    try:
        shutil.rmtree("image")
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
    if not os.path.exists("image"):
        os.makedirs("image")
    cv2.imwrite("image/heatmapImage.png", heatmapImage)
    img_width, img_height = 96, 96
    validation_data_dir = "image"
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
    predict_list = np.copy(predict_list / 255.0)
    os.chdir(og_dir)
    model_dir = "covid_1.model"
    model=load_model(model_dir)
    img_data = 0
    for i in range (0, nb_validation_samples):
        img_data=np.expand_dims(predict_list[i], 0)
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
        return cv2.resize(superimposed_img, (250, 350))

def eventImageList(event):
    global vectorFileDir, vectorFileName
    index = imageList.curselection()[0]
    seltext = imageList.get(index)
    fileDir = ""
    for i in range(len(vectorFileName)):
        if (vectorFileName[i] == seltext):
            fileDir = vectorFileDir[i]
    frame = cv2.imread(fileDir)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2image = cv2.resize(cv2image, (250, 350))
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image = img)
    lmain.imgtk = imgtk
    lmain.configure(image = imgtk)

def eventAfterList(event):
    global vectorFileDir, vectorFileName
    index = imageList1.curselection()[0]
    seltext = imageList1.get(index)
    date = str(cal.get_date())
    frame = cv2.imread("data/" + date + "/" + seltext)
    cv2image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2image1 = cv2.resize(cv2image1, (250, 350))
    img1 = Image.fromarray(cv2image1)
    imgtk1 = ImageTk.PhotoImage(image = img1)
    lmain1.imgtk1 = imgtk1
    lmain1.configure(image = imgtk1)

def eventImageList5(event):
    index = imageList5.curselection()[0]
    seltext = imageList5.get(index)
    frame = cv2.imread("data/" + mynumber.get() + "/" + seltext)
    cv2image5 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2image5 = cv2.resize(cv2image5, (250, 350))
    img5 = Image.fromarray(cv2image5)
    imgtk5 = ImageTk.PhotoImage(image = img5)
    lmain5.imgtk = imgtk5
    lmain5.configure(image = imgtk5)

def eventImageList6(event):
    index = imageList6.curselection()[0]
    seltext = imageList6.get(index)
    frame = cv2.imread("data/" + seltext + "/" + mynumber6.get())
    cv2image6 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2image6 = cv2.resize(cv2image6, (250, 350))
    img6 = Image.fromarray(cv2image6)
    imgtk6 = ImageTk.PhotoImage(image = img6)
    lmain6.imgtk = imgtk6
    lmain6.configure(image = imgtk6)

def currectSelection(evt):
    value = str(resultList.get(ACTIVE))

def chosingDate():
    imageList5.delete(0, END)
    for r, d, f in os.walk("data/" + mynumber.get()):
        for file in f:
            imageList5.insert(END, file)

def chosingName():
    imageList6.delete(0, END)
    for r, d, f in os.walk("data"):
        for file in f:
            if (file == mynumber6.get()):
                pos = 0
                for i in range(len(r)):
                    if (r[i] == "\\"):
                        pos = i
                imageList6.insert(END, r[pos + 1 : ])
            
def move_window(event):
    root.geometry("+{0}+{1}".format(event.x_root, event.y_root))
    
def change_on_hovering(event):
    global close_button
    close_button["bg"] = "red"
    
def return_to_normalstate(event):
    global close_button
    close_button["bg"] = "#2e2e2e"

def openImage():
    global vectorFileDir, vectorFileName
    fileDir = filedialog.askopenfilename(initialdir = '/home/', title = "Choose X-ray images", filetypes = [("Images", ".jpg")])
    for i in range(len(fileDir)):
        if (fileDir[i] == '/'): pos = i
    fileName = fileDir[pos + 1:]
    vectorFileDir.append(fileDir)
    vectorFileName.append(fileName)
    vectorCheck.append(True)
    imageList.insert(END, fileName)
    frame = cv2.imread(fileDir)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2image = cv2.resize(cv2image, (250, 350))
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image = img)
    lmain.imgtk = imgtk
    lmain.configure(image = imgtk)

def classify():
    global vectorFileDir, vectorFileName, cal, normalCount, covidCount, dateAxis, valueAxis, covidCountStack, normalCountStack, valueAxis, tupleDate, tupleName
    date = str(cal.get_date())
    if not os.path.exists("data/" + date):
        os.makedirs("data/" + date)
    covidCount = 0
    normalCount = 0
    for index in range(len(vectorFileDir)):
        if (vectorCheck[index] == True):
            tupleName.append(vectorFileName[index])
            vectorCheck[index] = False
            img = vectorFileDir[index]
            image = cv2.imread(img)
            output = imutils.resize(image, width = 400)
            image = cv2.resize(image, (256, 256))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis = 0)
            proba = model.predict(image)[0]
            idxs = np.argsort(proba)[::-1][:2]
            maxx = 0
            maxName = ""
            output = convertHeatMap(output)
            for (i, j) in enumerate(idxs):
                label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
                if (maxx < proba[j] * 100):
                    maxx = proba[j] * 100
                    maxName = mlb.classes_[j]
                cv2.putText(output, label, (10, (i * 30) + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if (maxName == "covid"): covidCount += 1
            else: normalCount += 1
            cv2.imwrite("data/" + date + "/" + vectorFileName[index], output)
            imageList1.insert(END, vectorFileName[index])
            cv2image1 = cv2.imread("data/" + date + "/" + vectorFileName[index])
            cv2image1 = cv2.cvtColor(cv2image1, cv2.COLOR_BGR2RGBA)
            cv2image1 = cv2.resize(cv2image1, (250, 350))
            img1 = Image.fromarray(cv2image1)
            imgtk1 = ImageTk.PhotoImage(image = img1)
            lmain1.imgtk1 = imgtk1
            lmain1.configure(image = imgtk1)
            progress["value"] += 100 / len(vectorFileDir)
    combobox6['values'] = tuple(tupleName)
    tupleDate.append(date)
    combobox['values'] = tuple(tupleDate)
    if (date in dateAxis):
        for i in range(len(dateAxis)):
            if (dateAxis[i] == date):
                covidCountStack[i] = covidCount
                normalCountStack[i] = normalCount
    else:
        dateAxis = np.append(dateAxis, date)
        covidCountStack = np.append(covidCountStack, covidCount)
        normalCountStack = np.append(normalCountStack, normalCount)
    valueAxis = np.row_stack((covidCountStack, normalCountStack))
    maxAxis = max(np.amax(covidCountStack), np.amax(normalCountStack))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(dateAxis, valueAxis[0, :], label = "Covid positive", color = 'r', marker = 'o')
    ax1.plot(dateAxis, valueAxis[1, :], label = "Covid negative", color = 'g', marker = 'o')
    axes = plt.axes()
    if (len(covidCountStack) == 0):
        maxAxis = 5
    else: maxAxis = max(5, max(np.amax(covidCountStack), np.amax(normalCountStack)))
    axes.set_ylim([0, maxAxis])
    axesNumber = []
    for i in range(int(maxAxis) + 1): axesNumber.append(i)
    axes.set_yticks(axesNumber)
    plt.xticks(dateAxis)
    plt.xlabel("Graph visualize relation between positive cases and negative cases")
    handles, labels = ax1.get_legend_handles_labels()
    lgd = ax1.legend(loc = "best")
    ax1.grid('on')
    plt.savefig("bieudo.png")
    accuracyText.set(str(covidCount) + " / " + str(covidCount + normalCount))
    cv2image4 = cv2.imread("bieudo.png")
    cv2image4 = cv2.cvtColor(cv2image4, cv2.COLOR_BGR2RGBA)
    cv2image4 = cv2.resize(cv2image4, (700, 500))
    img4 = Image.fromarray(cv2image4)
    imgtk4 = ImageTk.PhotoImage(image = img4)
    lmain4.imgtk4 = imgtk4
    lmain4.configure(image = imgtk4)
    progress["value"] = 100

root = Tk()
root.overrideredirect(True)
root.title("Covid-19 and pneumonia diagnosis tool via x-ray images") 
root.geometry('1040x700')
root.resizable(0, 0)
title_bar = tk.Frame(root, bg = "#2e2e2e", relief = "raised", bd = 2, height = 30, highlightthickness = 0)
title_bar.pack(side = TOP, fill = "both")
close_button = tk.Button(title_bar, text = 'X', command = root.destroy, bg = "#2e2e2e", padx = 2, pady = 2, activebackground = "red", bd = 0, font = "bold", fg = "white", highlightthickness = 0)
close_button.pack(side = RIGHT)
title = Label(title_bar, text = " ")
title.config(font = ("Courier", 12), background = "#2e2e2e", foreground = "#ffffff")
title.pack(side = LEFT)
app3 = Frame(title_bar)
app3.pack(side = LEFT)
lmain3 = Label(app3)
lmain3.grid(padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = E + S)
cv2image3 = cv2.imread("icon.png")
cv2image3 = cv2.cvtColor(cv2image3, cv2.COLOR_BGR2RGBA)
cv2image3 = cv2.resize(cv2image3, (15, 15))
img3 = Image.fromarray(cv2image3)
imgtk3 = ImageTk.PhotoImage(image = img3)
lmain3.imgtk3 = imgtk3
lmain3.configure(image = imgtk3)
title = Label(title_bar, text = " Import Keras - Hackcovy")
title.config(font = ("Courier", 12), background = "#2e2e2e", foreground = "#ffffff")
title.pack(side = LEFT)
title_bar.bind("<B1-Motion>", move_window)
close_button.bind("<Enter>", change_on_hovering)
close_button.bind("<Leave>", return_to_normalstate)
app7 = Frame(root)
app7.pack(side = TOP, fill = "both")
lmain7 = Label(app7)
lmain7.grid(padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = E + S)
cv2image7 = cv2.imread("banner.png")
cv2image7 = cv2.cvtColor(cv2image7, cv2.COLOR_BGR2RGBA)
cv2image7 = cv2.resize(cv2image7, (1036, 100))
img7 = Image.fromarray(cv2image7)
imgtk7 = ImageTk.PhotoImage(image = img7)
lmain7.imgtk7 = imgtk7
lmain7.configure(image = imgtk7)
tabParent = ttk.Notebook(root)
tab1 = ttk.Frame(tabParent)
tab2 = ttk.Frame(tabParent)
tab3 = ttk.Frame(tabParent)
tabParent.add(tab1, text = "X-ray images")
tabParent.add(tab2, text = "Statistcal")
tabParent.add(tab3, text = "Analyze history")
tabParent.pack(side = TOP, expand = 1, fill = "both")
nullCol1 = Frame(tab1, height = 20)
nullCol1.grid(column = 1, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
nullRow1 = Frame(tab1, width = 30)
nullRow1.grid(row = 2, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
inputFrame = LabelFrame(tab1, text = "Chest X-ray images")
inputFrame.grid(row = 2, column = 2, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
app = Frame(inputFrame)
app.grid(row = 2, rowspan = 2, column = 2, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
lmain = Label(app)
lmain.grid(padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = E + W + N + S)
cv2image = cv2.imread("untitled.png")
cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
cv2image = cv2.resize(cv2image, (250, 350))
img = Image.fromarray(cv2image)
imgtk = ImageTk.PhotoImage(image = img)
lmain.imgtk = imgtk
lmain.configure(image = imgtk)
imageList = Listbox(inputFrame, width = 5, height = 15, font = ("times", 13))
imageList = tk.Listbox(inputFrame, width = 8, height = 15, font = ("times", 13))
imageList.grid(row = 2, column = 3, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
imageList.bind("<ButtonRelease-1>", eventImageList)
Button(inputFrame, text = "Select", command = openImage).grid(row = 3, column = 3, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
nullCol3 = Frame(tab1, width = 10)
nullCol3.grid(column = 3, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
resultFrame = LabelFrame(tab1, text = "Result")
resultFrame.grid(row = 2, column = 4, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
app1 = Frame(resultFrame)
app1.grid(row = 2, rowspan = 2, column = 2, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
lmain1 = Label(app1)
lmain1.grid(padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = E + W + N + S)
cv2image1 = cv2.imread("untitled.png")
cv2image1 = cv2.cvtColor(cv2image1, cv2.COLOR_BGR2RGBA)
cv2image1 = cv2.resize(cv2image1, (250, 350))
img1 = Image.fromarray(cv2image1)
imgtk1 = ImageTk.PhotoImage(image = img1)
lmain1.imgtk1 = imgtk1
lmain1.configure(image = imgtk1)
imageList1 = Listbox(resultFrame, width = 5, height = 15, font = ('times', 13))
imageList1 = tk.Listbox(resultFrame, width = 8, height = 15, font = ('times', 13))
imageList1.grid(row = 2, rowspan = 2, column = 3, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
imageList1.bind('<ButtonRelease-1>', eventAfterList)
nullCol4 = Frame(tab1, width = 30)
nullCol4.grid(column = 7, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
systemFrame = LabelFrame(tab1, text = "System", width = 150, height = 50) 
systemFrame.grid(row = 2, rowspan = 2, column = 8, padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = E + W + N)
Label(systemFrame, text = " Analyze Date").grid(row = 0, sticky = E + W + N)
cal = DateEntry(systemFrame, width = 12, background = "darkblue", foreground = "white", borderwidth = 2)
cal.grid(row = 1, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N)
Button(systemFrame, text = "Analyze", command = classify).grid(row = 2, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N)
Button(systemFrame, text = "Cancel").grid(row = 3, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N)
Label(systemFrame, text = " Probability have Covid-19").grid(row = 4, sticky = E + W + N)
accuracyText = StringVar()
accuracy = Entry(systemFrame, textvariable = accuracyText)
accuracy.grid(row = 5, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
nullRow2 = Frame(tab1, height = 30)
nullRow2.grid(row = 3, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
app2 = Frame(tab1)
app2.grid(row = 3, rowspan = 2, column = 8, padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = E + S)
lmain2 = Label(app2)
lmain2.grid(padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = E + S)
cv2image2 = cv2.imread("hackcovy2.png")
cv2image2 = cv2.cvtColor(cv2image2, cv2.COLOR_BGR2RGBA)
cv2image2 = cv2.resize(cv2image2, (95, 90))
img2 = Image.fromarray(cv2image2)
imgtk2 = ImageTk.PhotoImage(image = img2)
lmain2.imgtk2 = imgtk2
lmain2.configure(image = imgtk2)
progress = Progressbar(tab1, orient = HORIZONTAL, length = 245, mode = "determinate")
progress.grid(row = 4, column = 2, columnspan = 5, padx = 5, pady = 0, ipadx = 0, ipady = 0, sticky = E + W + S)
progress["value"] = 0
app4 = Frame(tab2)
app4.grid(row = 1, column = 1, padx = (1040 - 700) // 2, pady = 10, ipadx = 0, ipady = 0, sticky = E + S)
lmain4 = Label(app4)
lmain4.grid(padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = E + S)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(dateAxis, valueAxis[0, :], label = "Covid positive", color = 'r', marker = 'o')
ax1.plot(dateAxis, valueAxis[1, :], label = "Covid negative", color = 'g', marker = 'o')
axes = plt.axes()
if (len(covidCountStack) == 0):
    maxAxis = 5
else: maxAxis = max(5, max(np.amax(covidCountStack), np.amax(normalCountStack)))
axes.set_ylim([0, maxAxis])
axesNumber = []
for i in range(int(maxAxis) + 1): axesNumber.append(i)
axes.set_yticks(axesNumber)
plt.xticks(dateAxis)
plt.xlabel("Graph visualize relation between positive cases and negative cases")
handles, labels = ax1.get_legend_handles_labels()
lgd = ax1.legend(loc = "best")
ax1.grid('on')
plt.savefig("bieudo.png")
cv2image4 = cv2.imread("bieudo.png")
cv2image4 = cv2.cvtColor(cv2image4, cv2.COLOR_BGR2RGBA)
cv2image4 = cv2.resize(cv2image4, (700, 500))
img4 = Image.fromarray(cv2image4)
imgtk4 = ImageTk.PhotoImage(image = img4)
lmain4.imgtk4 = imgtk4
lmain4.configure(image = imgtk4)
nullCol1Tab3 = Frame(tab3, width = 90)
nullCol1Tab3.grid(column = 1, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
nullRow1Tab3 = Frame(tab3, height = 30)
nullRow1Tab3.grid(row = 1, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
filterDateFrame = LabelFrame(tab3, text = "Filter by day")
filterDateFrame.grid(row = 2, column = 2, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
app5 = Frame(filterDateFrame)
app5.grid(row = 1, column = 1, columnspan = 2, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
lmain5 = Label(app5)
lmain5.grid(padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = E + W + N + S)
cv2image5 = cv2.imread("untitled.png")
cv2image5 = cv2.cvtColor(cv2image5, cv2.COLOR_BGR2RGBA)
cv2image5 = cv2.resize(cv2image5, (250, 350))
img5 = Image.fromarray(cv2image5)
imgtk5 = ImageTk.PhotoImage(image = img5)
lmain5.imgtk = imgtk5
lmain5.configure(image = imgtk5)
imageList5 = Listbox(filterDateFrame, width = 5, height = 15, font = ('times', 13))
imageList5 = tk.Listbox(filterDateFrame, width = 8, height = 15, font = ('times', 13))
imageList5.grid(row = 1, column = 3, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
imageList5.bind("<ButtonRelease-1>", eventImageList5)
mynumber = tk.StringVar()
combobox = ttk.Combobox(filterDateFrame, textvariable = mynumber)
combobox["values"] = tuple(tupleDate)
combobox.grid(row = 2, column = 1, columnspan = 2, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
button = ttk.Button(filterDateFrame, text = "Day selection", command = chosingDate)
button.grid(row = 2, column = 3, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
nullCol2Tab3 = Frame(tab3, width = 70)
nullCol2Tab3.grid(column = 3, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
filterNameFrame = LabelFrame(tab3, text = "Patients filter")
filterNameFrame.grid(row = 2, column = 4, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
app6 = Frame(filterNameFrame)
app6.grid(row = 1, column = 1, columnspan = 2, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
lmain6 = Label(app6)
lmain6.grid(padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = E + W + N + S)
cv2image6 = cv2.imread("untitled.png")
cv2image6 = cv2.cvtColor(cv2image6, cv2.COLOR_BGR2RGBA)
cv2image6 = cv2.resize(cv2image6, (250, 350))
img6 = Image.fromarray(cv2image6)
imgtk6 = ImageTk.PhotoImage(image = img6)
lmain6.imgtk = imgtk6
lmain6.configure(image = imgtk6)
imageList6 = Listbox(filterNameFrame, width = 5, height = 15, font = ('times', 13))
imageList6 = tk.Listbox(filterNameFrame, width = 8, height = 15, font = ('times', 13))
imageList6.grid(row = 1, column = 3, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
imageList6.bind("<ButtonRelease-1>", eventImageList6)
mynumber6 = tk.StringVar()
combobox6 = ttk.Combobox(filterNameFrame, textvariable = mynumber6)
combobox6["values"] = tuple(tupleName)
combobox6.grid(row = 2, column = 1, columnspan = 2, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
button6 = ttk.Button(filterNameFrame, text = "Name selection", command = chosingName)
button6.grid(row = 2, column = 3, padx = 5, pady = 5, ipadx = 0, ipady = 0, sticky = E + W + N + S)
root.mainloop()
