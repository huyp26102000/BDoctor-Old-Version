
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os


# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", default="D:/keras-multi-label/covid_1.model",
# 	help="path to trained model model")
# ap.add_argument("-l", "--labelbin", default="D:/keras-multi-label/covid_1.pkl",
# 	help="path to label binarizer")
# ap.add_argument("-i", "--image", default="normal3.jpeg",
# 	help="path to input image")
# args = vars(ap.parse_args())
def classify(img):
# load the image
	image = cv2.imread(img)
	output = imutils.resize(image, width=400)
 

	image = cv2.resize(image, (96,96))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)


	print("[INFO] loading network...")
	model = load_model("covid_1.model")
	mlb = pickle.loads(open("covid_1.pkl", "rb").read())


	print("[INFO] classifying image...")
	proba = model.predict(image)[0]
	idxs = np.argsort(proba)[::-1][:2]


	for (i, j) in enumerate(idxs):

		label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
		cv2.putText(output, label, (10, (i * 30) + 25), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


	for (label, p) in zip(mlb.classes_, proba):
		print("{}: {:.2f}%".format(label, p * 100))

# show the output image
	cv2.imshow("kết quả", output)
	cv2.waitKey(0)

classify("normal.jpeg")
