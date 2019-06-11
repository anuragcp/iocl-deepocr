from Model import get_Model
import sys
import itertools
import cv2
import os
import numpy as np
from parameter import *


ASCII_VECTOR = '-+=!@#$%^&*(){}[]|\'"\\/?<>;:0123456789'

CHAR_VECTOR = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

model = get_Model(training=False)
#model.load_weights(os.getcwd()+"/"+sys.argv[1])
model.load_weights("LSTM+BN5--05--20.957.hdf5")
print("Model loaded...")

def real(img,j,h,w,g): #input grayscale image specifying height, width, file to be written
		#try:
		img = img.astype(np.float32)       # convert to float 32
		img = (img / 255.0) * 2.0 - 1.0    # normalising
		img_pred = img.T                   # (h,w) -> (w,h)
		img_pred = np.expand_dims(img_pred, axis=-1)  # (w,h,1)
		img_pred = np.expand_dims(img_pred, axis=0)   # (1,w,h,1)
		print("image_pred shape = ", img_pred.shape)
		X_data = np.ones([img_w, img_h, 1])           # (w,h,1)
		print("X_data shape = ", X_data.shape)
		X_data[:img_pred.shape[1]] = img_pred
		img_pred = np.expand_dims(X_data, axis=0)     # (1,w,h,1)
		out = model.predict(img_pred)                 # prediction
		letters = [letter for letter in CHAR_VECTOR]
		out_best = list(np.argmax(out[0, 2:], axis=1)) #ignore the first two outputs, and take the index of the maximum 
		out_best = [k for k, g in itertools.groupby(out_best)]
		print(out_best)
		outstr = ''
		for i in out_best:
			if i < len(letters):
				outstr += letters[i]           # conjoin each charto form the required string
		g.write(str("var")+ " = "+outstr+'-'+str(j)+"-"+str(h)+"-"+str(w)+'\n')
		print(str("var") +" = "+outstr)
		#except as e:
		#print(e)
		#outstr = 'error on prediction'
		return outstr

image_dir = os.listdir("./lisence_plate/")

img = cv2.imread("./lisence_plate/69.jpg", 0)
height = 32
aspect_ratio = (img.shape[1]/img.shape[0])
height =32
width = int(aspect_ratio * height)
print("width and height = {},{}".format(width, height))
img = cv2.resize(img,(width,height))
print(img.shape)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("input image shape = ",img.shape)
with open("result.csv", "w+") as g:
	real(img, 0, img.shape[0],img.shape[1], g)

"""for file in image_dir:
	img = cv2.imread(file)
	print("picture loaded")
	cv2.imshow("img", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print("Image shape {}".format(img.shape))
	with open("result.csv", "w+") as g:
		real(img, 0, img.shape[0],img.shape[1], g)
"""

