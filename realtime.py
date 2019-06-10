from Model import get_Model
import sys
import itertools
import cv2
import os
import numpy as np
from parameter import *
from Prediction import *
from PIL import Image

"""MAL_VECTOR = '  ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^   ^                                                               $'

ASCII_VECTOR = '-+=!@#$%^&*(){}[]|\'"\\/?<>;:0123456789'

CHAR_VECTOR = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

model = get_Model(training=False)
model.load_weights(os.getcwd()+"/"+sys.argv[1])
"""
PATH = "./lisence_plate/"

image_dir = os.listdir(PATH)

print(image_dir)

for file in image_dir:
	img = cv2.imread(PATH+file,0)
	print(img.shape)
	width, height = img.shape[1], img.shape[0]
	with open("result.csv", "w+") as g:
		print(real(img, 0, width,height, g))
