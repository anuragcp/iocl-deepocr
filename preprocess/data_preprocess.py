import os
import cv2 as cv

IMG_PATH = "./Image/"	#path to image directory whis is to be parsed
ANN_PATH = "./groundtruth_localization/" 	#path to annotation firectory
TAR_PATH = "./train/"	#path where do you want to save the croped image
LAB_PATH = "./groundtruth_recognition/"	#path to saved labels

list_files = os.listdir(IMG_PATH)

list_files.remove("Thumbs.db")

xmin = None
xmax = None
ymin = None
ymax = None
label_name = None
data_cache = []

iteration = 0

for file in list_files:
	img = cv.imread(IMG_PATH+file,1)
	#with open(TAR_PATH+"data.csv", 'w+') as data_file:
	with open(ANN_PATH+file.strip(".jpg")+".txt") as loc_file:
		xmin, ymax, xmax, ymin = [int(loc_file.readline().strip("\n")), int(loc_file.readline().strip("\n")), int(loc_file.readline().strip("\n")), int(loc_file.readline().strip("\n"))]
		crop_img = img[xmin:xmax, ymin:ymax]
		cv.imwrite(TAR_PATH+file, crop_img)
		with open(LAB_PATH+file.strip(".jpg")+".txt") as label_file:
			cur_ground_truth = label_file.read().replace('\n','')
			data_cache.append(cur_ground_truth+'\n')
			print("fetched = {}".format(cur_ground_truth))
			#data_file.write(str(iteration)+"-"+cur_ground_truth)
		iteration = iteration+1

with open(TAR_PATH+"data.csv", 'w+') as data_file:
	data_file.write(data_cache)

print("last iteration = {}".format(iteration))