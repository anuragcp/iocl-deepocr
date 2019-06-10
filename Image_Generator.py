import cv2
import os, random
import numpy as np
from parameter import letters,max_text_len
import os.path


## Input Label to Text generator
def labels_to_text(labels):   #generated labels is converted to text taking info from CHAR_VECTOR 
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):     #label text is converted to index value taking info from CHAR_VECTOR 
    return list(map(lambda x: letters.index(x), text))


class TextImageGenerator:
    def __init__(self, img_dirpath, img_w, img_h,
                 batch_size, downsample_factor, num,max_text_len=max_text_len):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor      
        self.img_dirpath = img_dirpath                  # image dir path
        self.img_dir = os.listdir(self.img_dirpath)     # images list
        self.n = num                                    # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []


    def build_data(self,filename):                      # loading the entire image data into RAM, this need optimization
        print(self.n, " Image Loading start...")
        f = open('DB/'+filename,'r')
        read = f.read()
        itr = read.split('\n')
        j=0
        for i, line in enumerate(itr):
            if line != '':
                    img_file,text = line.split("-")
                    if os.path.isfile(self.img_dirpath + img_file+'.jpg'):
                            img = cv2.imread(self.img_dirpath + img_file+'.jpg', cv2.IMREAD_GRAYSCALE)
                            print("image shape {}".format(img.shape))
                            ar = img.shape[0]/img.shape[1]
                            img = cv2.resize(img, (int(self.img_h/ar), self.img_h))
                            img = img.astype(np.float32)
                            img = (img / 255.0) * 2.0 - 1.0            # normalizing the image to (-1-0-1) range
                            print(img.shape)
                            if img.shape[1] <= self.img_w and len(text) <= self.max_text_len:
                                print([len(self.texts),j])
                                self.imgs[j, :, :img.shape[1]] = img   # stores imgs
                                print(text)  
                                self.texts.append(text)                # stores texts
                                j=j+1
                                if len(self.texts) == self.n:
                                    break                              # breaks after the specified total data need to trained.
        print(self.texts)
        print(len(self.texts))
        print(len(self.imgs))
        print(self.n)
        print(len(self.texts) == len(self.imgs))
        print(len(self.texts) == self.n)
        print(self.n, " Image Loading finish...")

    def next_sample(self):      # send one sample, increment the index to select next data 
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
            print(self.indexes[self.cur_index])
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):       # next batch generator.
        while True:
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])     # (batchsize(bs), 800, 32, 1)
            Y_data = np.ones([self.batch_size, self.max_text_len])             # (bs, 60)
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)  # (bs, 1) RNN input length
            label_length = np.zeros((self.batch_size, 1))           # (bs, 1)  # RNN output true label

            for i in range(self.batch_size):
                img, text = self.next_sample()     # get each sample (h,w)
                img = img.T                        # transpose (w,h)
                img = np.expand_dims(img, -1)      # expand dimensions (w,h,1)
                X_data[i] = img                    # (i,w,h,1)
                Y_data[i,:len(text_to_labels(text))] = text_to_labels(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,               # (bs, 800, 32, 1)
                'the_labels': Y_data,              # (bs, 60)
                'input_length': input_length,      # (bs, 1)
                'label_length': label_length       # (bs, 1)
            }
            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1)
            yield (inputs, outputs)
