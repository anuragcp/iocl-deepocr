# CRNN (CNN+RNN) 

**Initial codebase is taken from** https://github.com/qjadud1994/CRNN-Keras.

**Part of the code documentation is also taken from the above repo unless otherwise stated.**

**CRNN** is a network that combines CNN and RNN to process images containing sequence information such as letters.

https://arxiv.org/pdf/1507.05717.pdf

It is mainly used for OCR technology and has the following advantages.
1. End-to-end learning is possible.
2. Sequence data of arbitrary length can be processed because of LSTM which is free in size of input and output sequence.
3. There is no need for a detector or cropping technique to find each character one by one.


Slightly modified version of the original CRNN model.
(Input size : 700x32)

## Network

![](photo/Network.jpg)

### Convolutional Layer
Extracts features through CNN Layer (VGGNet, ResNet ...).

### Recurrent Layer
Splits the features into a certain size and inserts them into the input of the Bidirectional LSTM.

### Transcription Layer

Conversion of Feature-specific predictions to Label using CTC (Connectionist Temporal Classification).

- - -

### How to Train

* First, you need a lot of cropped/synthetic image data of words. <br/>
* In my case, close to 2 million words were synthetically generated using available malayalam fonts.<br />
* Files inside test/ and train/ folders are named according to an index, like "0.jpg","1.jpg" ..., where each filename can be parsed and traced to the appropriate image file using "data.csv" file as reference. <br />


![](DB/test/0.jpg)
<br/>
Filename: "0.jpg" <br />
From "data.csv", 0-സിദ്ധാന്തത്തിൻെറ<br/>
True: സിദ്ധാന്തത്തിൻെറ <br/>
Predicted: സിദ്ധാന്തത്തിൻെറ <br/>

For training run, `python training.py`.

## File Description

os : Ubuntu 16.04.4 LTS

Python : 3.5.2

Tensorflow : 1.5.0

Keras : 2.1.3


|       File         |Description                                       |
|--------------------|--------------------------------------------------|
|Model .py           |Network using CNN (VGG) + Bidirectional LSTM      |
|Image_Generator. py |Image batch generator for training                |
|parameter. py       |Parameters used in CRNN                           |
|training. py        |CRNN training                                     |
|Prediction. py      |CRNN prediction                                   |
"# iocl-deepocr" 
