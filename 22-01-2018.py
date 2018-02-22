import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import cv2
from datetime import datetime

videoName = "D1/20151110175712-16km-D1-NORMAL1-SECONDARY/20151110175712-16km-D1-NORMAL1-SECONDARY.mp4"
dataFolderName = "D1/20151110175712-16km-D1-NORMAL1-SECONDARY"

cap = cv2.VideoCapture(videoName)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

videoDateString = videoName.split('/')[-1][0:14]
dataDateString = dataFolderName.split('/')[-1][0:14]
filename = 'SEMANTIC_ONLINE.txt'
videoDate = datetime.strptime(videoDateString, "%Y%m%d%H%M%S")
dataDate = datetime.strptime(dataDateString, "%Y%m%d%H%M%S")
delayVideoToData = (dataDate - videoDate).total_seconds()
fc = 0

print(delayVideoToData)
semantic = np.loadtxt(filename,usecols=(11,12,13))


model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(frameHeight, frameWidth, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

if(delayVideoToData<=0):	
	max_size = int(min(np.round(frameCount/30),np.shape(semantic)[0]+delayVideoToData))
	ret = True

	while (fc<max_size and ret):
		buf = np.empty((30, frameHeight, frameWidth, 3), np.dtype('uint8'))
		while(fc1<30):		
			ret, buf[fc1] = cap.read()
			fc1 += 1
		train_data = buf
		train_label = semantic[delayVideoToData+fc:delayVideoToData+fc+1,:]
		model.train_on_batch(train_data, train_label, sample_weight=None, class_weigth=None)
		fc += 1 
	
elif(delayVideoToData>0):	
	max_size = int(min(np.round(frameCount/30)-delayVideoToData,np.shape(semantic)[0]))
	
	ret = True

	while (fc<(max_size) and ret):
		buf = np.empty((30, frameHeight, frameWidth, 3), np.dtype('uint8'))
		while(fc1<30):
			ret, buf[fc1] = cap.read()
			fc1 += 1
		train_data = buf
		train_label = semantic[fc:(fc+1),:]
		model.train_on_batch(train_data, train_label, sample_weight=None, class_weigth=None)
		fc += 1
	
cap.release()
#score = model.evaluate(x_test, y_test, batch_size=32)
