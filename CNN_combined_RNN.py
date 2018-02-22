from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, GRU, Dropout
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2
from datetime import datetime
from skimage.transform import resize
import random

videoName = ["20151110175712-16km-D1-NORMAL1-SECONDARY.mp4","20151110180846-16km-D1-NORMAL2-SECONDARY.mp4","20151111123123-25km-D1-NORMAL-MOTORWAY.mp4","20151111125204-24km-D1-AGGRESSIVE-MOTORWAY.mp4","20151111132343-25km-D1-DROWSY-MOTORWAY.mp4","20151111134542-16km-D1-AGGRESSIVE-SECONDARY..mp4","20151111135605-13km-D1-DROWSY-SECONDARY.mp4",
			"20151120131704-26km-D2-NORMAL-MOTORWAY.mp4","20151120133457-26km-D2-AGGRESSIVE-MOTORWAY.mp4","20151120135150-25km-D2-DROWSY-MOTORWAY.mp4","20151120160903-16km-D2-NORMAL1-SECONDARY.mp4","20151120162104-17km-D2-NORMAL2-SECONDARY.mp4","20151120163335-16km-D2-AGGRESSIVE-SECONDARY.mp4","20151120165604-16km-D2-DROWSY-SECONDARY.mp4",
			"20151126110501-26km-D3-NORMAL-MOTORWAY.mp4","20151126113753-26km-D3-DROWSY-MOTORWAY.mp4","20151126124207-16km-D3-NORMAL1-SECONDARY.mp4","20151126125500-16km-D3-NORMAL2-SECONDARY.mp4","20151126130708-16km-D3-AGGRESSIVE-SECONDARY.mp4","20151126132012-16km-D3-DROWSY-SECONDARY.mp4","20151126134731-26km-D3-AGGRESSIVE-MOTORWAY.mp4",
			"20151203171759-16km-D4-NORMAL1-SECONDARY.mp4","20151203173100-17km-D4-NORMAL2-SECONDARY.mp4","20151203174323-16km-D4-AGGRESSIVE-SECONDARY.mp4","20151203175659-17km-D4-DROWSY-SECONDARY.mp4","20151204152839-25km-D4-NORMAL-MOTORWAY.mp4","20151204154908-25km-D4-AGGRESSIVE-MOTORWAY.mp4","20151204160822-25km-D4-DROWSY-MOTORWAY.mp4",
			"20151209151237-25km-D5-NORMAL-MOTORWAY.mp4","20151209153136-25km-D5-AGGRESSIVE-MOTORWAY.mp4","20151211160230-25km-D5-DROWSY-MOTORWAY.mp4","20151211162829-16km-D5-NORMAL1-SECONDARY.mp4","220151211164123-17km-D5-NORMAL2-SECONDARY.mp4","20151211165349-12km-D5-AGGRESSIVE-SECONDARY.mp4","20151211170500-16km-D5-DROWSY-SECONDARY.mp4",
			"20151217162706-26km-D6-NORMAL-MOTORWAY.mp4","20151217164653-25km-D6-DROWSY-MOTORWAY.mp4","20151221112444-D6-NORMAL-SECONDARY.mp4","20151221113845-16km-D6-DROWSY-SECONDARY.mp4","20151221120048-D6-AGGRESSIVE-MOTORWAY.mp4"]

dataFolderName = ["D1/20151110175712-16km-D1-NORMAL1-SECONDARY/","D1/20151110180824-16km-D1-NORMAL2-SECONDARY/","D1/20151111123124-25km-D1-NORMAL-MOTORWAY/","D1/20151111125233-24km-D1-AGGRESSIVE-MOTORWAY/","D1/20151111132348-25km-D1-DROWSY-MOTORWAY/","D1/20151111134545-16km-D1-AGGRESSIVE-SECONDARY/","D1/20151111135612-13km-D1-DROWSY-SECONDARY/",
				"D2/20151120131714-26km-D2-NORMAL-MOTORWAY/","D2/20151120133502-26km-D2-AGGRESSIVE-MOTORWAY/","D2/20151120135152-25km-D2-DROWSY-MOTORWAY/","D2/20151120160904-16km-D2-NORMAL1-SECONDARY/","D2/20151120162105-17km-D2-NORMAL2-SECONDARY/","D2/20151120163350-16km-D2-AGGRESSIVE-SECONDARY/","D2/20151120164606-16km-D2-DROWSY-SECONDARY/",
				"D3/20151126110502-26km-D3-NORMAL-MOTORWAY/","D3/20151126113754-26km-D3-DROWSY-MOTORWAY/","D3/20151126124208-16km-D3-NORMAL1-SECONDARY/","D3/20151126125458-16km-D3-NORMAL2-SECONDARY/","D3/20151126130707-16km-D3-AGGRESSIVE-SECONDARY/","D3/20151126132013-17km-D3-DROWSY-SECONDARY/","D3/20151126134736-26km-D3-AGGRESSIVE-MOTORWAY/",
				"D4/20151203171800-16km-D4-NORMAL1-SECONDARY/","D4/20151203173103-17km-D4-NORMAL2-SECONDARY/","D4/20151203174324-16km-D4-AGGRESSIVE-SECONDARY/","D4/20151203175637-17km-D4-DROWSY-SECONDARY/","D4/20151204152848-25km-D4-NORMAL-MOTORWAY/","D4/20151204154908-25km-D4-AGGRESSIVE-MOTORWAY/","D4/20151204160823-25km-D4-DROWSY-MOTORWAY/",
				"D5/20151209151242-25km-D5-NORMAL-MOTORWAY/","D5/20151209153137-25km-D5-AGGRESSIVE-MOTORWAY/","D5/20151211160213-25km-D5-DROWSY-MOTORWAY/","D5/20151211162829-16km-D5-NORMAL1-SECONDARY/","D5/20151211164124-17km-D5-NORMAL2-SECONDARY/","D5/20151211165606-12km-D5-AGGRESSIVE-SECONDARY/","D5/20151211170502-16km-D5-DROWSY-SECONDARY/",
				"D6/20151217162714-26km-D6-NORMAL-MOTORWAY/","D6/20151217164730-25km-D6-DROWSY-MOTORWAY/","D6/20151221112434-17km-D6-NORMAL-SECONDARY/","D6/20151221113846-16km-D6-DROWSY-SECONDARY/","D6/20151221120051-26km-D6-AGGRESSIVE-MOTORWAY/"]

timesteps = 1800
data_dim = 1000

model1 = ResNet50(weights='imagenet')

model = Sequential()
model.add(LSTM(32, dropout = 0.5, recurrent_dropout = 0.5, input_shape=(timesteps,data_dim)))  # returns a sequence of vectors of dimension 32
model.add(Dense(3))  # returns a sequence of vectors of dimension 3

model2 = Sequential()
model2.add(GRU(128, dropout = 0.5, recurrent_dropout = 0.2, return_sequences=True, input_shape=(timesteps,data_dim)))  # returns a sequence of vectors of dimension 32
model2.add(GRU(64, dropout = 0.5, recurrent_dropout = 0.2, return_sequences=True))  # returns a sequence of vectors of dimension 3
model2.add(GRU(3, dropout = 0.5, recurrent_dropout = 0.2))

model3 = Sequential()
model3.add(Merge([model, model2], mode= 'concat'))
model3.add(Dense(3, activation = 'relu'))

model3.compile(loss='mse', optimizer='Adam', metrics=['mae', 'accuracy'])


for i in range(39):
	videoName1 = dataFolderName[i]+videoName[i]
	cap = cv2.VideoCapture(videoName1)
	frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	videoDateString = videoName1.split('/')[-1][0:14]
	dataDateString = dataFolderName[i].split('/')[-2][0:14]
	filename = 'SEMANTIC_ONLINE.txt'
	videoDate = datetime.strptime(videoDateString, "%Y%m%d%H%M%S")
	dataDate = datetime.strptime(dataDateString, "%Y%m%d%H%M%S")
	delayVideoToData = (dataDate - videoDate).total_seconds()
	fc = 0
	print("Training Video no",i,":")
	print(delayVideoToData)
	semantic = np.loadtxt(dataFolderName[i]+filename,usecols=(11,12,13))

	if(delayVideoToData<=0):	
		max_size = int(min(np.round(frameCount/30),np.shape(semantic)[0]+delayVideoToData))
		ret = True
		RNN_train_data_current = []
		print(max_size)

		while (fc<max_size and ret):
			buf = np.empty((30, frameHeight, frameWidth, 3), np.dtype('uint8'))
			fc1 = 0
			print(fc)
			while(fc1<30):		
				ret, buf[fc1] = cap.read()
				x = image.img_to_array(np.round(resize(buf[fc1],(224,224,3))))	
				x = np.expand_dims(x, axis=0)
				x = preprocess_input(x)
				features = model1.predict(x)
				l = np.shape(features)
				#print(l)
				RNN_train_data_current = np.append(RNN_train_data_current,features)
				fc1 += 1
			fc += 1 
		
		RNN_train_data = []
		RNN_train_label = []

		x_train_buf = np.loadtxt(train_data[i],usecols=(4,5,6,7,8,9,10))
		x_train = np.zeros((np.shape(x_train_buf)[0]-59-int(delayVideoToData),60,np.shape(x_train_buf)[1]))
		for k in range(np.shape(x_train_buf)[0]-59-int(delayVideoToData)):
			p = k+int(delayVideoToData)
			x_train[k] = x_train_buf[p:p+60]

		x_train = x_train/100
		
		print("Resizing of Data",i)
		for r in range(max_size-59):
			print(r)
			fc = r+60
			RNN_train_data = np.append(RNN_train_data,np.array(RNN_train_data_current[l[1]*(fc*30-1800):(l[1]*((fc+1)*30))]))
			RNN_train_label = np.append(RNN_train_label,semantic[fc+int(delayVideoToData),:])

		RNN_train_data = np.reshape(RNN_train_data,(int(len(RNN_train_data)/(l[1]*1800)),1800,l[1]))
		RNN_train_data = RNN_train_data/255	
		RNN_train_label = np.reshape(RNN_train_label,(int(len(RNN_train_label)/3),3))
				
		#model.fit(RNN_train_data,np.reshape(semantic[fc+int(delayVideoToData),:],(1,3)),batch_size=1,epochs=1)
		

	elif(delayVideoToData>0):	
		max_size = int(min(np.round(frameCount/30)-delayVideoToData,np.shape(semantic)[0]))

		ret = True
		t = 0
		while(t<delayVideoToData*30):
			discard = cap.read()
			t += 1
		RNN_train_data_current = []
		print(max_size)

		while (fc<(max_size) and ret):
			buf = np.empty((30, frameHeight, frameWidth, 3), np.dtype('uint8'))
			fc1 = 0
			while(fc1<30):
				ret, buf[fc1] = cap.read()
				x = image.img_to_array(np.round(resize(buf[fc1],(224,224,3))))	
				x = np.expand_dims(x, axis=0)
				x = preprocess_input(x)
				features = model1.predict(x)
				l = np.shape(features)
				#print(l)
				RNN_train_data_current = np.append(RNN_train_data_current,features)
				fc1 += 1
			fc += 1				
			
		RNN_train_data = []
		RNN_train_label = []

		x_train_buf = np.loadtxt(train_data[i],usecols=(4,5,6,7,8,9,10))
		x_train = np.zeros((np.shape(x_train_buf)[0]-59),60,np.shape(x_train_buf)[1])
		for k in range(np.shape(x_train_buf)[0]-59):
			x_train[k] = x_train_buf[k:k+60]

		x_train = x_train/100
		print("Resizing of Data",i)

		for r in range(max_size-59):
			fc = r+60
			RNN_train_data = np.append(RNN_train_data,np.array(RNN_train_data_current[l[1]*(fc*30-1800):(l[1]*((fc+1)*30))]))
			RNN_train_label = np.append(RNN_train_label,semantic[fc,:])

		RNN_train_data = np.reshape(RNN_train_data,(int(len(RNN_train_data)/(l[1]*1800)),1800,l[1]))
		RNN_train_data = RNN_train_data/255	
		RNN_train_label = np.reshape(RNN_train_label,(int(len(RNN_train_label)/3),3))

	cap.release()		
	
	print("Creating final features and labels")
	if (i==0):
		RCNN_train_data = RNN_train_data
		RCNN_train_label = RNN_train_label
		RNN_data = x_train
	else:
		RCNN_train_data = np.append(RCNN_train_data,RNN_train_data,axis=2)
		RCNN_train_label = np.append(RCNN_train_label,RNN_train_label,axis=0)
		RNN_data = np.append(RNN_data,x_train,axis=2)

			
model.fit([RCNN_train_data, RNN_data],RCNN_train_label,batch_size=10,epochs=1)
				
				
print("Evaluation started")
model.save("save_with_dropout.h5")

#model = load_model("save_with_dropout.h5")
videoName1 = dataFolderName[39]+videoName[39]
cap = cv2.VideoCapture(videoName1)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

videoDateString = videoName1.split('/')[-1][0:14]
dataDateString = dataFolderName[39].split('/')[-2][0:14]
filename = 'SEMANTIC_ONLINE.txt'
videoDate = datetime.strptime(videoDateString, "%Y%m%d%H%M%S")
dataDate = datetime.strptime(dataDateString, "%Y%m%d%H%M%S")
delayVideoToData = (dataDate - videoDate).total_seconds()
fc = 0

print(delayVideoToData)
semantic = np.loadtxt(dataFolderName[39]+filename,usecols=(11,12,13))

if(delayVideoToData<=0):	
	max_size = int(min(np.round(frameCount/30),np.shape(semantic)[0]+delayVideoToData))
	ret = True
	RNN_test_data_current = []
	print(max_size)
	while (fc<max_size and ret):
		fc1 = 0
		buf = np.empty((30, frameHeight, frameWidth, 3), np.dtype('uint8'))
		print(fc)
		while(fc1<30):		
			ret, buf[fc1] = cap.read()
			x = image.img_to_array(np.round(resize(buf[fc1],(224,224,3))))	
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			features = model1.predict(x)
			l = np.shape(features)
			RNN_test_data_current = np.append(RNN_test_data_current,features)
			fc1 += 1

		fc += 1 

	RNN_test_data = []
	RNN_test_label = []

	x_test_buf = np.loadtxt(train_data[39],usecols=(4,5,6,7,8,9,10))
	x_test = np.zeros((np.shape(x_test_buf)[0]-59-int(delayVideoToData),60,np.shape(x_test_buf)[1]))
	for k in range(np.shape(x_test_buf)[0]-59-int(delayVideoToData)):
		p = k+int(delayVideoToData)
		x_test[k] = x_test_buf[p:p+60]

	x_test = x_test/100
	print("Resizing of Data")

	for r in range(max_size-59):
		fc = r+60
		RNN_test_data = np.append(RNN_test_data,np.array(RNN_test_data_current[l[1]*(fc*30-1800):(l[1]*((fc+1)*30))]))
		RNN_test_label = np.append(RNN_test_label,semantic[fc+int(delayVideoToData),:])

	RNN_test_data = np.reshape(RNN_test_data,(int(len(RNN_test_data)/(l[1]*1800)),1800,l[1]))
	RNN_test_data = RNN_test_data/255	
	RNN_test_label = np.reshape(RNN_test_label,(int(len(RNN_train_label)/3),3))	

	
	
elif(delayVideoToData>0):	
	max_size = int(min(np.round(frameCount/30)-delayVideoToData,np.shape(semantic)[0]))

	print(max_size)

	ret = True
	t = 0
	while(t<delayVideoToData*30):
		discard = cap.read()
		t += 1
	print(t)
	RNN_test_data_current = []
	while (fc<(max_size) and ret):
		buf = np.empty((30, frameHeight, frameWidth, 3), np.dtype('uint8'))
		fc1= 0
		print(fc)
		while(fc1<30):
			ret, buf[fc1] = cap.read()
			x = image.img_to_array(np.round(resize(buf[fc1],(224,224,3))))	
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			features = model1.predict(x)
			l = np.shape(features)
			RNN_test_data_current = np.append(RNN_test_data_current,features)
			fc1 += 1
		fc += 1
		
	RNN_test_data = []
	RNN_test_label = []

	x_test_buf = np.loadtxt(train_data[39],usecols=(4,5,6,7,8,9,10))
	x_test = np.zeros((np.shape(x_test_buf)[0]-59),60,np.shape(x_test_buf)[1])
	for k in range(np.shape(x_test_buf)[0]-59):
		x_test[k] = x_test_buf[k:k+60]

	x_test = x_test/100
	print("Resizing of Data",i)
	
	for r in range(max_size-59):
		fc = r+60
		RNN_test_data = np.append(RNN_test_data,np.array(RNN_test_data_current[l[1]*(fc*30-1800):(l[1]*((fc+1)*30))]))
		RNN_test_label = np.append(RNN_test_label,semantic[fc,:])

	RNN_test_data = np.reshape(RNN_test_data,(int(len(RNN_test_data)/(l[1]*1800)),1800,l[1]))
	RNN_test_data = RNN_test_data/255	
	RNN_test_label = np.reshape(RNN_test_label,(int(len(RNN_test_label)/3),3))
	
	
score = model.evaluate([RNN_test_data, x_test],RNN_test_label,batch_size=10)
print(score)
cap.release()
