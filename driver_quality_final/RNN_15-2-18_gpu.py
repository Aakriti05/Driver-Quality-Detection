from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, GRU
from keras.optimizers import Adam, SGD, RMSprop
import numpy as np
import random
import cv2
import display as dp
from datetime import datetime

def arrange_data(x_train, y_train):
	y_train = np.nan_to_num(y_train)
	print("y_train:", np.shape(y_train))
	
	#x_train = np.reshape(x_train,(np.shape(x_train)[0],np.shape(x_train)[1]),int(np.shape(x_train)[2]/60))
	x_train = np.nan_to_num(x_train)
	print("x_train:", np.shape(x_train))
	return x_train, y_train

train_data = ["D1/20151110175712-16km-D1-NORMAL1-SECONDARY/SEMANTIC_ONLINE.txt","D1/20151110180824-16km-D1-NORMAL2-SECONDARY/SEMANTIC_ONLINE.txt","D1/20151111123124-25km-D1-NORMAL-MOTORWAY/SEMANTIC_ONLINE.txt","D1/20151111125233-24km-D1-AGGRESSIVE-MOTORWAY/SEMANTIC_ONLINE.txt","D1/20151111132348-25km-D1-DROWSY-MOTORWAY/SEMANTIC_ONLINE.txt","D1/20151111134545-16km-D1-AGGRESSIVE-SECONDARY/SEMANTIC_ONLINE.txt","D1/20151111135612-13km-D1-DROWSY-SECONDARY/SEMANTIC_ONLINE.txt",
				"D2/20151120131714-26km-D2-NORMAL-MOTORWAY/SEMANTIC_ONLINE.txt","D2/20151120133502-26km-D2-AGGRESSIVE-MOTORWAY/SEMANTIC_ONLINE.txt","D2/20151120135152-25km-D2-DROWSY-MOTORWAY/SEMANTIC_ONLINE.txt","D2/20151120160904-16km-D2-NORMAL1-SECONDARY/SEMANTIC_ONLINE.txt","D2/20151120162105-17km-D2-NORMAL2-SECONDARY/SEMANTIC_ONLINE.txt","D2/20151120163350-16km-D2-AGGRESSIVE-SECONDARY/SEMANTIC_ONLINE.txt","D2/20151120164606-16km-D2-DROWSY-SECONDARY/SEMANTIC_ONLINE.txt",
				"D3/20151126110502-26km-D3-NORMAL-MOTORWAY/SEMANTIC_ONLINE.txt","D3/20151126113754-26km-D3-DROWSY-MOTORWAY/SEMANTIC_ONLINE.txt","D3/20151126124208-16km-D3-NORMAL1-SECONDARY/SEMANTIC_ONLINE.txt","D3/20151126125458-16km-D3-NORMAL2-SECONDARY/SEMANTIC_ONLINE.txt","D3/20151126130707-16km-D3-AGGRESSIVE-SECONDARY/SEMANTIC_ONLINE.txt","D3/20151126132013-17km-D3-DROWSY-SECONDARY/SEMANTIC_ONLINE.txt","D3/20151126134736-26km-D3-AGGRESSIVE-MOTORWAY/SEMANTIC_ONLINE.txt",
				"D4/20151203171800-16km-D4-NORMAL1-SECONDARY/SEMANTIC_ONLINE.txt","D4/20151203173103-17km-D4-NORMAL2-SECONDARY/SEMANTIC_ONLINE.txt","D4/20151203174324-16km-D4-AGGRESSIVE-SECONDARY/SEMANTIC_ONLINE.txt","D4/20151203175637-17km-D4-DROWSY-SECONDARY/SEMANTIC_ONLINE.txt","D4/20151204152848-25km-D4-NORMAL-MOTORWAY/SEMANTIC_ONLINE.txt","D4/20151204154908-25km-D4-AGGRESSIVE-MOTORWAY/SEMANTIC_ONLINE.txt","D4/20151204160823-25km-D4-DROWSY-MOTORWAY/SEMANTIC_ONLINE.txt",
				"D5/20151209151242-25km-D5-NORMAL-MOTORWAY/SEMANTIC_ONLINE.txt","D5/20151209153137-25km-D5-AGGRESSIVE-MOTORWAY/SEMANTIC_ONLINE.txt","D5/20151211160213-25km-D5-DROWSY-MOTORWAY/SEMANTIC_ONLINE.txt","D5/20151211162829-16km-D5-NORMAL1-SECONDARY/SEMANTIC_ONLINE.txt","D5/20151211164124-17km-D5-NORMAL2-SECONDARY/SEMANTIC_ONLINE.txt","D5/20151211165606-12km-D5-AGGRESSIVE-SECONDARY/SEMANTIC_ONLINE.txt","D5/20151211170502-16km-D5-DROWSY-SECONDARY/SEMANTIC_ONLINE.txt",
				"D6/20151217162714-26km-D6-NORMAL-MOTORWAY/SEMANTIC_ONLINE.txt","D6/20151217164730-25km-D6-DROWSY-MOTORWAY/SEMANTIC_ONLINE.txt","D6/20151221112434-17km-D6-NORMAL-SECONDARY/SEMANTIC_ONLINE.txt","D6/20151221113846-16km-D6-DROWSY-SECONDARY/SEMANTIC_ONLINE.txt","D6/20151221120051-26km-D6-AGGRESSIVE-MOTORWAY/SEMANTIC_ONLINE.txt"]
# ,"D3/20151126113754-26km-D3-DROWSY-MOTORWAY/SEMANTIC_ONLINE.txt"
random.shuffle(train_data)

#test_data = ["D6/20151217162714-26km-D6-NORMAL-MOTORWAY/SEMANTIC_ONLINE.txt","D6/20151217164730-25km-D6-DROWSY-MOTORWAY/SEMANTIC_ONLINE.txt","D6/20151221112434-17km-D6-NORMAL-SECONDARY/SEMANTIC_ONLINE.txt","D6/20151221113846-16km-D6-DROWSY-SECONDARY/SEMANTIC_ONLINE.txt","D6/20151221120051-26km-D6-AGGRESSIVE-MOTORWAY/SEMANTIC_ONLINE.txt"]
# test_file = ["D1/20151110175712-16km-D1-NORMAL1-SECONDARY/"]
# videoName = ["20151110175712-16km-D1-NORMAL1-SECONDARY.mp4"]

test_file = ["D3/20151126113754-26km-D3-DROWSY-MOTORWAY/"]
videoName = ["20151126113753-26km-D3-DROWSY-MOTORWAY.mp4"]

data_dim = 7
timesteps = 60 #29150
num_classes = 3
'''
model = Sequential()
model.add(GRU(128, dropout = 0.25, recurrent_dropout = 0.2, return_sequences=True, input_shape=(timesteps,data_dim)))  # returns a sequence of vectors of dimension 32
model.add(GRU(134, dropout = 0.25,recurrent_dropout = 0.2, return_sequences=True))  # returns a sequence of vectors of dimension 3
model.add(GRU(3, dropout = 0.25, recurrent_dropout = 0.2))
opt = SGD(lr=0.001, decay=5e-4, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=opt, metrics=['mae','accuracy'])

y_train_buf = np.loadtxt(train_data[0],usecols=(11,12,13))
y_train = y_train_buf[59:,:]

for j in range(len(train_data)):
	if(j == 0):
		continue
	y_temp_buf = np.loadtxt(train_data[j],usecols=(11,12,13))
	y_temp = y_temp_buf[59:,:]
	y_train = np.append(y_train,y_temp, axis = 0)


x_train_buf = np.loadtxt(train_data[0],usecols=(4,5,6,7,8,9,10))
x_train = np.zeros((np.shape(x_train_buf)[0]-59,60,np.shape(x_train_buf)[1]))
print(np.shape(x_train))

for k in range(np.shape(x_train_buf)[0]-59):
	x_train[k] = x_train_buf[k:k+60]
	
for j in range(len(train_data)):
	if(j == 0):
		continue
	x_temp_buf = np.loadtxt(train_data[j],usecols=(4,5,6,7,8,9,10))
	x_temp = np.zeros((np.shape(x_temp_buf)[0]-59,60,np.shape(x_temp_buf)[1]))

	for k in range(np.shape(x_temp_buf)[0]-59):
		x_temp[k] = x_temp_buf[k:k+60]
	
	x_train = np.append(x_train,x_temp, axis = 0)
x_train = x_train/100
x_train, y_train = arrange_data(x_train, y_train)
model.fit(x_train,y_train,batch_size=1024,epochs=150, verbose = 2)

model.save("save_RNN.h5")
'''
model = load_model("save_RNN_decent.h5")
'''
print("Evaluating:")
y_test_buf = np.loadtxt(test_data[0],usecols=(11,12,13))
y_test = y_test_buf[59:,:]

for j in range(len(test_data)):
	if(j == 0):
		continue
	y_temp_buf = np.loadtxt(test_data[j],usecols=(11,12,13))
	y_temp = y_temp_buf[59:,:]
	y_test = np.append(y_test,y_temp, axis = 0)
print(np.shape(y_test))

x_test_buf = np.loadtxt(test_data[0],usecols=(4,5,6,7,8,9,10))
x_test = np.zeros((np.shape(x_test_buf)[0]-59,60,np.shape(x_test_buf)[1]))

for k in range(np.shape(x_test_buf)[0]-59):
	x_test[k] = x_test_buf[k:k+60]
	
for j in range(len(test_data)):
	if(j == 0):
		continue
	x_temp_buf = np.loadtxt(test_data[j],usecols=(4,5,6,7,8,9,10))
	x_temp = np.zeros((np.shape(x_temp_buf)[0]-59,60,np.shape(x_temp_buf)[1]))


	for k in range(np.shape(x_temp_buf)[0]-59):
		x_temp[k] = x_temp_buf[k:k+60]
	
	x_test = np.append(x_test,x_temp, axis = 0)
x_test = x_test/100
x_test, y_test = arrange_data(x_test, y_test)

y_test_predict = model.predict(x_test)
for i in range(np.shape(y_test_predict)[0]):
	print(y_test_predict[i],y_test[i])
score = model.evaluate(x_test,y_test,batch_size=1500)
print("Score:",score)'''

print("Evaluating:")

for j in range(len(test_file)):
	test_data = test_file[j] + "SEMANTIC_ONLINE.txt"
	y_temp_buf = np.loadtxt(test_data,usecols=(11,12,13))
	y_temp = y_temp_buf[59:,:]
	x_temp_buf = np.loadtxt(test_data,usecols=(4,5,6,7,8,9,10))
	
	videoName1 = test_file[j] + videoName[j]

	videoDateString = videoName1.split('/')[-1][0:14]
	dataDateString = test_file[j].split('/')[-2][0:14]
	videoDate = datetime.strptime(videoDateString, "%Y%m%d%H%M%S")
	dataDate = datetime.strptime(dataDateString, "%Y%m%d%H%M%S")
	delayVideoToData = (dataDate - videoDate).total_seconds()

	cap = cv2.VideoCapture(videoName1)
	if(delayVideoToData<=0):
		print("Initial one minute")
		for i in range(1800):
			_,frame = cap.read()
			dp.display(frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		for k in range(np.shape(x_temp_buf)[0]-59):
			p = k-int(delayVideoToData)
			x_temp = x_temp_buf[p:p+60]
			x_temp = np.reshape(x_temp,(1,60,7))
			x_temp = x_temp/100
			y_pred = model.predict(x_temp) #for every 
			sen = x_temp[:,59,:]
			print(y_pred,y_temp[k])
			y_pred = 10*y_pred
			for i in range(30):
				ret, frame = cap.read()
				dp.display(frame,np.reshape(y_pred,(1,3)).tolist()[0],np.reshape(sen,(1,7)).tolist()[0])
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
		
	elif(delayVideoToData>0):
		print('Initial one minute')
		for i in range(1800+int(delayVideoToData)*30):
			_,frame = cap.read()
			dp.display(frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		for k in range(np.shape(x_temp_buf)[0]-59):
			x_temp = x_temp_buf[k:k+60]
			x_temp = np.reshape(x_temp,(1,60,7))
			x_temp = x_temp/100
			y_pred = model.predict(x_temp) #for every minute
			sen = x_temp[:,59,:]
			print(y_pred,y_temp[k])
			y_pred = 10*y_pred
			for i in range(30):
				ret, frame = cap.read()
				dp.display(frame,np.reshape(y_pred,(1,3)).tolist()[0],np.reshape(sen,(1,7)).tolist()[0])
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break





