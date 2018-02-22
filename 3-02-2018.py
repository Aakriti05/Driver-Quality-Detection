from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import merge, Convolution2D, MaxPooling2D, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
import cv2
from datetime import datetime
import numpy as np

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




def makeModel(frameHeight, frameWidth, nbChannels=3, nbClasses=3, nbRCL=5,
		 nbFilters=128, filtersize = 3):


	model = BuildRCNN(frameHeight, frameWidth, nbChannels, nbClasses, nbRCL, nbFilters, filtersize)
	return model

def BuildRCNN(frameHeight, frameWidth, nbChannels, nbClasses, nbRCL, nbFilters, filtersize):
    
    def RCL_block(l_settings, l, pool=True, increase_dim=False):
        input_num_filters = l_settings.output_shape[1]
        if increase_dim:
            out_num_filters = input_num_filters*2
        else:
            out_num_filters = input_num_filters
		   
        conv1 = Convolution2D(out_num_filters, 1, 1, border_mode='same')
        stack1 = conv1(l)   	
        stack2 = BatchNormalization()(stack1)
        stack3 = PReLU()(stack2)
        
        conv2 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', init = 'he_normal')
        stack4 = conv2(stack3)
        stack5 = merge([stack1, stack4], mode='sum')
        stack6 = BatchNormalization()(stack5)
        stack7 = PReLU()(stack6)
    	
        conv3 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', weights = conv2.get_weights())
        stack8 = conv3(stack7)
        stack9 = merge([stack1, stack8], mode='sum')
        stack10 = BatchNormalization()(stack9)
        stack11 = PReLU()(stack10)    
        
        conv4 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', weights = conv2.get_weights())
        stack12 = conv4(stack11)
        stack13 = merge([stack1, stack12], mode='sum')
        stack14 = BatchNormalization()(stack13)
        stack15 = PReLU()(stack14)    
        
        if pool:
            stack16 = MaxPooling2D((2, 2), border_mode='same')(stack15) 
            stack17 = Dropout(0.1)(stack16)
        else:
            stack17 = Dropout(0.1)(stack15)
            
        return stack17

    #Build Network
    input_img = Input(shape=(frameHeight, frameWidth, nbChannels))
    conv_l = Convolution2D(nbFilters, filtersize, filtersize, border_mode='same', activation='relu')
    l = conv_l(input_img)
    
    for n in range(nbRCL):
        if n % 2 ==0:
            l = RCL_block(conv_l, l, pool=False)
        else:
            l = RCL_block(conv_l, l, pool=True)
    
    out = Flatten()(l)        
    l_out = Dense(nbClasses, activation = 'softmax')(out)
    
    model = Model(input = input_img, output = l_out)
    
    return model

model = BuildRCNN(frameHeight, frameWidth, nbChannels, nbClasses, nbRCL, nbFilters, filtersize)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

for j in range(5):
	for i in range(39):
		videoName1 = dataFolderName[i]+videoName[i]
		cap = cv2.VideoCapture(videoName1)
		frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		videoDateString = videoName1.split('/')[-1][0:14]
		dataDateString = dataFolderName[i].split('/')[-1][0:14]
		filename = 'SEMANTIC_ONLINE.txt'
		videoDate = datetime.strptime(videoDateString, "%Y%m%d%H%M%S")
		dataDate = datetime.strptime(dataDateString, "%Y%m%d%H%M%S")
		delayVideoToData = (dataDate - videoDate).total_seconds()
		fc = 0

		print(delayVideoToData)
		semantic = np.loadtxt(dataFolderName[i]+filename,usecols=(11,12,13))

		if(delayVideoToData<=0):	
			max_size = int(min(np.round(frameCount/30),np.shape(semantic)[0]+delayVideoToData))
			ret = True

			while (fc<max_size and ret):
				buf = np.empty((30, frameHeight, frameWidth, 3), np.dtype('uint8'))
				while(fc1<30):		
					ret, buf[fc1] = cap.read()
					fc1 += 1
				train_data = buf

				train_data = train_data/255
				train_label = semantic[(-delayVideoToData+fc):(-delayVideoToData+fc+1),:]
				model.fit(train_data,train_label,batch_size=1,epochs=1)
				fc += 1 
	
		elif(delayVideoToData>0):	
			max_size = int(min(np.round(frameCount/30)-delayVideoToData,np.shape(semantic)[0]))
	
			ret = True
			t = 0
			while(t<delayVideoToData*30):
				discard = cap.read()
				t += 1

			while (fc<(max_size) and ret):
				buf = np.empty((30, frameHeight, frameWidth, 3), np.dtype('uint8'))
				while(fc1<30):
					ret, buf[fc1] = cap.read()
					fc1 += 1
				train_data = buf

				train_data = train_data/255
				train_label = semantic[fc:(fc+1),:]
				model.fit(train_data,train_label,batch_size=1,epochs=1)
				fc += 1
		cap.release()


videoName1 = dataFolderName[39]+videoName[39]
cap = cv2.VideoCapture(videoName1)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

videoDateString = videoName1.split('/')[-1][0:14]
dataDateString = dataFolderName[i].split('/')[-1][0:14]
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
	while (fc<max_size and ret):
		buf = np.empty((30, frameHeight, frameWidth, 3), np.dtype('uint8'))
		while(fc1<30):		
			ret, buf[fc1] = cap.read()
			fc1 += 1
		test_data = buf
		test_data = test_data/255
		test_label = semantic[(-delayVideoToData+fc):(-delayVideoToData+fc+1),:]
		score = model.evaluate(test_data,test_label,batch_size=1)
		print(score)
		fc += 1 
	
elif(delayVideoToData>0):	
	max_size = int(min(np.round(frameCount/30)-delayVideoToData,np.shape(semantic)[0]))

	ret = True
	t = 0
	while(t<delayVideoToData*30):
		discard = cap.read()
		t += 1

	while (fc<(max_size) and ret):
		buf = np.empty((30, frameHeight, frameWidth, 3), np.dtype('uint8'))
		while(fc1<30):
			ret, buf[fc1] = cap.read()
			fc1 += 1
		test_data = buf		
		test_data = test_data/255
		test_label = semantic[fc:(fc+1),:]
		score = model.evaluate(test_data,test_label,batch_size=1)
		print(score)
		fc += 1
cap.release()
#ln -s cv2.cpyh.......so cv2.so