# Advanced Driver Assistance System for Indian Conditions

**Abstract**— Driver assistance system is useful because it helps the driver manage in the situations of concern. In this, a four subsystem driver assistance system has been proposed. To make the system work, various advanced Deep Learning and computer vision based techniques have been proposed. The findings and observations suggest that driver assistance systems can have a positive effect on the casualties and damage resulting from road accidents. An accuracy as high as 92% was been obtained from pedestrian detection system, Driver Drowsiness Detection was found to be highly effective, Driver Rating system which is a new research topic also gave 86% of accuracy. (I took up the work for Driver Rating Subsystem in a team of 5 where everybody was alloted different subsystem)

## Diver Quality Detection
### Dataset
We have chosen _UAH-Driveset_ for training the model. The UAH-DriveSet is a public collection of data captured by a driving monitoring app DriveSafe (available on apple market) when used by various testers in different environments.
The dataset includes per second data on Acceleration, Braking, Turning, Weaving, Drifting, Overspeed, Car follow The above mentioned features are taken as input features for the RNN model that we have defined , whereas video data is input for our RCNN model. Output labels include Normal, Drowsy and Aggressive. It tells the degree of each on the scale of 1-10.

### Architecture: 
Our architecture consists of Recurrent Neural Network and Recurrent Convolution Neural Network. 
RNN : 7 features listed above are taken as input for 60 seconds. Quality of driving in terms of Normal Driving, Drowsy Driving and Aggressive Driving are predicted for this one minute. 
RCNN: Video data is input to Recurrent CNN along with Normal, Drowsy and Aggressive as the predicted labels.

A huge amount of research has been done on CNN [3] for image classification, object detection etc and it has been giving great result. But video classification is a much difficult task and not too much work has been done due to its major drawbacks of lack of sufficient amount of classifiable data and computational cost. Along with spatial, temporal relationship is also to be established. There are 5 video classification methods available:
1. Classifying one frame at a time with a ConvNet.
2. Using a time-distributed ConvNet and passing the features to an RNN, in one network.
3. Using a 3D convolutional network. [1] [4]
4. Extracting features from each frame with a ConvNet and passing the sequence to a separate MLP.
5. Extracting features from each frame with a ConvNet and passing the sequence to a separate RNN. (This is out final architecture) [2]
6. Having CNN as nodes in RNN making a Long/Short Term Memory of Images. 

Last two architectures give highly accurate results and therefore are more popular than the rest. We have used 5th architecture due to its lesser computational power.
Pre-trained weights of Resnet50 model is used for extracting out features from each frame of video, converting 960x540x3 video data into 1000 features. Input to Resnet50 is supposed to be of size 224x224x3 therefore, 960x540x3 is resized to the same. This reduces resolution and thus computational cost. Each second contains 30 video frames. Thus, 30x60 frames are converted into 30x60 feature vectors, which are then fed into RNN.
Temporal relationship is then found out for past 60 seconds for both RNN and RCNN. Combining the two architecture together improved the performance when compared with individual architectures.  

Note : It can be noted that first 59 values of features and labels will be discarded for each individual. But this does not create problem as the size of each individual’s data is large. Also n sec video has been bifurcated to give n-59 sec second small sized video. This is useful for short term temporal relationship.

### References
1. S. Ji, W. Xu, M. Yang, and K. Yu. 3D convolutional neural networks for human action recognition. PAMI, 35(1):221–231, 2013.
2. Long-term Recurrent Convolutional Networks for Visual Recognition and Description Jeff Donahue, Lisa Anne Hendricks, Marcus Rohrbach, Subhashini Venugopalan, Sergio Guadarrama, Kate Saenko, Trevor Darrell
3.  Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradientbased learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.
4. M. Baccouche, F. Mamalet, C. Wolf, C. Garcia, and A. Baskurt. Sequential deep learning for human action
recognition. In Human Behavior Understanding, pages 29–  39. Springer, 2011. 2, 3
