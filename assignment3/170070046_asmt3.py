#author: Rishabh Ramteke

#Importing libraries
import numpy as np
import scipy.io

#data 
no_features = 200
no_labels = 16

#Load Data
mat1 = scipy.io.loadmat('Indian_pines_corrected.mat')
mat2 = scipy.io.loadmat('Indian_pines_gt.mat')

data1 = mat1["indian_pines_corrected"]
data2 = mat2["indian_pines_gt"]

data1 = data1.reshape(21025,200)
data2 = data2.reshape(21025,1)

data1 = np.array(data1, dtype=float)
data2 = np.array(data2, dtype=int)

#remove zero class 
LIST = list()
for i in range(len(data2)):
	if data2[i]==0:
		LIST = LIST + [i]

data2 = np.delete(data2, LIST, axis=0)
data1 = np.delete(data1, LIST, axis=0)

#concatenate attributes and labels
data4 = np.concatenate((data1, data2), axis=1)
np.random.shuffle(data4)
print('Shape of data :', np.shape(data4))

#50% traing data & 50% test data
print('###Shapes###')
Xdata_train = data4[:5125,:200]
print('Training set X data:',np.shape(Xdata_train))
Xdata_test = data4[5125:,:200]
print('Test set X data:',np.shape(Xdata_test))
Ydata_train = data4[:5125,200]
print('Training set Y data:',np.shape(Ydata_train))
Ydata_test = data4[5125:,200]
print('Test set Y data:',np.shape(Ydata_test))

#normalize data
Xdata_train = Xdata_train/np.max(Xdata_train)

#Parameter
theta = np.random.rand(16,200)

#one hot encoding
a = Ydata_train.ravel()
for i in range(len(a)):
	a[i] = a[i] - 1
a = a.astype(int)

Ytrain_onehot = np.eye(no_labels)[a]
print('One hot encoding(train) shape :', np.shape(Ytrain_onehot))
#shape of Ytrain_onehot is (5125,16)
b = Ydata_test.ravel()
for i in range(len(b)):
	b[i] = b[i] - 1
b = b.astype(int)
Ytest_onehot = np.eye(no_labels)[b]
print('One hot encoding(test) shape :', np.shape(Ytest_onehot))

#hyperparamters
print('###Hyperparamters###')
alpha = 0.009;
print('learning rate : ', alpha)
#Lambda = 0.01;
no_iteration = 501

#Defining softmax
def softmax(X, W):
	#W is weight
	scores = Xdata_train.dot(W.transpose())
	# scores shape = (5125,16)
	scores_exp = np.exp(scores)
	scores_exp_sum=np.transpose(scores_exp.sum(axis=1))
	#shape of sum is (5125,)
	prob = np.zeros((5125,16))
	prediction = np.zeros((5125,16))
	for j in range(5125):
		prob[j] = scores_exp[j]/scores_exp_sum[j]
	#shape of prob is (5125,16)	
	#shape of prediction must be (5125,16)
	for i in range(5125):
		for j in range(16):
			#the class which has the maximum probability is our predicted class
			if (np.max(prob[i]) == prob[i][j]): 
				prediction[i][j] = 1


	return prob, prediction

#Define loss and gradient
def gradient_descent(X_train, Ytrain_onehot,W,prob, log_prob):
	#J is loss function
	J = 0;
	#Using softmax regression method
	for i in range(X_train.shape[0]): 
		for j in range(no_labels): #16 is no. of classes
			J = J - Ytrain_onehot[i][j]*log_prob[i][j]
	
	J /= X_train.shape[0]
	grads_W =prob.transpose().dot(X_train) #Using softmax regression method
	grads_W /= X_train.shape[0]
	return J,grads_W

#training
for i in range(no_iteration):
	probability, prediction = softmax(Xdata_train, theta)
	log_prob = np.log(probability)
	LOSS, grads_theta = gradient_descent(Xdata_train,Ytrain_onehot,theta,probability, log_prob)
	if (i % 100 == 0):
		print('epoch : ', i, 'loss : ',LOSS)
	theta = theta - alpha*grads_theta;

#Checking Test Accuracy 
Xdata_test = Xdata_test/np.max(Xdata_test);
probability_test ,prediction_test = softmax(Xdata_test,theta)

count=0; #to check no. of correct labels
for i in range(Xdata_test.shape[0]):
	for j in range(no_labels):
		if (Ytest_onehot[i][j]==prediction_test[i][j]):
			count = count+1

count /= no_labels #since we get 16 equalities for correct classification
accuracy = (count/Xdata_test.shape[0]) *100;
print('Test Accuracy', accuracy)
