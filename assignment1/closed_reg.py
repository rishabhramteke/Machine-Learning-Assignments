import csv
import numpy as np
import matplotlib.pyplot as plt
filepath ="/home/rishabh/Downloads/data1.csv"

file = np.genfromtxt(filepath, delimiter=',')
file=file[1:]
file1=np.ones((135,19))
file1[:,1:] = file
file = file1


#Other Params
alpha = 0.0001
epoch = 55000
Lambda = 0.1
#---------------------

norm_fact = [1,27,4,5,1,3,2,1,1,1,4,1,7,1,8,1,4,1,23.4]
norm_fact = np.array(norm_fact)
file = file/norm_fact

X_train=file[:108,:-1]
Y_train=file[:108,18]
X_test=file[108:,:-1]
Y_test=file[108:,18]


#Closed Regression
def closed_regression(x, y, l):
    
 	# W2 = np.zeros(18)
	# W2 = X_train.transpose().dot(X_train.dot(X_train.transpose())).dot(Y_train)
	# W2 = W2/
	# print(W2)

    I = np.identity(18)
    
    w = np.dot(x.T,x)
    w = np.add(w,np.dot(l,I))
    w = np.linalg.inv(w)
    w = np.dot(w,(x.T))
    w = np.dot(w,y)

    #w is 18 X 1
    return w

W_train = closed_regression(X_train, Y_train, Lambda)
#W_test = closed_regression(X_test, Y_test, Lambda)

print W_train

#Train Loss
J= (X_train.dot(W_train) - Y_train)
J = J.dot(J.transpose())
J = 0.5* np.sum(J) 
J /= X_train.shape[0]

print ("train_loss", J)

#Test Loss
J1= (X_test.dot(W_train) - Y_test)
J1 = J1.dot(J1.transpose())
J1 = 0.5* np.sum(J1) 
J1 /= X_train.shape[0]

print ("test_loss", J1)

#Train Variance
variance=np.sqrt((np.sum(J*J))/108)
print ("Train_Variance =", variance)

#Test Variance
variance1=np.sqrt((np.sum(J1*J1))/108)
print ("Test_Variance =", variance1)

