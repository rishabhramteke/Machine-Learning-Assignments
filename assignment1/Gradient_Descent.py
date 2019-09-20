import csv
import numpy as np
import matplotlib.pyplot as plt
filepath ="/home/rishabh/Downloads/GNR652/asmt1/data1.csv"

file = np.genfromtxt(filepath, delimiter=',')
file=file[1:]
file1=np.ones((135,19))
file1[:,1:] = file
file = file1

#Initializing W and other variables
# W1=np.random.rand(18,1)  Not working
W1=np.random.random_sample((18))

loss =0
grad = 0
train_loss_history ={}
test_loss_history ={}
#----------------------

#Other Parameters
alpha = 0.0001
epochs = 100000
Lambda = 0.1
#---------------------

#Normalization
norm = [1,27,4,5,1,3,2,1,1,1,4,1,7,1,8,1,4,1,23.4]
norm = np.array(norm)
file = file/norm
#--------------------------

X_train=file[:108,:-1]
Y_train=file[:108,18]
X_test=file[108:,:-1]
Y_test=file[108:,18]

print("\n------Data Shapes-----\n")
print("MyData_shape =",file.shape)
print("X_train_shape =",X_train.shape,"Y_train_shape =",Y_train.shape)
print("X_test_shape =",X_test.shape,"X_test_shape =",Y_test.shape)
print("\n -----------Training ----------\n\n")

def loss_n_grad(X_train,Y_train,W1,alpha):
	J= (X_train.dot(W1) - Y_train)
	J = J.dot(J.transpose())
	J = 0.5* np.sum(J) + Lambda * (np.sum(W1*W1) - W1[0]**2)
	J /= X_train.shape[0] 
	grads_W =X_train.transpose().dot((X_train.dot(W1) - Y_train)) + 2*Lambda*W1 
	

	return J,grads_W


loss , grad = loss_n_grad(X_train,Y_train,W1,alpha)
print("Initial loss : ",loss)

# For optimizing W
for i in range(epochs):
	loss , grad  = loss_n_grad(X_train,Y_train,W1,alpha)
	loss_test, g = loss_n_grad(X_test,Y_test,W1,alpha)
	W1 = W1 - alpha * grad
	if i>1000:
		train_loss_history[i/100] = loss
		test_loss_history[i/100] = loss_test
	if(i%10000==0):
		print(loss)



y_pred_train = X_train.dot(W1)
diff_train = (y_pred_train - Y_train)*23.4
per_train_error = diff_train/(Y_train*23.4)*100
per_train_error = np.absolute(per_train_error)
per_train_error = np.sum(per_train_error)/108
print ("Percentage train error:\n",per_train_error)

y_pred1=X_test.dot(W1)
diff= (y_pred1 - Y_test)*23.4
MSE = np.sqrt(np.sum(diff **2))
MSE /=27

loss_test, g = loss_n_grad(X_test,Y_test,W1,alpha)
print("MSE_error",MSE)
print("loss_test : ",loss_test)
per_test_error = diff/(Y_test*23.4)*100
per_test_error = np.absolute(per_test_error)
per_test_error = np.sum(per_test_error)/27
print ("Percentage test error:",per_test_error)
print (W1)

#Test Variance
variance=np.sqrt((np.sum(loss_test*loss_test))/108)
print ("Variance =", variance)


# save_name_txt = "/home/rishabh/Downloads/GNR652/asmt1/gnr_data/models/file:"+"  Percentage_test_error:"+str(per_test_error)+"%, MSE:"+str(MSE)+".txt"
# model_file = open(save_name_txt,"w")
# model_file.write("Model details \n #")
# model_file.write("\n\n W1\n"+str(W1))
# model_file.write("\n\n diff\n"+str(diff))
# model_file.write("\n\n MSE\n"+str(MSE))
# model_file.write("\n\n train error per\n"+str(per_train_error))
# model_file.write("\n\n test error per\n"+str(per_test_error))
# model_file.write("\n\n epochs\n"+str(epochs))
# model_file.close()

plt.plot(range(len(train_loss_history)), list(train_loss_history.values()))
plt.plot(range(len(test_loss_history)), list(test_loss_history.values()))
plt.legend(['Train Loss','Test Loss'])
plt.xlabel('Number of iterations(in 100s)')
plt.ylabel('Training Loss')
plt.grid(True)
plt.title('Loss vs iterations')
plt.savefig("/home/rishabh/Downloads/GNR652/asmt1/gnr_data/models/file:"+"  Percentage_test_error:"+str(per_test_error)+".png")
plt.show()





