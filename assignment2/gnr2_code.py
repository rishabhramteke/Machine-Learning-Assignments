#Importing Libraries
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy.linalg import inv
from cvxopt import matrix as cvxopt_matrix #Importing with custom names to avoid issues with numpy matrix
from cvxopt import solvers as cvxopt_solvers

#Importing data
filepath ="/home/rishabh/Downloads/170070046_assignment2/set5.csv"
File = np.genfromtxt(filepath, delimiter=',') #converting csv to numpy
# X_1 = File[:98, 0:30]
# X_2 = File[98:, 0:30]
np.random.shuffle(File) #shuffle data in File

#defining training and testing set
X_train = File[:180, 0:30]
Y_train = File[:180, [30]]
X_test = File[180:, 0:30]
Y_test = File[180:, [30]]

#Data shapes
print("\n####  Data Shapes  ####\n")
print("MyData_shape =",File.shape)
print("X_train_shape =",X_train.shape,"Y_train_shape =",Y_train.shape)
print("X_test_shape =",X_test.shape,"X_test_shape =",Y_test.shape)

# Initializing values and computing H.
m, n = X_train.shape
y = Y_train.reshape(-1, 1) * 1.
X_dash = y * X_train
H = np.dot(X_dash, X_dash.T) * 1.

# Converting into cvxopt format
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(-np.eye(m))
h = cvxopt_matrix(np.zeros(m))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

# Setting solver parameters 
cvxopt_solvers.options['show_progress'] = False
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10

# Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x']) #alphas are langrange multipliers

#Computing W parameter
w = np.multiply(alphas, Y_train)
w = w.transpose().dot(X_train)

#computing b parameter

#Method 1
# l = w.dot(X_1.transpose())
# z = w.dot(X_2.transpose())
# c1 = l.min(1)
# c2 = z.max(1)
# b = (-0.5) * (c1 + c2)

#Method 2
S = (alphas > 1e-40).flatten()
b = y[S] - np.dot(X_train[S], w.T)

#parameter shapes
print("\n####  Parameter Shapes  ####\n")
print("#### alphas shape =",alphas.shape)
print("#### W shape =",w.shape)
print("#### b shape =",b.shape)

#Display results
print("\n####  Display results  ####\n")
print('Alphas = ',alphas[alphas > 1e-10])
print('w = ', w.flatten())
print('b = ', b[0])

# Testing data
k = w.dot(X_test.transpose()) + b
k = np.multiply(k, Y_test.transpose())
e, f = k.shape
c = np.ones((f, 1))

#Accuracy
for i in range(f):
    if k[0][i] >= 0:
        c[i] = 1
    else:
        c[i] = 0
#print(c)
ans = np.mean(c) * 100
print("\n####  Testing Data  ####\n")
print("#### Accuracy: " + str(ans))
