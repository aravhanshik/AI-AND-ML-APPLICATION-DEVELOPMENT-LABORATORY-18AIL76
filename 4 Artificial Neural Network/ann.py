import numpy as np

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
y = y/100

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))
  
#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)
  
#Variable initialization
epoch=10000
lr=0.1
inputlayer_neurons = 2
hiddenlayer_neurons = 3
output_neurons = 1

#weight initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))

for i in range(epoch):
  
    #Forward Propogation
    hinp=np.dot(X,wh)
    hlayer_act = sigmoid(hinp)
    outinp=np.dot(hlayer_act,wout)
    output = sigmoid(outinp)
  
    #Backpropagation
    EO = y-output
    outgrad = derivatives_sigmoid(output)
    d_output = EO* outgrad
    EH = d_output.dot(wout.T)
    hiddengrad = derivatives_sigmoid(hlayer_act)
    d_hiddenlayer = EH * hiddengrad
    wout += hlayer_act.T.dot(d_output) *lr
    wh += X.T.dot(d_hiddenlayer) *lr
  
print("Input: \n" + str(X))
y=y*100
output=output*100
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)
