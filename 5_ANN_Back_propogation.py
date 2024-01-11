import numpy as np

X = np.array(([2,9],[1,5],[3,6]), dtype=float)
Y = np.array(([92],[86],[89]), dtype=float)
X = X/np.amax(X)
Y = Y/100

def sigmoid(X):
    return 1/(1+np.exp(-X))
def derivatives_sigmoid(X):
    return 1 * (1 - X)

epoch = 5000
lr = 0.1
inputlayer_neurons = 2
hlayer_neurons = 3
output_neurons  = 1

wh = np.random.uniform(size=(inputlayer_neurons,hlayer_neurons))
bh = np.random.uniform(size=(1,hlayer_neurons))
wout = np.random.uniform(size=(hlayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

for i in range(epoch):
    hinp = np.dot(X,wh)
    hlayer_act = sigmoid(hinp)
    inpout = np.dot(hlayer_act, wout)
    output = sigmoid(inpout)

    outgrad = derivatives_sigmoid(output)
    hlayergrad = derivatives_sigmoid(hlayer_act)

    EO = Y-output
    d_output = EO * outgrad

    EH = d_output.dot(wout.T)
    d_hlayer = EH * hlayergrad

    wout += hlayer_act.T.dot(d_output) * lr
    wh += X.T.dot(d_hlayer) * lr

print("\n Input : \n",str(X))
print("\n Actual output : \n",str(Y))
print("\n Predicted output : \n",output)