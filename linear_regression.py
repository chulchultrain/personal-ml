#module specifically for linear regression using numpy and then one with tensorflow
import numpy as np 

def predict(w,x,b):
    res = np.dot(w,x) + b  
    return res 

def calc_djdwi(w,x,b,y,i):
    res = 0 
    m = x.shape[0]
    predictions = np.zeros(m)
    for j in range(m):
        predictions[j] = predict(w,x[j],b)
    err = predictions - y
    term = err * x[:,i]
    total = np.sum(term)
    res = total / m 
    return res

def calc_djdb(w,x,b,y):
    res = 0 
    m = x.shape[0] 
    predictions = np.zeros(m)
    for j in range(m):
        predictions[j] = predict(w,x[j],b)
    err = predictions - y 
    total = np.sum(err)
    res = total / m 
    return res 

def gradient_descent(y,initial_w,x,initial_b):
    
    w = initial_w 
    b = initial_b 
    n = w.shape[0]
    steps = 10000
    djdw = np.zeros(n)
    djdb = 0
    alpha = .4
    for k in range(steps):
        for i in range(n):
            djdw[i] = calc_djdwi(w,x,b,y,i)
        djdb = calc_djdb(w,x,b,y)

    for i in range(n):  
        w[i] = w[i] - alpha * djdw[i]
    b  = b - alpha * djdb 

    return w,b

def cost_function(y,w,x,b):

    m = y.shape[0]
    res = 0
    predictions = np.zeros(m)
    for i in range(m):
        predictions[i] = predict(w,x[i],b)
    
    res = np.sum(np.square((y - predictions))) / (2 * m)
    # for i in range(m):
    #     sqe = (predict(w,x[i],b) - y[i]) ** 2 
    #     res += sqe 
    # res /= (2 * m)
    return res



x = np.array([(1,2),(2,3),(3,7)])
w = np.array([1,1])
b = 1 
y = np.array([4,6,12])
cost = cost_function(y,w,x,b)

new_w,new_b = gradient_descent(y,w,x,b)
new_cost = cost_function(y,new_w,x,new_b)
print(new_w,new_b)
print(new_cost)