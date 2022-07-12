import numpy as np
import nnetwork as NN

class Learning:
    def __init__(self,neural_network):
        self.neural_network = neural_network
    
    def numerical_gradient(func,x):
        h=1e-4
        grad=np.zeros_like(x)
        for idx in range(x.size):
            tmp_val = x[idx]
            x[idx]=tmp_val+h
            fxh1=func(x)

            x[idx]=tmp_val-h
            fxh2=func(x)
            
            grad[idx]=(fxh1-fxh2)/(2*h)
            x[idx]=tmp_val
        return grad

    def gradient_descent(func,init_input_arr,lr=0.01,step_num=100):
        x=init_input_arr
        for i in range(step_num):
            grad=Learning.numerical_gradient(func,x)
            x-=lr*grad
        return x

    def cross_entropy_error(output_arr, teach_arr):
        if output_arr.ndim == 1:
            y=output_arr.reshape(1,output_arr.size)
            t=teach_arr.reshape(1,output_arr.size)
        else:
            y=output_arr
            t=teach_arr
        return -np.sum(t*np.log(y+1e-7))/y.shape[0]

    def loss(input_arr,teach_arr, neural_network):
        y=neural_network.solve(input_arr)
        return Learning.cross_entropy_error(y,teach_arr)