import numpy as np
from matplotlib import pyplot as plt

def step(x):
    return np.array(x>0,dtype=np.int32)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def identity_function(x):
    return x

def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    return exp_a/sum_exp_a

class NN_layer:
    def __init__(self,in_vector_num,out_vector_num,func,prev_layer=None,next_layer=None,weight_matrix_init=np.array([])):
        self.input_num=in_vector_num
        self.output_num=out_vector_num
        self.func=func
        self.connect_layer(prev_layer=prev_layer,next_layer=next_layer)
        self.weight_matrix_init=weight_matrix_init
        IS_VARID_SIZE = weight_matrix_init.size != 0 and weight_matrix_init.shape[0] == self.input_num + 1 and weight_matrix_init.shape[1] == self.output_num
        if not IS_VARID_SIZE:
            print("invalid size")
            self.weight_matrix_init = np.arange((self.input_num + 1)*self.output_num).reshape((self.input_num + 1,self.output_num))

    def connect_layer(self,prev_layer=None,next_layer=None):
        #行列の数が合う必要あり
        self.prev_layer=prev_layer
        self.next_layer=next_layer
        if prev_layer != None:
            if self.input_num==prev_layer.output_num:
                prev_layer.next_layer=self
            else:
                print("error:前layerと要素数が一致していません")
        if next_layer != None:
            if self.output_num==next_layer.input_num:
                next_layer.prev_layer=self
            else: 
                print("error:後layerと要素数が一致していません")
    
    def calc(self,input_vector):
        if input_vector.size != self.input_num:
            print("不正な入力")
            return None
        input_vector = np.append(input_vector,1)                            #定数項θの追加　weightで重みづけされて各ノードへ伝わる
        output_vector = np.dot(input_vector,self.weight_matrix_init)
        return self.func(output_vector)

class Neural_Network:
    def __init__(self,start_layer):
        self.start_layer=start_layer
        iterator=start_layer
        while(iterator.next_layer!=None):
            iterator=iterator.next_layer
        self.end_layer=iterator
    
    def solve(self,input_vector):
        iterator=self.start_layer
        input=self.start_layer.calc(input_vector)
        while iterator.next_layer!=None:
            iterator=iterator.next_layer
            input=iterator.calc(input)
        return input

W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6],[0.1,0.2,0.3]])
W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6],[0.1,0.2]])
W3=np.array([[0.1,0.3],[0.2,0.4],[0.1,0.2]])
layer1=NN_layer(2,3,sigmoid,weight_matrix_init=W1)
layer2=NN_layer(3,2,sigmoid,prev_layer=layer1,weight_matrix_init=W2)
layer3=NN_layer(2,2,identity_function,prev_layer=layer2,weight_matrix_init=W3)

NN=Neural_Network(layer1)
x=np.array([1.0,0.5])
y=NN.solve(x)
print('y:',y)