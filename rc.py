from matplotlib import pyplot as plt
import numpy as np


#[時系列,要素]の入出力を想定
#三次元以上(要素が２次元、教師データが複数など)を想定していない
class ESN:
    rng=np.random.default_rng()
    def __init__(self,input_size,output_size,reserver_size,func=np.tanh,nonzero_probability=0.1,feedback_scale=0.3,input_scale=1,leadout_input_scale=0,spectrum_radius=1,rcond=1.5e-10):
        self.input_size=input_size
        self.output_size=output_size
        self.reserver_size=reserver_size
        self.func=func
        self.leadout_input_scale=leadout_input_scale
        self.rcond=rcond

        #重み行列の初期化((-1,1)の一様分布乱数)
        #入力、リザバーリカレント、出力フィードバック結合重み行列は初期値のまま固定
        self.w_input=(ESN.rng.random(size=(self.input_size,self.reserver_size))*2-1)*input_scale
        self.w_reserver=ESN.rng.random(size=(self.reserver_size,self.reserver_size))
        self.w_reserver=np.where(self.w_reserver<=nonzero_probability,1,0)
        random_arr=ESN.rng.random(size=(reserver_size,reserver_size))*2-1
        self.w_reserver=np.multiply(self.w_reserver,random_arr)
        value,vector=np.linalg.eig(self.w_reserver)
        params=spectrum_radius/np.max(np.abs(value))
        self.w_reserver=params*self.w_reserver
        self.w_feedback=(ESN.rng.random(size=(self.output_size,self.reserver_size))*2-1)*feedback_scale

        self.w_output=ESN.rng.random(size=(self.reserver_size+self.input_size+1,self.output_size))

    def solve(self,input_arr,after_step):
        output_arr=np.empty(shape=(input_arr.shape[0]+after_step,self.output_size))
        leadout_input_arr=input_arr*self.leadout_input_scale

        reserver_arr=self.func(np.dot(input_arr[0,:],self.w_input))
        output_arr[0,:]=np.dot(np.r_[1,leadout_input_arr[0,:],reserver_arr],self.w_output)

        for i in range(1,input_arr.shape[0]):
            reserver_arr=self.func(np.dot(input_arr[i,:],self.w_input)+np.dot(reserver_arr,self.w_reserver)+np.dot(output_arr[i-1],self.w_feedback))
            output_arr[i,:]=np.dot(np.r_[1,leadout_input_arr[i,:],reserver_arr],self.w_output)

        for i in range(input_arr.shape[0],input_arr.shape[0]+after_step):
            reserver_arr=self.func(np.dot(reserver_arr,self.w_reserver)+np.dot(output_arr[i-1],self.w_feedback))
            output_arr[i,:]=np.dot(np.r_[1,np.zeros((self.input_size)),reserver_arr],self.w_output)

        return output_arr

    # input_size <= teach_size
    # input_size < teach_size の時、未来予測の学習を行う
    def learning(self,input_arr,teach_arr):
        reserver_params=np.empty(shape=(teach_arr.shape[0],input_arr.shape[1]+1+self.reserver_size))
        output_arr=np.empty(shape=(teach_arr.shape[0],self.output_size))

        leadout_input_arr=input_arr*self.leadout_input_scale
        reserver_arr=self.func(np.dot(input_arr[0,:],self.w_input))
        reserver_params[0,:]=np.r_[1,leadout_input_arr[0,:],reserver_arr]
        output_arr[0,:]=np.dot(reserver_params[0,:],self.w_output)
        print('solving')
        for i in range(1,input_arr.shape[0]):
            a=np.dot(input_arr[i,:],self.w_input)
            b=np.dot(reserver_arr,self.w_reserver)
            c=np.dot(output_arr[i-1],self.w_feedback)
            reserver_arr=a+b+c
            reserver_arr=self.func(reserver_arr)
            
            reserver_params[i,:]=np.r_[1,leadout_input_arr[i,:],reserver_arr]
            output_arr[i,:]=np.dot(reserver_params[i,:],self.w_output)

        if input_arr.shape[0] < teach_arr.shape[0]:
            after_step = teach_arr.shape[0] - input_arr.shape[0]
            for i in range(input_arr.shape[0],input_arr.shape[0]+after_step):
                reserver_arr=self.func(np.dot(reserver_arr,self.w_reserver)+np.dot(output_arr[i-1],self.w_feedback))
                reserver_params[i,:]=np.r_[1,np.zeros(self.input_size),reserver_arr]
                output_arr[i,:]=np.dot(reserver_params[i,:],self.w_output)

        print('calc w_output')
        X=reserver_params
        X_T=X.T
        D=teach_arr
        X_TX=np.dot(X_T,X)
        pinvX_TX=np.linalg.pinv(X_TX,rcond=self.rcond)
        pinvX_TXX_T=np.dot(pinvX_TX,X_T)
        pinvX_TXX_TD=np.dot(pinvX_TXX_T,D)
        # print(pinvX_TXX_TD.max(),pinvX_TXX_TD.shape,np.linalg.matrix_rank(pinvX_TXX_TD))
        # print(self.w_output.max(),self.w_output.shape,np.linalg.matrix_rank(self.w_output))
        self.w_output=pinvX_TXX_TD

    def rmse(predict_arr,true_arr):
        square=np.square(predict_arr-true_arr)
        l2_square=np.sum(square,axis=1)
        rmse=np.sqrt(np.mean(l2_square))
        return rmse
        
    def nrmse(predict_arr,true_arr):
        true_mean=np.ones(true_arr.shape)*np.mean(true_arr)
        rmse=ESN.rmse(predict_arr,true_arr)
        sigma=ESN.rmse(true_arr,true_mean)
        return rmse / sigma