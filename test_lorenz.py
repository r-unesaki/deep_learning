from rc import ESN
from matplotlib import pyplot as plt
import numpy as np

# csv 読み込み
print('loading csv')
lorenz63 = np.loadtxt('lorenz63.csv',skiprows=1,delimiter=',')
dt=2000
# 学習
learn_input_t=4000
#learn_teach_t=6000
learning_input=lorenz63[0:learn_input_t,1:]
learning_teach=lorenz63[dt:dt+learn_input_t,1:]
# 予測
predict_input_t=6000
#predict_true_t=7000
predict_input=lorenz63[0:predict_input_t,1:]
predict_true=lorenz63[dt:dt+predict_input_t,1:]
#after_frame=predict_true_t-predict_input_t


esn=ESN(3,3,500,feedback_scale=0,spectrum_radius=0.95,rcond=1e-8,input_scale=0.1)
esn.learning(learning_input,learning_teach)
predicted=esn.solve(predict_input,0)

rmse=ESN.rmse(predicted,predict_true)
nrmse=ESN.nrmse(predicted,predict_true)
print("rmse:",rmse)
print("nrmse:",nrmse)

fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.plot(predict_true[:,0], predict_true[:,1], predict_true[:,2])
ax3d.set_xlabel('x')
ax3d.set_ylabel('y')
ax3d.set_zlabel('z')
fig3d.tight_layout()

fig3d_2 = plt.figure()
ax3d_2 = fig3d_2.add_subplot(111, projection='3d')
ax3d_2.plot(predicted[:,0], predicted[:,1], predicted[:,2])
ax3d_2.set_xlabel('x')
ax3d_2.set_ylabel('y')
ax3d_2.set_zlabel('z')
fig3d_2.tight_layout()

fig2d = plt.figure()
axx = fig2d.add_subplot(311)
axy = fig2d.add_subplot(312)
axz = fig2d.add_subplot(313)
axx.plot(lorenz63[dt:dt+predict_input_t,0], predict_true[:,0])
axx.plot(lorenz63[dt:dt+predict_input_t,0], predicted[:,0])
axx.set_ylabel('x')
axy.plot(lorenz63[dt:dt+predict_input_t,0], predict_true[:,1])
axy.plot(lorenz63[dt:dt+predict_input_t,0], predicted[:,1])
axy.set_ylabel('y')
axz.plot(lorenz63[dt:dt+predict_input_t,0], predict_true[:,2])
axz.plot(lorenz63[dt:dt+predict_input_t,0], predicted[:,2])
axz.set_ylabel('z')
axz.set_xlabel('t = '+str(lorenz63[predict_input_t,0]))
fig2d.tight_layout()

plt.show()


