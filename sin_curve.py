from matplotlib import pyplot as plt
import numpy as np
from rc import ESN

print('start')

step=0.1
# 1周期にかかるステップ数
step_1period = int(2 * np.pi / step)
dt=30 * step_1period

x_base=np.arange(0,80*np.pi,step)
value_base=np.sin(x_base).reshape((x_base.size,1))

# 学習
# 入力：0.5周期分
input_learning_step = int(step_1period * 1)
x_learning_input = x_base[:input_learning_step]
value_learning_input = value_base[:input_learning_step]
# 教師：10周期分
#teach_learning_step = step_1period * 5
x_learning_teach = x_base[dt:dt+input_learning_step]
value_learning_teach = value_base[dt:dt+input_learning_step]

# 予測
# 入力；0.5周期分
input_predict_step = int(step_1period * 10)
x_predict_input = x_base[:input_predict_step]
value_predict_input = value_base[:input_predict_step]
# 出力：15周期分
#output_predict_step = step_1period * 15
x_predict_output = x_base[dt:dt+input_predict_step]
value_predict_output = value_base[dt:dt+input_predict_step]
# 出力サイズ - 入力サイズ　が予測フレーム
#predict_frame = output_predict_step - input_predict_step


esn=ESN(1,1,300,feedback_scale=0,rcond=1e-5)
esn.learning(value_learning_input,value_learning_teach)
value_predicted=esn.solve(value_predict_input,0)
plt.plot(x_predict_output,value_predicted)
plt.plot(x_predict_output,value_predict_output)
plt.show()
rmse=ESN.rmse(value_predicted,value_predict_output)
print(rmse)