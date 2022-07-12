import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class Lorenz63():

    def __init__(self):
        self.msg = 'Lorenz63 simulation'
        # --- solve_ivp引数 --- #
        # 時刻tの間隔 t0からtfまで
        self.t0 = 0
        self.tf = 100
        self.t_span = (self.t0, self.tf)
        # 関数(Lorenz63)の初期値
        self.init_x = 1
        self.init_y = 1
        self.init_z = 1
        self.y0 = np.array([self.init_x, self.init_y, self.init_z])
        self.y0_ = np.array([self.init_x, self.init_y, self.init_z+0.001])
        # 数値計算手法
        self.method = 'RK45'  # 4次のルンゲクッタ 精度は5次相当
        # 連続値フラグ
        self.dense_output = True  # Falseだと最終値しか戻らない
        # 関数の初期値以外の設定値(Lorenz63パラメタ)
        self.s = 10
        self.r = 28
        self.b = 8/3
        self.args = (self.s, self.r, self.b)
        # 時刻tの刻み幅
        self.dt = 0.01
        self.t = np.arange(self.t0, self.tf, self.dt)

        # --- 出力 --- #
        self.fname = 'lorenz63.csv'

    def Process(self):
        print(self.msg)
        # --- 常微分方程式計算 --- #
        # scipy.integrate.solve_ivpでLorenz63モデルを計算
        self.solver = solve_ivp(fun=self.lorenz63, t_span=self.t_span, y0=self.y0, method='RK45',
                                t_eval=self.t, dense_output=self.dense_output, args=self.args)

        # z初期値ちょいずれ
        self.solver_ = solve_ivp(fun=self.lorenz63, t_span=self.t_span, y0=self.y0_, method='RK45',
                                t_eval=self.t, dense_output=self.dense_output, args=self.args)

        # 変数入れ替え
        self.xyz = self.solver.y
        self.xyz_ = self.solver_.y

        # --- csv保存 --- #
        self.arr_txyz = np.vstack((self.t, self.xyz))
        self.df_txyz = pd.DataFrame(self.arr_txyz.T, columns=['t', 'x', 'y', 'z'])  # 行と列を入れ替え
        self.df_txyz.to_csv(self.fname, index=False)

        # --- グラフ --- #
        # 3次元
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.plot(self.xyz[0], self.xyz[1], self.xyz[2])
        ax3d.set_xlabel('x')
        ax3d.set_ylabel('y')
        ax3d.set_zlabel('z')
        fig3d.tight_layout()
        # 2次元
        fig2d = plt.figure()
        axx = fig2d.add_subplot(311)
        axy = fig2d.add_subplot(312)
        axz = fig2d.add_subplot(313)
        axx.plot(self.t, self.xyz[0])
        axx.set_ylabel('x')
        axy.plot(self.t, self.xyz[1])
        axy.set_ylabel('y')
        axz.plot(self.t, self.xyz[2])
        axz.set_ylabel('z')
        axz.set_xlabel('t')
        fig2d.tight_layout()
        # 比較
        fig2d_ = plt.figure()
        axx_ = fig2d_.add_subplot(311)
        axy_ = fig2d_.add_subplot(312)
        axz_ = fig2d_.add_subplot(313)
        axx_.plot(self.t, self.xyz[0]-self.xyz_[0])
        axx_.set_ylabel('x')
        axy_.plot(self.t, self.xyz[1]-self.xyz_[1])
        axy_.set_ylabel('y')
        axz_.plot(self.t, self.xyz[2]-self.xyz_[2])
        axz_.set_ylabel('z')
        axz_.set_xlabel('t')
        fig2d_.tight_layout()
        # 表示
        plt.show()

    def lorenz63(self, t, arr_xyz, ss, rr, bb):
        # 時刻t-1の値
        x, y, z = arr_xyz
        # Lorenz63モデルのパラメタ
        s, r, b = ss, rr, bb

        # 時刻tの値
        dxdt = -s * (x-y)
        dydt = -x*z + r*x -y
        dzdt = x*y - b*z

        # arrayで戻す
        return np.array([dxdt, dydt, dzdt])


if __name__ == '__main__':
    proc = Lorenz63()
    proc.Process()