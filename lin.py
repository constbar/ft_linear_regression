#!/usr/bin/python3
import numpy as np  # ?
import matplotlib.animation as animation  # ?


import pandas as pd
import matplotlib.pyplot as plt

# try mypy
# import plotext as plt
# y = plt.sin() # sinusoidal signal
# plt.scatter(y)
# plt.title('Scatter Plot')
# plt.show()
# def key_values for multik
# try with -2/n

# can i use plt??
# other val of lr
# other val of b0 and b1

# make func of r2
# можно попробовать задать сразу большой шаг
# можно попробовать без минуса, для регресси по возврастанию + с большим шагом
# lr можно задавать как параметр
# молжно попробовать функицю нормализации без 2йки 37:00

# сделать инпут на имя файла, не хард код
# try old normilize


# self.b0 = 7900.660
# self.b1 = -0.019
# сделать флаг на отрисовку

class Linear_Regression: # -> ???
    # def __init__(self, lr=0.01, epochs=1001):
    def __init__(self, lr=0.0001, epochs=10000): # try other lr | lr=0.01 is ok
        self.df = pd.read_csv('data.csv')  # make try open
        self.b0 = 0.0
        self.b1 = 0.0
        self.lr = lr
        self.ep = epochs
        self.frame = 100

        self.max_x = self.df['km'].max()
        self.max_y = self.df['price'].max()
        self.norm_x, self.norm_y = self.normalize()


        self.sst = None
        self.sse_list = list()
        self.mse_list = list()
        self.ssr_list = list()
        self.line_params = list()

        self.train_model()
        self.make_animation()

        # print(self.derivative_b0)
        # print(self.derivative_b1)

    def train_model(self) -> None:

        for i in range(self.ep):
            self.gradient_descent()
            
            # if vizualize is ok
            if not i % self.frame:
                self.line_params.append([self.derivative_b0, self.derivative_b1])
                self.ssr_list.append(sum((self.count - self.df['price'].mean()) **2 ))
                self.sse_list.append(sum ( (self.df['price'] - self.count)** 2  ))
                self.mse_list.append((sum ( (self.df['price'] - self.count)** 2  ))  / 2 * len(self.df))
        
        self.sst = sum((self.df['price'] - self.df['price'].mean())**2)
        self.line_params = np.array(self.line_params)


    def normalize(self) -> tuple:
        nx = np.array(self.df['km'] / self.df['km'].max())
        ny = np.array(self.df['price'] / self.df['price'].max())
        return nx, ny

    @property
    def count(self) -> pd.core.series.Series:
        # print(type(self.derivative_b0 + self.derivative_b1 * self.df['km']))
        return self.derivative_b0 + self.derivative_b1 * self.df['km']

    @property
    def predicted_price(self) -> np.ndarray:
        return self.b0 + self.b1 * self.norm_x

    @property
    def derivative_b0(self) -> np.float64:
        return self.b0 * self.max_y

    @property
    def derivative_b1(self) -> np.float64:
        return self.b1 * (self.max_y / self.max_x)

    def gradient_descent(self) -> None:
        self.b0 = self.b0 - self.lr * sum(self.predicted_price - self.norm_y) / len(self.norm_y)
        self.b1 = self.b1 - self.lr * sum((self.predicted_price - self.norm_y) * self.norm_x) / len(self.norm_y)

    def make_animation(self) -> None:
        fig, ax = plt.subplots(dpi=100, num='ft_linear_regression')
        ax.scatter(self.df.km, self.df.price, color='navy', label='observations')
        line_width = np.array(range(int(self.df['km'].min()), int(self.df['km'].max())))
        ax.set_axisbelow(True)
        intercept, slope = self.line_params[0]
        reg_line = intercept + line_width * slope
        ln, = ax.plot(line_width, reg_line, color='red', label='linear regression')
        title = ax.text(0.1, 0.1, '', bbox={'facecolor':'w', 'alpha':0.5, 'pad':4}, transform=ax.transAxes)

        def animator(frame):
            intercept, slope = self.line_params[frame]
            reg_line = intercept + line_width * slope
            ln.set_data(line_width, reg_line)

            analitics = f'iteration: {frame * self.frame}\n'
            analitics += f'theta0: {intercept:.2f}\n'
            analitics += f'theta1: {slope:.4f}\n'
            analitics += f'mse = {int(self.mse_list[frame]):,}\n'.replace(',', ' ')
            analitics += f'sse = {int(self.sse_list[frame]):,}\n'.replace(',', ' ')
            analitics += f'r² =  {(self.ssr_list[frame] / self.sst):.4f}'            
            title.set_text(analitics)

        _ = animation.FuncAnimation(fig, func=animator, frames=(self.ep // self.frame), interval=300)
        plt.grid()
        plt.xlabel('km')
        plt.ylabel('price')
        plt.title('model training')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    Linear_Regression() 
    # if epochs loser tahn 100 -> error
    # catch ValueError -> if lr больше 1
    # if lr minus to libiya uhodit vniz
