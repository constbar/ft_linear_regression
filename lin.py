#!/usr/bin/python3
import numpy as np  # ?
import matplotlib.animation as animation  # ?


import pandas as pd
import matplotlib.pyplot as plt

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


# self.b0 = 7900.660
# self.b1 = -0.019


class Linear_Regression:
    # def __init__(self, lr=0.01, epochs=1001):
    def __init__(self, lr=0.1, epochs=500):
        self.b0 = 0.0
        self.b1 = 0.0
        self.lr = lr
        self.ep = epochs
        self.frame = 100

        self.sst = None # need i?
        self.sse_list = list()
        self.mse_list = list()

        self.ssr_list = list()

        self.line_params = list()
        

        self.df = pd.read_csv('data.csv')
        self.train_model()

        # self.make_animation()
        # print result
        print(self.b0)
        print(self.b1)

        # self.make_plot()

    def train_model(self) -> None:
        self.normalize_km()
        self.make_calculation()

        for i in range(self.ep):
            # if not i % self.frame:
            #     self.line_params.append([self.b0, self.denormalize])

            #     self.ssr_list.append(sum((self.predicted_price - self.df['price'].mean() ) **2 ))

            #     self.sse_list.append(sum ( (self.df['price'] - self.predicted_price)** 2  ))
            #     self.mse_list.append((sum ( (self.df['price'] - self.predicted_price)** 2  ))   / 2 * len(self.df))
            self.gradient_descent()
            self.make_calculation()

        self.sst = sum((self.df['price'] - self.df['price'].mean())**2)
        # self.b1 = self.denormalize

        self.denormalize()

        self.line_params = np.array(self.line_params)

    def make_animation(self):  # maek animation
        fig, ax = plt.subplots(dpi=100, num='ft_linear_regression')
        ax.scatter(self.df.km, self.df.price, color='navy', label='observations')
        line_width = np.array(range(int(self.df['km'].min()), int(self.df['km'].max())))
        ax.set_axisbelow(True)

        intercept, slope = self.line_params[0]
        reg_line = intercept + line_width * slope
        ln, = ax.plot(line_width, reg_line, color='red', label='linear regression') # was [0]

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

            analitics += f'r² =  {(self.ssr_list[frame] / self.sst):.4f}\n' # ² -squared
            # analitics += f'r² =  {(1 - self.sse_list[frame] / self.sst):.4f}\n' # ² -squared
            # del last \n
            title.set_text(analitics)

        _ = animation.FuncAnimation(fig, func=animator, frames=self.ep // self.frame, interval=300)

        plt.grid()
        plt.xlabel('km')
        plt.ylabel('price')
        plt.title('model training')
        plt.legend()

        plt.show()

    def normalize_km(self) -> pd.core.series.Series:
        self.df['normalize_km'] = (self.df['km'] - self.df['km'].min()) /\
            (self.df['km'].max() - self.df['km'].min())



    # @property
    def denormalize(self):
        # return self.b1 / self.df.km.max()
        self.b1 = self.b1 / self.df.km.max()

    def make_calculation(self) -> pd.core.series.Series:
        self.df['predicted_price'] = self.predicted_price

    @property
    def predicted_price(self) -> pd.core.series.Series:
        return self.b0 + self.b1 * self.df['normalize_km']

    def gradient_descent(self):
        self.b1 = self.b1 - self.lr * self.derivative_b1
        self.b0 = self.b0 - self.lr * self.derivative_b0

    @property
    def derivative_b0(self) -> float:
        return (self.df['predicted_price'] - self.df['price']).mean()

    @property
    def derivative_b1(self) -> float:
        return (self.df['normalize_km'] * (self.df['predicted_price'] - self.df['price'])).mean()


if __name__ == '__main__':
    Linear_Regression()
    # pass

