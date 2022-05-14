#!/usr/bin/python3

import sys
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# try mypy
# try pep8

# можно попробовать задать сразу большой шаг
# можно попробовать без минуса, для регресси по возврастанию + с большим шагом
 # try other lr | lr=0.01 is ok

class Linear_Regression:
    def __init__(self, df, lr, epochs, visualization):
        self.df = df
        self.lr = lr
        self.ep = epochs
        self.vi = visualization

        self.b0 = 0.0
        self.b1 = 0.0
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
        if self.vi:
            self.make_animation()


        # print fin results and calcualrtes the R^2
        # print(self.derivative_b0)
        # print(self.derivative_b1)
        # and write data to file
        print(self.derivative_b0, self.derivative_b1) # shole be more 1000 iterations

    def normalize(self) -> tuple:
        nx = np.array(self.df['km'] / self.df['km'].max())
        ny = np.array(self.df['price'] / self.df['price'].max())
        return nx, ny

    def train_model(self) -> None:
        for i in range(self.ep):
            self.gradient_descent()
            
            if self.vi:
                if not i % self.frame:
                    self.line_params.append([self.derivative_b0, self.derivative_b1]) # del
                    self.ssr_list.append(sum((self.predicted_coordinates - self.df['price'].mean())**2))
                    self.sse_list.append(sum((self.df['price'] - self.predicted_coordinates)** 2))
                    self.mse_list.append((sum((self.df['price'] - self.predicted_coordinates)** 2  )) / 2 * len(self.df))
        
        if self.vi:
            self.sst = sum((self.df['price'] - self.df['price'].mean())**2)
            self.line_params = np.array(self.line_params)

    def gradient_descent(self) -> None:
        self.b0 = self.b0 - self.lr * sum(self.predicted_price - self.norm_y) / len(self.norm_y)
        self.b1 = self.b1 - self.lr * sum((self.predicted_price - self.norm_y) * self.norm_x) / len(self.norm_y)

    @property
    def predicted_price(self) -> np.ndarray: # this not price -> predicted norm coef
        return self.b0 + self.b1 * self.norm_x

    @property
    def predicted_coordinates(self) -> pd.core.series.Series:
        return self.derivative_b0 + self.derivative_b1 * self.df['km']
        # return self.b0 * self.max_y + self.b1 * (self.max_y / self.max_x) * self.df['km'] # ??? () ???

    @property # del
    def derivative_b0(self) -> np.float64: # this is not derivative
        return self.b0 * self.max_y

    @property
    def derivative_b1(self) -> np.float64:
        return self.b1 * (self.max_y / self.max_x)


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
            analitics += f'r² =  {(self.ssr_list[frame] / self.sst):.4f}' # try count it thru other func
            title.set_text(analitics)

        _ = animation.FuncAnimation(fig, func=animator, frames=(self.ep // self.frame), interval=300)
        plt.grid()
        plt.xlabel('km')
        plt.ylabel('price')
        plt.title('model training')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ft_linear_regression - predicts the price of a car by \
        using a linear function train with a gradient descent algorithm')
    parser.add_argument('data',
                        type=str,
                        metavar='',
                        help='data with observations')
    parser.add_argument('-lr',
                        type=float,
                        metavar='',
                        required=False,
                        help='set learning rate')
    parser.add_argument('-ep',
                        type=int,
                        metavar='',
                        required=False,
                        help='set number of epochs')
    parser.add_argument('-v', '--visualize',
                        action='store_true',
                        help='show a linear regression plot')
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        sys.exit('need a valid file')

    if args.lr is None:
        args.lr = 0.0001 # maybe change val
    if args.ep is None:
        args.ep=10000 # change val

    if args.lr <= 0 or args.lr >= 1:
        sys.exit('try the valid learning rate')
    if not isinstance(args.ep, int) or args.ep <= 0:
        sys.exit('the number of epochs must be a positive integer')
    elif args.ep < 100 and args.visualize:
        sys.exit('for visualization, the number of epochs must be from 100')
    # catch ValueError -> if lr больше 1

    # print(args.data)
    # print('lr',args.lr)
    # print('ep',args.ep)
    # print(args.visualize)

    Linear_Regression(df, args.lr, args.ep, args.visualize)
