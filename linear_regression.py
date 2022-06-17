import sys
from typing import Any, List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from termcolor import colored


class LinearRegression:
    def __init__(self, df: pd.DataFrame, lr: float, epochs: int, visualization: bool):
        self.df = df
        self.lr = lr
        self.ep = epochs
        self.vis = visualization

        self.b0 = 0.0
        self.b1 = 0.0
        self.frame = 100

        self.max_x = self.df['km'].max()
        self.max_y = self.df['price'].max()
        self.norm_x, self.norm_y = self.normalize()

        self.sst: Any = None
        self.sse_list: List[float] = list()
        self.mse_list: List[float] = list()
        self.line_params: Any = list()

        self.train_model()
        if self.vis:
            self.make_animation()
        self.print_calculations()

    def normalize(self) -> tuple:
        nx = np.array(self.df['km'] / self.df['km'].max())
        ny = np.array(self.df['price'] / self.df['price'].max())
        return nx, ny

    def train_model(self) -> None:
        for i in range(self.ep):
            self.gradient_descent()
            
            if self.vis:
                if not i % self.frame:
                    self.line_params.append([self.b0 * self.max_y, self.b1 * (self.max_y / self.max_x)])
                    self.sse_list.append(sum((self.df['price'] - self.predicted_coordinates) ** 2))
                    self.mse_list.append((sum((self.df['price'] - self.predicted_coordinates) ** 2)) / len(self.df))

        if self.vis:
            self.sst = sum((self.df['price'] - self.df['price'].mean()) ** 2)
            self.line_params = np.array(self.line_params)

    def gradient_descent(self) -> None:
        self.b0 = self.b0 - self.lr * sum(self.normalized_hypothesis - self.norm_y) / len(self.norm_y)
        self.b1 = self.b1 - self.lr * sum((self.normalized_hypothesis - self.norm_y) * self.norm_x) / len(self.norm_y)

    @property
    def normalized_hypothesis(self) -> np.ndarray:
        return self.b0 + self.b1 * self.norm_x

    @property
    def predicted_coordinates(self) -> pd.Series:
        return self.b0 * self.max_y + self.b1 * (self.max_y / self.max_x) * self.df['km']

    def make_animation(self) -> None:
        fig, ax = plt.subplots(dpi=100, num='ft_linear_regression')
        ax.scatter(self.df['km'], self.df['price'], color='navy', label='observations')
        line_width = np.array(range(int(self.df['km'].min()), int(self.df['km'].max())))
        ax.set_axisbelow(True)
        intercept, slope = self.line_params[0]
        reg_line = intercept + line_width * slope
        ln, = ax.plot(line_width, reg_line, color='red', label='linear regression')
        title = ax.text(0.1, 0.1, '', bbox={'facecolor': 'w', 'alpha': .5, 'pad': 4}, transform=ax.transAxes)

        def animator(frame: int) -> None:
            inter, slp = self.line_params[frame]
            line = inter + line_width * slp
            ln.set_data(line_width, line)

            analitics = f'iteration: {frame * self.frame}\n'
            analitics += f'theta0: {inter:.2f}\n'
            analitics += f'theta1: {slp:.4f}\n'
            analitics += f'mse = {int(self.mse_list[frame]):,}\n'.replace(',', ' ')
            analitics += f'sse = {int(self.sse_list[frame]):,}\n'.replace(',', ' ')
            analitics += f'r² =  {(1 - self.sse_list[frame]/self.sst):.4f}'
            title.set_text(analitics)

        _ = animation.FuncAnimation(fig, func=animator, frames=(self.ep // self.frame), interval=300)
        plt.grid()
        plt.xlabel('km')
        plt.ylabel('price')
        plt.title('model training')
        plt.legend()
        plt.show()

    def print_calculations(self) -> None:
        print('number of epochs: ', f'{self.ep}'.rjust(10))
        print('learning rate: ', f'{self.lr}'.rjust(13))
        print('theta0:', colored(f'{round(self.b0 * self.max_y, 2)}'.rjust(21), 'green'))
        print('theta1:', colored(f'{round(self.b1 * (self.max_y / self.max_x), 6)}'.rjust(21), 'green'))
        sst = sum((self.df['price'] - self.df['price'].mean()) ** 2)
        ssr = sum((self.predicted_coordinates - self.df['price'].mean()) ** 2)
        mse = ((sum((self.df['price'] - self.predicted_coordinates) ** 2)) / len(self.df))**.5
        print('standart error:', f'{mse:.2f}'.rjust(13))
        print('determination coef:', f'{(ssr / sst):.4f}'.rjust(9))

    def write_calculations(self) -> None:
        try:
            wr = pd.read_csv('weights.csv')
        except FileNotFoundError:
            sys.exit('need a valid file')
        wr.at[0, 'theta0'] = self.b0 * self.max_y
        wr.at[0, 'theta1'] = self.b1 * (self.max_y / self.max_x)
        wr.to_csv('weights.csv', index=False)
