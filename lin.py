#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt

# import plotext as plt
# y = plt.sin() # sinusoidal signal 
# plt.scatter(y)
# plt.title("Scatter Plot")
# plt.show()

# can i use plt??
# other val of lr
# other val of b0 and b1

# make func of r2


class Linear_Regression:
    """
    b0 - y-intercept
    b1 - slope
    ld - len of df
    """

    def __init__(self, lr=0.1, epochs=1000):
        self.b0 = 0
        # self.b0 = 8499.59 7900
        self.b1 = 0
        # self.b1 = -0.0214
        self.lr = lr

        self.df = pd.read_csv('data.csv')
        # print(self.df)
        # exit()
        self.ld = len(self.df) # maybe replace it with mean

        # print(self.df)
        # print(self.estimate_price(2))

        # print(self.df.head(5))

        # print(self.derivative_b0)
        # print(self.derivative_b1)

        for i in range(400):
            self.make_calculations()
            self.gradient_descent()
            # print()

        print(self.b1)
        print(self.b0)
        print()
        # self.df/['km'] = (self.df['km'] -self.df.min()) / (self.df['km'].max() -self.df.min())
        # print(self.df.head(1))

        # self.make_plot()

    def gradient_descent(self):
        # self.b0
        self.b0 += -self.derivative_b0 * self.lr
        self.b1 += -self.derivative_b1 * self.lr
        # self.make_calculations()
        # self.b0 += b00
        # self.b1 += b11
        # print(b00)
        # print(b11)


    def predict_price(self, km: pd.core.series.Series): # lol itisnt neccesary
        return self.b0 + self.b1 * km

    def make_calculations(self):
        self.df['predicted_price'] = self.predict_price(self.df['km'])
        self.df['residual'] = self.df['price'] - self.df['predicted_price']

    @property
    def derivative_b0(self):
        return -2 * self.df['residual'].mean()
        # return -2 * self.df['residual'].sum().mean()

    @property
    def derivative_b1(self, ):
        return -2 * (self.df.km * self.df['residual']).mean()

    def make_plot(self):
        print('here')
        plt.scatter(self.df.km, self.df.price, color='black')
        plt.show()


if __name__ == '__main__':
    Linear_Regression()
