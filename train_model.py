#!/usr/bin/python3

import argparse
import sys

import pandas as pd

from linear_regression import LinearRegression

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='prediction the price of a car by using a \
        linear function train with a gradient descent algorithm')
    parser.add_argument('data', type=str, help='data with observations')
    parser.add_argument('-lr', type=float, default=0.01, required=False,
                        help='set learning rate')
    parser.add_argument('-ep', type=int, default=6000, required=False,
                        help='set number of epochs')
    parser.add_argument('-vis', '--visualize', action='store_true',
                        help='show a linear regression plot')
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        sys.exit('need a valid file')

    if args.lr <= 0 or args.lr >= 1:
        sys.exit('need a valid learning rate')
    elif args.ep < 1:
        sys.exit('the number of epochs must be a positive integer')
    elif args.ep < 100 and args.visualize:
        sys.exit('for visualization, the number of epochs must be from 100')

    lin_reg = LinearRegression(df, args.lr, args.ep, args.visualize)
    lin_reg.write_calculations()
