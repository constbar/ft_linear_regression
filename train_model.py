#!/usr/bin/python3

import sys
import argparse
import pandas as pd
from linear_regression import LinearRegression


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='prediction the price of a car by using a \
        linear function train with a gradient descent algorithm')

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
    parser.add_argument('-vis', '--visualize',
                        action='store_true',
                        help='show a linear regression plot')

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        sys.exit('need a valid file')

    if args.lr is None:
        args.lr = 0.1
    if args.ep is None:
        args.ep = 2000

    if args.lr <= 0 or args.lr >= 1:
        sys.exit('need a valid learning rate')
    if not isinstance(args.ep, int) or args.ep <= 0:
        sys.exit('the number of epochs must be a positive integer')
    elif args.ep < 100 and args.visualize:
        sys.exit('for visualization, the number of epochs must be from 100')

    lin_reg = LinearRegression(df, args.lr, args.ep, args.visualize)
    lin_reg.write_calculations()
    