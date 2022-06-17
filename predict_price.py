#!/usr/bin/python3

import sys

import numpy as np
import pandas as pd
from termcolor import colored

if __name__ == '__main__':
	if len(sys.argv) != 2:
		sys.exit('mileage is needed to predict the price of a car')
	try:
		value = float(sys.argv[1])
		if value < 0:
			raise ValueError
	except ValueError:
		sys.exit('the passed parameter must be a floating point value greater than 0')

	try:
		rd = pd.read_csv('weights.csv')
	except FileNotFoundError:
		sys.exit('need a valid file')

	if np.isnan(rd.at[0, 'theta0']) or np.isnan(rd.at[0, 'theta1']):
		sys.exit('need to train the model')

	try:
		price = rd.at[0, 'theta0'] + rd.at[0, 'theta1'] * value
		if price < 0:
			print(colored('data extrapolation. you may have to pay extra for the sale of the car', 'red'))
		print('estimated selling price of the car is $', colored(str(round(price, 2)), 'green'), sep='')
	except KeyError:
		sys.exit('need valid data')
