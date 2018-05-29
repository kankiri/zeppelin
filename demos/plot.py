import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def read_csv(name, file_type):
		return name[:-(1 + len(file_type))], pd.read_csv(name, header=None, names=['rewards', 'episodes'])


def test(name, file_type, prefix=None, suffix=None, match=None):
	if not os.path.isfile(name) or not name.endswith(f'.{file_type}'):
		return False
	if prefix is not None and not name.startswith(prefix.strip()):
		return False
	if suffix is not None and not name.endswith(f'{suffix.strip()}.{file_type}'):
		return False
	if match is not None and match.strip() not in name:
		return False
	return True


def prepare(dataframe, size):
	dataframe.loc[1:, 'episodes'] = dataframe['episodes'].diff()[1:]
	dataframe['ratios'] = dataframe['rewards'] / dataframe['episodes']
	return dataframe.rolling(size).mean()


def average(dataframes):
	rewards = pd.concat((dataframe['rewards'] for dataframe in dataframes), axis=1).mean(axis=1)
	episodes = pd.concat((dataframe['episodes'] for dataframe in dataframes), axis=1).mean(axis=1)
	ratios = pd.concat((dataframe['ratios'] for dataframe in dataframes), axis=1).mean(axis=1)
	result = pd.concat((rewards, episodes, ratios), axis=1)
	result.columns = ['rewards', 'episodes', 'ratios']
	return result


def plot(data, columns):
	for name, dataframe in data.items():
		for column in columns:
			plt.plot(dataframe[column], label=f'{name}/{column}')
	plt.legend()
	plt.show(block=True)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Plot the learning performance of agents')
	parser.add_argument('-t', '--type', help='Required file type', default='csv')
	parser.add_argument('-p', '--prefix', help='Required filename prefix string')
	parser.add_argument('-s', '--suffix', help='Required filename suffix string')
	parser.add_argument('-m', '--match', help='Required filename match string')
	parser.add_argument('-g', '--group', help='Group by this filename string slice')
	parser.add_argument('-w', '--window', help='Mean window size for smoothing', default=5)
	parser.add_argument('-a', '--average', help='Plot mean of data only', action='store_true', default=False)
	parser.add_argument('-r', '--ratio', help='Plot rewards per episode only', action='store_true', default=False)
	parser.add_argument('-e', '--episodes', help='Plot number of episodes only', action='store_true', default=False)
	args = parser.parse_args()

	data = dict(
		read_csv(name, args.type) for name in os.listdir('.')
		if test(name, args.type, args.prefix, args.suffix, args.match)
	)
	data = {name: prepare(dataframe, int(args.window)) for name, dataframe in data.items()}
	if args.average:
		data = {'average': average(data.values())}

	columns = ['ratios'] if args.ratio else (['episodes'] if args.episodes else ['rewards'])
	plot(data, columns)
