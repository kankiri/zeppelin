import numpy as np


def discount(rewards, gamma=1):
	rg = range(len(rewards))
	return np.fromiter((
		sum( [gamma**t * rewards[i+t] for t in rg[:-i or None]] )
		for i in rg
	), dtype=np.float32)