import numpy as np


class Transitions(dict):
	def __init__(self, cause_keys, effect_keys, extra_keys=[], maxlen=1e6, part=3):
		super().__init__()
		self.cekeys = cause_keys + effect_keys
		self.effect_keys = effect_keys
		self.maxlen = maxlen
		self.part = part
		
		self.update({key: [] for key in self.cekeys + extra_keys})
		self.exp = self[self.cekeys[0]].__len__
	
	def store(self, *args, **kwargs):
		if self.exp() == self.maxlen:
			self.forget_part(self.part)
		virgin = self.exp()
		for key, value in zip(self.cekeys, args):
			self._append(key, value, virgin)
		for key, value in kwargs.items():
			self._append(key, value, virgin)
			
	def _append(self, key, value, virgin):
		if key in self.effect_keys:
			self[key][-1:] = (value, value) if virgin else ()
		else:
			self[key].append(value)
	
	def forget(self):
		for key in self.cekeys:
			self[key].clear()
	
	def forget_part(self, part):
		limit = int(self.exp()/part)
		for key in self.cekeys:
			self[key][:limit] = []
	
	def __getitem__(self, key):
		if isinstance(key, slice) or isinstance(key, int):
			return self.draw(index=key)
		return super().__getitem__(key)
	
	def draw(self, index=slice(0, -1)):
		if isinstance(index, int):
			return (self[key][index] for key in self.cekeys)
		return (np.array(self[key][index]) for key in self.cekeys)
	
	def shuffled(self, length=None, index=slice(0, -1)):
		result = self.draw(index)
		maxlen = len(self[self.cekeys[0]][index])
		index = np.random.permutation(maxlen)[0:length]
		return (array[index] for array in result)
