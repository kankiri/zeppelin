import itertools
from multiprocessing import Process


class Universe:
	def __init__(self, worlds=None):
		self.worlds = worlds
	
	def run(self, limit=None):
		for i in itertools.count():
			if i == limit:
				break
			for name, world in self.worlds.items():
				world.step()
	
	@staticmethod
	def async(universe, limit=None):
		for world in universe.worlds:
			p = Process(target=world.run, args=(limit,))
			p.start()
