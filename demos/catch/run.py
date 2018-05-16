try:
	import zeppelin
except ImportError:
	from os.path import abspath, join, pardir
	import sys
	sys.path.append(abspath(join(sys.path[0], pardir, pardir)))

from zeppelin import Universe, World
from zeppelin.engines import Catch
from zeppelin.utils import tracking

import tg


def wrap(react, self, *args, **kwargs):
	action = react(*args, **kwargs)
	if self.age % 10000 == 9999:
		track = sum(self.memory['perf'])
		self.memory['perf'].clear()
		with open(self.name+'.pf', 'a') as file:
			file.write(str(track) + ' ')
	return action


if __name__ == '__main__':
	dimensions = 2
	worlds = {}
	
	agent = tracking.add(tg.Agent('tg-agent', dimensions), wrap)
	engine = Catch({agent.name: agent}, dimensions)
	world = World('world-tg', engine)
	worlds[world.name] = world
	
	universe = Universe(worlds)
	universe.run(limit=4e6)
