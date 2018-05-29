from importlib import import_module

from zeppelin import Universe, World


class School:
	engines = {
		'frozenlake': 'zeppelin.engines.frozenlake',
		'cartpole': 'zeppelin.engines.cartpole',
		'find': 'zeppelin.engines.find',
		'mountaincar': 'zeppelin.engines.mountaincar'
	}

	def __init__(self, curriculum):
		self.universe = Universe({world_name: self.populate(world_name, curriculum) for world_name in curriculum})

	def populate(self, world_name, curriculum):
		agents = {agent.name: agent for agent in curriculum[world_name]['agents']}
		engine_name = curriculum[world_name]['engine']
		Engine = import_module(self.engines[engine_name]).Engine
		args = curriculum[world_name].get('args', []); kwargs = curriculum[world_name].get('kwargs', {})
		return World(world_name, Engine(agents, *args, **kwargs))

	def run(self, limit=None):
		self.universe.run(limit=limit)

	def async(self, limit=None):
		self.universe.async(limit=limit)
