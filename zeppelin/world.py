import itertools


class World:
	def __init__(self, name, engine):
		self.name = name
		self.engine = engine
		self.agents = engine.agents
		self.observations = self.engine.reset()
	
	def run(self, limit=None):
		for i in itertools.count():
			if i == limit:
				break
			self.step()
	
	def step(self):
		actions = {}
		for name, observation in self.observations.items():
			action = self.agents[name].react(**observation)
			actions[name] = action
		self.observations = self.engine.step(actions)
