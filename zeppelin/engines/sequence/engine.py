from random import choices, seed

from . import config
from zeppelin import Engine as BaseEngine


class Engine(BaseEngine):
	def __init__(self, agents, dimensions=5, length=10):
		super().__init__(agents)
		self.observation_shapes = config.observation_shapes
		self.action_shapes = config.action_shapes
		self.action_shapes['action']['max'] = dimensions

		seed(0)
		self.sequence = choices(range(dimensions), k=length)
		self.actions = None
	
	def reset(self):
		self.actions = {name: [] for name in self.agents}
		return {name: {} for name in self.agents}
	
	def step(self, actions):
		for name in actions:
			self.actions[name].append(actions[name]['action'])
			done = len(self.actions[name]) == len(self.sequence)
		if done:
			rewards = {name: self.score(self.actions[name])
				for name in self.agents}
			self.reset()
		else:
			rewards = {name: 0 for name in self.agents}
		return {name: {
			'reward': rewards[name],
			'done': done
		} for name in self.agents}
	
	def score(self, action):
		return sum(self.sequence[i] == action[i]
			for i in range(len(action)))
