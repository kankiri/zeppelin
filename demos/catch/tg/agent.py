import random

from zeppelin import Agent as BaseAgent
from zeppelin.utils import Transitions


class Agent(BaseAgent):
	def __init__(self, name, dimensions):
		super().__init__(name)
		self.num_actions = dimensions * 2
		self.actions = None
		self.memory = Transitions(['actions'], extra_keys=['perf'])
	
	def react(self, position, direction, time, reward=0, done=None):
		action = self.respond(direction, done)
		self.memory.store(action, perf=reward)
		if done:
			self.memory.forget()
		self.age += 1
		return {'action': action}
	
	def respond(self, direction, done):
		if self.memory.exp() >= 1:
			action, = self.memory[-1]
			if direction < 0:
				self.actions.remove(action)
		if not self.actions or done:
			self.actions = list(range(self.num_actions))
		return random.choice(self.actions)
